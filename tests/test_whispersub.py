import sys
from pathlib import Path
from unittest.mock import patch

import av.error
import pysubs2
import pytest
from faster_whisper.transcribe import Segment, Word

from whispersub import collect_videos, make_event, make_line_groups, merge_tokens, seg_to_events, validate_audio_tracks


def w(word: str, start: float = 0.0, end: float = 1.0, prob: float = 0.9) -> Word:
    return Word(start=start, end=end, word=word, probability=prob)


# ---------------------------------------------------------------------------
# merge_tokens
# ---------------------------------------------------------------------------


def test_merge_tokens_empty():
    """Empty input returns an empty list without raising."""
    result = merge_tokens([])
    assert result == []


def test_merge_tokens_single_word():
    """A single word is returned as-is."""
    result = merge_tokens([w(" hello", 0.0, 0.5, 0.9)])
    assert result == [Word(start=0.0, end=0.5, word=" hello", probability=0.9)]


def test_merge_tokens_no_continuations():
    """Words that all start with a space are returned unchanged."""
    result = merge_tokens([
        w(" one",   0.0, 0.3, 0.9),
        w(" two",   0.4, 0.7, 0.9),
        w(" three", 0.8, 1.1, 0.9),
    ])
    assert result == [
        Word(start=0.0, end=0.3, word=" one",   probability=0.9),
        Word(start=0.4, end=0.7, word=" two",   probability=0.9),
        Word(start=0.8, end=1.1, word=" three", probability=0.9),
    ]


def test_merge_tokens_simple_contraction():
    """A single continuation token is merged into the preceding word.

    " don" + "'t" → " don't": timing spans both tokens, probability is the minimum.
    """
    result = merge_tokens([
        w(" don", 0.0, 0.3, 0.9),
        w("'t",   0.3, 0.5, 0.8),
    ])
    assert result == [Word(start=0.0, end=0.5, word=" don't", probability=0.8)]


def test_merge_tokens_multiple_consecutive_continuations():
    """Several consecutive continuation tokens are all folded into one word.

    " wo" + "n" + "'t" → " won't", probability is the minimum across all three.
    """
    result = merge_tokens([
        w(" wo", 0.0, 0.2, 0.95),
        w("n",   0.2, 0.3, 0.85),
        w("'t",  0.3, 0.5, 0.7),
    ])
    assert result == [Word(start=0.0, end=0.5, word=" won't", probability=0.7)]


def test_merge_tokens_continuation_at_start():
    """A leading token with no space is kept as its own word; subsequent space-prefixed tokens are separate."""
    result = merge_tokens([
        w("'s",    0.0, 0.1, 0.9),
        w(" good", 0.2, 0.5, 0.8),
    ])
    assert result == [
        Word(start=0.0, end=0.1, word="'s",    probability=0.9),
        Word(start=0.2, end=0.5, word=" good", probability=0.8),
    ]


def test_merge_tokens_continuation_in_middle():
    """A continuation in the middle of a sequence only merges with the word immediately before it."""
    result = merge_tokens([
        w(" it", 0.0, 0.2, 0.9),
        w("'s",  0.2, 0.3, 0.8),
        w(" not", 0.4, 0.6, 0.95),
    ])
    assert result == [
        Word(start=0.0, end=0.3, word=" it's", probability=0.8),
        Word(start=0.4, end=0.6, word=" not",  probability=0.95),
    ]


# ---------------------------------------------------------------------------
# make_line_groups
# ---------------------------------------------------------------------------


def test_make_line_groups_empty():
    """Empty input yields nothing without raising."""
    assert list(make_line_groups([], 36)) == []


def test_make_line_groups_single_word():
    """A single word is returned in one group."""
    words = [w(" hello")]
    assert list(make_line_groups(words, 36)) == [words]


def test_make_line_groups_all_on_one_line():
    """Words whose total length is below max_line_width are returned in a single group.

    total_len=14, n_lines=1, target=14.  The last word reaches target but the
    soft-break guard (i < len-1) prevents a spurious second group.
    """
    words = [w(" one"), w(" two"), w(" three")]  # lens 4, 4, 6 → total 14
    assert list(make_line_groups(words, 20)) == [words]


def test_make_line_groups_soft_break_two_lines():
    """Soft break distributes words evenly across two lines.

    Words: " one"(4) " two"(4) " three"(6) " four"(5) " five"(5) " six"(4), max=20.
    total=28, n_lines=2, target=14.
    After " three": current_len=14 ≥ 14, i=2 < 5 → soft break.
    After " six":   current_len=14 ≥ 14, i=5 < 5? No → no break (last word guard).
    """
    words = [w(" one"), w(" two"), w(" three"), w(" four"), w(" five"), w(" six")]
    result = list(make_line_groups(words, 20))
    assert result == [words[:3], words[3:]]


def test_make_line_groups_soft_break_three_lines():
    """Soft break produces three balanced groups when total spans three target lengths.

    Nine words of width 3 each, max=10: total=27, n_lines=3, target=9.
    Breaks after index 2 (len=9, i < 8) and index 5 (len=9, i < 8).
    Index 8 reaches target but is the last word, so no trailing empty group.
    """
    words = [w(f" {c}{c}") for c in "abcdefghi"]  # " aa" … " ii", each len 3
    result = list(make_line_groups(words, 10))
    assert result == [words[0:3], words[3:6], words[6:9]]


def test_make_line_groups_hard_break():
    """A word that does not fit on the current line triggers a hard break.

    Words: " abcde"(6) " fghij"(6) " kl"(3), max=10.
    total=15, n_lines=2, target=7.5.
    " fghij" would bring current_len to 12 > 10 → hard break after [" abcde"].
    " kl" then fits on the new line alongside " fghij".
    """
    words = [w(" abcde"), w(" fghij"), w(" kl")]
    result = list(make_line_groups(words, 10))
    assert result == [words[:1], words[1:]]


def test_make_line_groups_oversized_word_not_hard_broken_from_empty():
    """A word wider than max_line_width is not broken off when it is first on a line.

    The hard-break condition requires current to be non-empty, so an oversized word
    that opens a fresh line is appended without triggering a break.  The soft-break
    then fires (13 ≥ target=4) before the final short word.

    Words: " xxxxxxxxxxxx"(13) " cd"(3), max=5.
    total=16, n_lines=4, target=4.
    """
    words = [w(" xxxxxxxxxxxx"), w(" cd")]
    result = list(make_line_groups(words, 5))
    assert result == [words[:1], words[1:]]


def test_make_line_groups_word_objects_preserved():
    """Word objects (start, end, probability) pass through into the groups unchanged.

    Words: " hello"(6) " world"(6) " foo"(4), max=12.
    total=16, n_lines=2, target=8.
    After " world": current_len=12 ≥ 8, i=1 < 2 → soft break.
    The returned lists contain the original Word instances, not copies.
    """
    word_a = Word(start=0.1, end=0.4, word=" hello", probability=0.95)
    word_b = Word(start=0.5, end=0.8, word=" world", probability=0.70)
    word_c = Word(start=0.9, end=1.2, word=" foo",   probability=0.85)
    result = list(make_line_groups([word_a, word_b, word_c], 12))
    assert result == [[word_a, word_b], [word_c]]
    assert result[0][0] is word_a
    assert result[0][1] is word_b
    assert result[1][0] is word_c


def test_make_line_groups_no_spurious_trailing_group():
    """The last word reaching the soft-break target does not produce an empty trailing group.

    Words: " abcd"(5) " efgh"(5), max=8.
    total=10, n_lines=2, target=5.
    After " abcd": current_len=5 ≥ 5, i=0 < 1 → soft break, yield [" abcd"].
    After " efgh": current_len=5 ≥ 5 but i=1 < 1 is False → no break.
    Final flush yields [" efgh"].  Result has exactly two groups.
    """
    words = [w(" abcd"), w(" efgh")]
    result = list(make_line_groups(words, 8))
    assert result == [words[:1], words[1:]]
    assert len(result) == 2


# ---------------------------------------------------------------------------
# make_event
# ---------------------------------------------------------------------------


def test_make_event_single_word():
    """A single-word card produces an event with that word's text and timing."""
    card = [[w(" hello", 1.0, 2.0)]]
    e = make_event(card, name="n")
    assert e.text == "hello"
    assert e.start == pysubs2.make_time(s=1.0)
    assert e.end == pysubs2.make_time(s=2.0)


def test_make_event_single_line_joined_and_stripped():
    """Words on one line are joined by concatenation and the leading space is stripped.

    Whisper tokens carry a leading space (" hello", " world"), so the join
    produces " hello world"; strip() removes the leading space.
    """
    card = [[w(" hello", 0.0, 0.5), w(" world", 0.6, 1.0)]]
    e = make_event(card, name="n")
    assert e.text == "hello world"


def test_make_event_two_lines_separated_by_ass_newline():
    """Two lines in a card are separated by the ASS hard-newline tag \\N."""
    card = [
        [w(" foo", 0.0, 0.5), w(" bar", 0.6, 1.0)],
        [w(" baz", 1.1, 1.5), w(" qux", 1.6, 2.0)],
    ]
    e = make_event(card, name="n")
    assert e.text == r"foo bar\Nbaz qux"


def test_make_event_three_lines():
    """Three lines produce two \\N separators."""
    card = [[w(" one")], [w(" two")], [w(" three")]]
    e = make_event(card, name="n")
    assert e.text == r"one\Ntwo\Nthree"


def test_make_event_timing_from_boundary_words():
    """Start comes from the first word of line 0; end from the last word of the last line."""
    card = [
        [w(" a", 1.0, 1.5), w(" b", 1.6, 2.0)],
        [w(" c", 2.1, 2.5), w(" d", 3.0, 4.5)],
    ]
    e = make_event(card, name="n")
    assert e.start == pysubs2.make_time(s=1.0)
    assert e.end == pysubs2.make_time(s=4.5)


def test_make_event_name_passed_through():
    """The name argument is stored verbatim in the SSAEvent name field."""
    card = [[w(" hi")]]
    e = make_event(card, name="seg:3 logp:-0.42 nsp:0.05 cr:1.10 t:0.0")
    assert e.name == "seg:3 logp:-0.42 nsp:0.05 cr:1.10 t:0.0"


def test_make_event_each_line_stripped_independently():
    """strip() is applied per line, not to the whole text block."""
    card = [
        [w(" leading space line")],       # join → " leading space line" → stripped
        [w(" second"), w(" line")],       # join → " second line" → stripped
    ]
    e = make_event(card, name="n")
    assert e.text == r"leading space line\Nsecond line"


# ---------------------------------------------------------------------------
# seg_to_events
# ---------------------------------------------------------------------------


def make_seg(
    words: list[Word] | None,
    *,
    start: float = 0.0,
    end: float = 1.0,
    text: str = "hello",
    avg_logprob: float = -0.5,
    no_speech_prob: float = 0.1,
    compression_ratio: float = 1.0,
    temperature: float = 0.0,
) -> Segment:
    return Segment(
        id=0, seek=0, start=start, end=end, text=text, tokens=[],
        avg_logprob=avg_logprob, compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob, words=words, temperature=temperature,
    )


def test_seg_to_events_no_words_uses_seg_timing_and_strips_text():
    """No-words path yields one event using seg.start/end with text stripped."""
    seg = make_seg(words=[], start=1.5, end=3.0, text="  Hello world.  ")
    events = list(seg_to_events(seg, seg_id=0, max_line_width=36, max_line_count=2))
    assert len(events) == 1
    assert events[0].start == pysubs2.make_time(s=1.5)
    assert events[0].end == pysubs2.make_time(s=3.0)
    assert events[0].text == "Hello world."


def test_seg_to_events_name_field_format():
    """The name encodes seg_id and all four segment quality metrics."""
    seg = make_seg(words=[], avg_logprob=-0.5, no_speech_prob=0.1,
                   compression_ratio=1.0, temperature=0.0)
    events = list(seg_to_events(seg, seg_id=7, max_line_width=36, max_line_count=2))
    assert events[0].name == "seg:7 logp:-0.50 nsp:0.10 cr:1.00 t:0.0"


def test_seg_to_events_single_line_single_card():
    """Words that fit on one line produce one event whose text is the joined, stripped line.

    Words: " one"(4) " two"(4) " three"(6), max_line_width=20.
    total=14, n_lines=1 → one group, one card, one event.
    Timing comes from the first and last word.
    """
    words = [w(" one", 0.0, 0.3), w(" two", 0.4, 0.7), w(" three", 0.8, 1.1)]
    seg = make_seg(words=words)
    events = list(seg_to_events(seg, seg_id=0, max_line_width=20, max_line_count=2))
    assert len(events) == 1
    assert events[0].text == "one two three"
    assert events[0].start == pysubs2.make_time(s=0.0)
    assert events[0].end == pysubs2.make_time(s=1.1)


def test_seg_to_events_two_lines_merged_into_one_card():
    """Two lines that fit within max_line_count=2 are merged into a single event.

    Uses the soft-break-two-lines scenario (total=28, max=20 → two groups).
    Both groups accumulate in one card before the full-card flush fires.
    """
    words = [w(" one"), w(" two"), w(" three"), w(" four"), w(" five"), w(" six")]
    seg = make_seg(words=words)
    events = list(seg_to_events(seg, seg_id=0, max_line_width=20, max_line_count=2))
    assert len(events) == 1
    assert events[0].text == r"one two three\Nfour five six"


def test_seg_to_events_overflows_into_two_cards():
    """Lines exceeding max_line_count spill into a second card.

    Nine " xx" words, max_line_width=10: make_line_groups yields three groups of
    three (total=27, n_lines=3, target=9).  max_line_count=2: groups 0–1 fill
    the first card and are flushed; group 2 is flushed by the trailing if-card.
    Timing: card 1 spans words aa→ff (0.0–6.0 s), card 2 spans gg→ii (6.1–9.0 s).
    """
    words = [
        w(" aa", 0.0, 1.0), w(" bb", 1.1, 2.0), w(" cc", 2.1, 3.0),
        w(" dd", 3.1, 4.0), w(" ee", 4.1, 5.0), w(" ff", 5.1, 6.0),
        w(" gg", 6.1, 7.0), w(" hh", 7.1, 8.0), w(" ii", 8.1, 9.0),
    ]
    seg = make_seg(words=words)
    events = list(seg_to_events(seg, seg_id=0, max_line_width=10, max_line_count=2))
    assert len(events) == 2
    assert events[0].text == r"aa bb cc\Ndd ee ff"
    assert events[0].start == pysubs2.make_time(s=0.0)
    assert events[0].end == pysubs2.make_time(s=6.0)
    assert events[1].text == "gg hh ii"
    assert events[1].start == pysubs2.make_time(s=6.1)
    assert events[1].end == pysubs2.make_time(s=9.0)


def test_seg_to_events_timing_comes_from_words_not_seg():
    """Event timing is taken from word boundaries, not from seg.start/seg.end."""
    words = [w(" hello", 1.5, 2.5), w(" world", 3.0, 4.5)]
    seg = make_seg(words=words, start=0.0, end=10.0)
    events = list(seg_to_events(seg, seg_id=0, max_line_width=20, max_line_count=2))
    assert len(events) == 1
    assert events[0].start == pysubs2.make_time(s=1.5)
    assert events[0].end == pysubs2.make_time(s=4.5)


def test_seg_to_events_all_cards_share_name():
    """Every card produced from the same segment carries an identical name.

    reconstruct_segments in transsub groups events back into segments by name,
    so all cards from one seg_to_events call must have the same name string.
    """
    words = [
        w(" aa", 0.0, 1.0), w(" bb", 1.1, 2.0), w(" cc", 2.1, 3.0),
        w(" dd", 3.1, 4.0), w(" ee", 4.1, 5.0), w(" ff", 5.1, 6.0),
        w(" gg", 6.1, 7.0), w(" hh", 7.1, 8.0), w(" ii", 8.1, 9.0),
    ]
    seg = make_seg(words=words)
    events = list(seg_to_events(seg, seg_id=0, max_line_width=10, max_line_count=2))
    assert len({e.name for e in events}) == 1


# ---------------------------------------------------------------------------
# collect_videos
# ---------------------------------------------------------------------------


def test_collect_videos_single_file(tmp_path):
    """A single video file path is returned as a one-element list."""
    f = tmp_path / "movie.mkv"
    f.touch()
    result = collect_videos([str(f)])
    assert result == [f]


def test_collect_videos_directory_scanned_recursively(tmp_path):
    """A directory input is scanned recursively; non-video files are ignored."""
    sub = tmp_path / "sub"
    sub.mkdir()
    video1 = tmp_path / "a.mp4"
    video2 = sub / "b.mkv"
    other = tmp_path / "readme.txt"
    for p in (video1, video2, other):
        p.touch()
    result = collect_videos([str(tmp_path)])
    assert sorted(result) == sorted([video1, video2])


def test_collect_videos_deduplicates(tmp_path):
    """The same file given twice (once directly, once via its directory) appears once."""
    f = tmp_path / "movie.mp4"
    f.touch()
    result = collect_videos([str(f), str(tmp_path)])
    assert result == [f]


def test_collect_videos_result_is_sorted(tmp_path):
    """Returned paths are in sorted order regardless of discovery order."""
    for name in ("c.mp4", "a.mkv", "b.avi"):
        (tmp_path / name).touch()
    result = collect_videos([str(tmp_path)])
    assert result == sorted(result)


def test_collect_videos_nonexistent_path_exits(tmp_path):
    """A path that does not exist causes sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        collect_videos([str(tmp_path / "missing.mp4")])
    assert exc_info.value.code == 1


def test_collect_videos_unrecognised_extension_exits(tmp_path):
    """A file with an unrecognised extension causes sys.exit(1)."""
    f = tmp_path / "document.pdf"
    f.touch()
    with pytest.raises(SystemExit) as exc_info:
        collect_videos([str(f)])
    assert exc_info.value.code == 1


def test_collect_videos_empty_directory(tmp_path):
    """An empty directory returns an empty list."""
    result = collect_videos([str(tmp_path)])
    assert result == []


# ---------------------------------------------------------------------------
# validate_audio_tracks
# ---------------------------------------------------------------------------


def _ffmpeg_error(msg: str) -> av.error.FFmpegError:
    """Construct a minimal FFmpegError with the given strerror."""
    exc = av.error.FFmpegError.__new__(av.error.FFmpegError)
    OSError.__init__(exc, msg)
    return exc


def test_validate_audio_tracks_single_track_no_request():
    """One track and no --audio-track requested: no error, no exit."""
    with patch("whispersub.list_audio_tracks", return_value=["#0 eng aac 48000Hz stereo"]):
        validate_audio_tracks([Path("a.mkv")], None)  # must not raise


def test_validate_audio_tracks_explicit_track_valid():
    """Requested track index within range: no error."""
    tracks = ["#0 eng aac", "#1 jpn ac3"]
    with patch("whispersub.list_audio_tracks", return_value=tracks):
        validate_audio_tracks([Path("a.mkv")], 1)


def test_validate_audio_tracks_no_tracks_exits():
    """A video with no audio tracks causes sys.exit(1)."""
    with patch("whispersub.list_audio_tracks", return_value=[]):
        with pytest.raises(SystemExit) as exc_info:
            validate_audio_tracks([Path("silent.mkv")], None)
    assert exc_info.value.code == 1


def test_validate_audio_tracks_multiple_tracks_no_selection_exits():
    """Multiple tracks without --audio-track causes sys.exit(1)."""
    tracks = ["#0 eng aac", "#1 jpn ac3"]
    with patch("whispersub.list_audio_tracks", return_value=tracks):
        with pytest.raises(SystemExit) as exc_info:
            validate_audio_tracks([Path("multi.mkv")], None)
    assert exc_info.value.code == 1


def test_validate_audio_tracks_out_of_range_exits():
    """--audio-track index beyond the available tracks causes sys.exit(1)."""
    with patch("whispersub.list_audio_tracks", return_value=["#0 eng aac"]):
        with pytest.raises(SystemExit) as exc_info:
            validate_audio_tracks([Path("a.mkv")], 5)
    assert exc_info.value.code == 1


def test_validate_audio_tracks_broken_file_exits():
    """A file that raises FFmpegError is reported and causes sys.exit(1)."""
    with patch("whispersub.list_audio_tracks", side_effect=av.error.InvalidDataError(1, "Invalid data")):
        with pytest.raises(SystemExit) as exc_info:
            validate_audio_tracks([Path("broken.mkv")], None)
    assert exc_info.value.code == 1


def test_validate_audio_tracks_all_errors_reported_together(capsys):
    """All pre-flight errors across multiple videos are collected before exiting."""
    def fake_tracks(video):
        if video.name == "broken.mkv":
            raise av.error.InvalidDataError(1, "Invalid data")
        return []

    with patch("whispersub.list_audio_tracks", side_effect=fake_tracks):
        with pytest.raises(SystemExit):
            validate_audio_tracks([Path("broken.mkv"), Path("silent.mkv")], None)

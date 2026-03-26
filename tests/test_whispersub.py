import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import av.error
import numpy as np
import pysubs2
import pytest
from faster_whisper.transcribe import Segment, Word

from whispersub import AssCommentPayload, _cuda_encode_works, _detect_drift, _offset_segment, _surround_mix_weights, _transcribe_with_retry, collect_videos, is_hallucination, load_model, main, make_event, make_line_groups, make_segment_comment, merge_tokens, rotate_backups, seg_to_events, set_script_info, validate_audio_tracks


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
# rotate_backups
# ---------------------------------------------------------------------------


def test_rotate_backups_no_existing_file(tmp_path):
    """When dest does not exist, nothing happens."""
    dest = tmp_path / "movie.de.ass"
    rotate_backups(dest, keep=3)
    assert not dest.exists()
    assert not (tmp_path / "movie.de.ass.bak").exists()


def test_rotate_backups_first_overwrite(tmp_path):
    """First overwrite moves current file to .bak."""
    dest = tmp_path / "movie.de.ass"
    dest.write_text("v1")
    rotate_backups(dest, keep=3)
    assert not dest.exists()
    assert (tmp_path / "movie.de.ass.bak").read_text() == "v1"


def test_rotate_backups_second_overwrite(tmp_path):
    """Second overwrite shifts .bak to .bak.1 and creates new .bak."""
    dest = tmp_path / "movie.de.ass"
    bak = tmp_path / "movie.de.ass.bak"
    dest.write_text("v2")
    bak.write_text("v1")
    rotate_backups(dest, keep=3)
    assert not dest.exists()
    assert (tmp_path / "movie.de.ass.bak").read_text() == "v2"
    assert (tmp_path / "movie.de.ass.bak.1").read_text() == "v1"


def test_rotate_backups_third_overwrite(tmp_path):
    """Third overwrite: .bak.1→.bak.2, .bak→.bak.1, dest→.bak."""
    dest = tmp_path / "movie.de.ass"
    dest.write_text("v3")
    (tmp_path / "movie.de.ass.bak").write_text("v2")
    (tmp_path / "movie.de.ass.bak.1").write_text("v1")
    rotate_backups(dest, keep=3)
    assert not dest.exists()
    assert (tmp_path / "movie.de.ass.bak").read_text() == "v3"
    assert (tmp_path / "movie.de.ass.bak.1").read_text() == "v2"
    assert (tmp_path / "movie.de.ass.bak.2").read_text() == "v1"


def test_rotate_backups_excess_deleted(tmp_path):
    """Fourth overwrite with keep=3: oldest backup is deleted."""
    dest = tmp_path / "movie.de.ass"
    dest.write_text("v4")
    (tmp_path / "movie.de.ass.bak").write_text("v3")
    (tmp_path / "movie.de.ass.bak.1").write_text("v2")
    (tmp_path / "movie.de.ass.bak.2").write_text("v1")
    rotate_backups(dest, keep=3)
    assert not dest.exists()
    assert (tmp_path / "movie.de.ass.bak").read_text() == "v4"
    assert (tmp_path / "movie.de.ass.bak.1").read_text() == "v3"
    assert (tmp_path / "movie.de.ass.bak.2").read_text() == "v2"
    assert not (tmp_path / "movie.de.ass.bak.3").exists()


def test_rotate_backups_keep_zero(tmp_path):
    """keep=0 deletes the existing file with no backups."""
    dest = tmp_path / "movie.de.ass"
    dest.write_text("v1")
    rotate_backups(dest, keep=0)
    assert not dest.exists()
    assert not (tmp_path / "movie.de.ass.bak").exists()


def test_rotate_backups_keep_one(tmp_path):
    """keep=1 keeps only .bak, deletes any numbered backups."""
    dest = tmp_path / "movie.de.ass"
    dest.write_text("v3")
    (tmp_path / "movie.de.ass.bak").write_text("v2")
    (tmp_path / "movie.de.ass.bak.1").write_text("v1")
    rotate_backups(dest, keep=1)
    assert not dest.exists()
    assert (tmp_path / "movie.de.ass.bak").read_text() == "v3"
    assert not (tmp_path / "movie.de.ass.bak.1").exists()


def test_rotate_backups_prints_rename(tmp_path):
    """When a console is provided, each rename is printed in Saved:-style."""
    from rich.console import Console
    from io import StringIO

    buf = StringIO()
    con = Console(file=buf, highlight=False)
    dest = tmp_path / "movie.de.ass"
    dest.write_text("v1")
    rotate_backups(dest, keep=3, console=con)
    output = buf.getvalue()
    assert "Backup:" in output
    assert "movie.de.ass" in output
    assert ".bak" in output


def test_rotate_backups_no_print_without_console(tmp_path, capsys):
    """Without a console argument, nothing is printed (backwards compat)."""
    dest = tmp_path / "movie.de.ass"
    dest.write_text("v1")
    rotate_backups(dest, keep=3)
    assert capsys.readouterr().out == ""


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


# ---------------------------------------------------------------------------
# --list-audio-tracks
# ---------------------------------------------------------------------------


def test_list_audio_tracks_flag_aggregates_by_config(capsys):
    """--list-audio-tracks groups identical configurations with filename-based labels."""
    single = ["#0 eng aac 48000Hz stereo"]
    dual = ["#0 eng aac 48000Hz stereo", "#1 jpn ac3 48000Hz stereo"]
    videos = [Path("a.mkv"), Path("b.mkv"), Path("c.mkv")]

    def fake_tracks(video):
        return dual if video.name == "a.mkv" else single

    with (
        patch("sys.argv", ["whispersub", ".", "--list-audio-tracks"]),
        patch("whispersub.collect_videos", return_value=videos),
        patch("whispersub.list_audio_tracks", side_effect=fake_tracks),
    ):
        main()

    out = capsys.readouterr().out
    # Two files share the single-track config — label shows first file + "1 other file"
    assert "b.mkv and 1 other file:" in out
    # One file has the dual-track config — label shows filename only
    assert "a.mkv:" in out
    # Each distinct track string appears exactly once per configuration group
    assert out.count("#1 jpn") == 1


def test_list_audio_tracks_flag_returns_without_transcribing(capsys):
    """--list-audio-tracks returns without loading the model."""
    tracks = ["#0 eng aac 48000Hz stereo"]
    with (
        patch("sys.argv", ["whispersub", "video.mkv", "--list-audio-tracks"]),
        patch("whispersub.collect_videos", return_value=[Path("video.mkv")]),
        patch("whispersub.list_audio_tracks", return_value=tracks),
    ):
        main()  # must return normally, not call sys.exit


def test_list_audio_tracks_flag_no_videos(capsys):
    """--list-audio-tracks with no matching videos prints a message and returns."""
    with (
        patch("sys.argv", ["whispersub", "empty/", "--list-audio-tracks"]),
        patch("whispersub.collect_videos", return_value=[]),
    ):
        main()


def test_list_audio_tracks_flag_ffmpeg_error(capsys):
    """--list-audio-tracks continues past unreadable files without exiting."""
    with (
        patch("sys.argv", ["whispersub", "bad.mkv", "--list-audio-tracks"]),
        patch("whispersub.collect_videos", return_value=[Path("bad.mkv")]),
        patch("whispersub.list_audio_tracks", side_effect=av.error.InvalidDataError(1, "Invalid data")),
    ):
        main()  # must not raise


# ---------------------------------------------------------------------------
# set_script_info
# ---------------------------------------------------------------------------


def test_set_script_info_contains_version():
    """The X-Version field is set to the installed package version."""
    subs = pysubs2.SSAFile()
    info = MagicMock()
    info.language = "de"
    info.language_probability = 0.999
    info.all_language_probs = [("de", 0.999), ("en", 0.001)]
    info.duration = 120.0
    set_script_info(subs, info, Path("movie.mkv"))
    assert "X-Version" in subs.info
    # Should be a valid version string (not empty)
    assert len(subs.info["X-Version"]) > 0


def test_set_script_info_fields():
    """All expected X- fields are populated."""
    subs = pysubs2.SSAFile()
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.95
    info.all_language_probs = [("en", 0.95), ("fr", 0.05)]
    info.duration = 300.0
    set_script_info(subs, info, Path("test.mkv"))
    for field in ["X-Source", "X-Transcribed-At", "X-Language", "X-Languages",
                  "X-Duration", "X-Model", "X-Version", "X-Transcribe-Params"]:
        assert field in subs.info, f"{field} missing"
    assert subs.info["X-Source"] == "test.mkv"
    assert "en" in subs.info["X-Language"]


# ---------------------------------------------------------------------------
# _surround_mix_weights
# ---------------------------------------------------------------------------


def test_surround_mix_weights_standard_51():
    """Standard 5.1 layout: FC gets dominant weight, FL/FR get less, LFE/surrounds are zero."""
    channels = ["FL", "FR", "FC", "LFE", "BL", "BR"]
    weights = _surround_mix_weights(channels)

    assert weights.shape == (6,)
    assert abs(weights.sum() - 1.0) < 1e-6  # normalised

    fc_idx, fl_idx, fr_idx = channels.index("FC"), channels.index("FL"), channels.index("FR")
    lfe_idx, bl_idx, br_idx = channels.index("LFE"), channels.index("BL"), channels.index("BR")

    assert weights[fc_idx] > weights[fl_idx]   # centre louder than fronts
    assert weights[fl_idx] == pytest.approx(weights[fr_idx])  # symmetric
    assert weights[lfe_idx] == 0.0             # LFE excluded
    assert weights[bl_idx] == 0.0             # surround excluded
    assert weights[br_idx] == 0.0             # surround excluded


def test_surround_mix_weights_centre_only():
    """A layout with only an FC channel gets all the weight."""
    weights = _surround_mix_weights(["FC"])
    assert weights == pytest.approx([1.0])


def test_surround_mix_weights_unknown_channels_fallback():
    """When no known channels are present, falls back to equal mix rather than silence."""
    channels = ["SL", "SR"]  # side surrounds - not in the weight table
    weights = _surround_mix_weights(channels)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(weights[1])  # equal mix


def test_surround_mix_weights_mix_arithmetic():
    """Weighted mix of a known frame produces the expected mono output."""
    # 3-channel layout: FL=0.3, FR=0.3, FC=1.0 → raw sum=1.6 → normalised FC≈0.625, FL=FR≈0.1875
    channels = ["FL", "FR", "FC"]
    weights = _surround_mix_weights(channels)

    # One sample per channel: FL=0, FR=0, FC=1.0 → mono should equal the FC weight
    frame = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)  # (channels, samples)
    mono = (frame * weights[:, np.newaxis]).sum(axis=0)

    assert mono[0] == pytest.approx(weights[channels.index("FC")])


# ---------------------------------------------------------------------------
# _cuda_encode_works / load_model CUDA fallback
# ---------------------------------------------------------------------------


def test_cuda_encode_works_returns_true_when_encode_succeeds():
    """Smoke test passes when detect_language succeeds."""
    model = MagicMock()
    model.detect_language.return_value = ("en", 0.9, [("en", 0.9)])
    assert _cuda_encode_works(model) is True
    model.detect_language.assert_called_once()


def test_cuda_encode_works_returns_false_on_runtime_error():
    """Smoke test catches RuntimeError from missing CUDA libs."""
    model = MagicMock()
    model.detect_language.side_effect = RuntimeError("cublas64_12.dll not found or cannot be loaded")
    assert _cuda_encode_works(model) is False


def test_load_model_falls_back_to_cpu_when_encode_fails():
    """load_model retries with device='cpu' when the CUDA smoke test fails."""
    gpu_model = MagicMock()
    cpu_model = MagicMock()

    # First WhisperModel() call returns gpu_model; second returns cpu_model
    with (
        patch("whispersub.WhisperModel", side_effect=[gpu_model, cpu_model]) as mock_cls,
        patch("whispersub._cuda_encode_works", return_value=False),
        patch("whispersub.console"),
    ):
        result = load_model(max_threads=None)

    assert result is cpu_model
    # Second call should explicitly request CPU
    assert mock_cls.call_count == 2
    _, kwargs = mock_cls.call_args_list[1]
    assert kwargs["device"] == "cpu"


def test_load_model_keeps_gpu_when_encode_succeeds():
    """load_model returns the GPU model when the smoke test passes."""
    gpu_model = MagicMock()

    with (
        patch("whispersub.WhisperModel", return_value=gpu_model),
        patch("whispersub._cuda_encode_works", return_value=True),
        patch("whispersub.console"),
    ):
        result = load_model(max_threads=None)

    assert result is gpu_model


def test_load_model_reraises_unrelated_runtime_errors():
    """RuntimeErrors not about missing libs propagate instead of falling back."""
    with (
        patch("whispersub.WhisperModel", side_effect=RuntimeError("out of memory")),
        patch("whispersub.console"),
        pytest.raises(RuntimeError, match="out of memory"),
    ):
        load_model(max_threads=None)


# ---------------------------------------------------------------------------
# is_hallucination
# ---------------------------------------------------------------------------


def test_is_hallucination_exact_match():
    """Known hallucination string is detected."""
    assert is_hallucination("Субтитры создавал DimaTorzok") is True


def test_is_hallucination_case_insensitive():
    """Matching ignores case differences."""
    assert is_hallucination("субтитры создавал DIMATORZOK") is True


def test_is_hallucination_trailing_punctuation():
    """Trailing sentence punctuation is stripped before matching."""
    assert is_hallucination("다음 영상에서 만나요.") is True
    assert is_hallucination("다음 영상에서 만나요!") is True


def test_is_hallucination_leading_whitespace():
    """Leading/trailing whitespace is stripped before matching."""
    assert is_hallucination("  Субтитры создавал DimaTorzok  ") is True


def test_is_hallucination_normal_text():
    """Normal dialogue is not flagged."""
    assert is_hallucination("Büro ist wie Achterbahnfahren") is False


def test_is_hallucination_empty_string():
    """Empty string is not a hallucination."""
    assert is_hallucination("") is False


def test_is_hallucination_korean():
    """Korean YouTube outro hallucination is detected."""
    assert is_hallucination("다음 영상에서 만나요") is True


# ---------------------------------------------------------------------------
# make_segment_comment
# ---------------------------------------------------------------------------


def _make_seg(text: str = "hello world", start: float = 1.0, end: float = 3.0, words=None):
    """Helper to build a Segment with optional word-level data."""
    if words is None:
        words = [w("hello", 1.0, 2.0), w(" world", 2.0, 3.0)]
    return Segment(
        id=0, seek=0, start=start, end=end, text=text, tokens=[],
        avg_logprob=-0.3, compression_ratio=1.2, no_speech_prob=0.01,
        words=words, temperature=0.0,
    )


def test_make_segment_comment_basic():
    """A segment with words produces a Comment event with seg_id and words."""
    seg = _make_seg()
    comment = make_segment_comment(seg, seg_id=5)
    assert comment is not None
    assert comment.type == "Comment"
    payload = json.loads(comment.text)
    assert payload["seg_id"] == 5
    assert len(payload["words"]) == 2
    assert payload["filtered"] is None


def test_make_segment_comment_no_words_returns_none():
    """A segment without word timestamps returns None."""
    seg = _make_seg(words=[])
    assert make_segment_comment(seg, seg_id=0) is None


def test_make_segment_comment_filtered_hallucination():
    """When filtered is set, the comment carries the filter reason."""
    seg = _make_seg(text="Субтитры создавал DimaTorzok")
    comment = make_segment_comment(seg, seg_id=2, filtered="hallucination")
    assert comment is not None
    assert comment.type == "Comment"
    payload = json.loads(comment.text)
    assert payload["seg_id"] == 2
    assert payload["filtered"] == "hallucination"
    assert len(payload["words"]) == 2


def test_make_segment_comment_filtered_preserves_timing():
    """Filtered comments still carry the segment's start/end timing."""
    seg = _make_seg(start=10.5, end=12.0)
    comment = make_segment_comment(seg, seg_id=0, filtered="hallucination")
    assert comment is not None
    assert comment.start == pysubs2.make_time(s=10.5)
    assert comment.end == pysubs2.make_time(s=12.0)


# ---------------------------------------------------------------------------
# _offset_segment
# ---------------------------------------------------------------------------


def test_offset_segment_shifts_timestamps():
    """Segment and word timestamps are shifted by the given offset."""
    seg = make_seg(
        words=[w(" hello", 1.0, 2.0, 0.9), w(" world", 2.5, 3.5, 0.8)],
        start=1.0, end=3.5,
    )
    shifted = _offset_segment(seg, 10.0)
    assert shifted.start == pytest.approx(11.0)
    assert shifted.end == pytest.approx(13.5)
    assert shifted.words[0].start == pytest.approx(11.0)
    assert shifted.words[0].end == pytest.approx(12.0)
    assert shifted.words[1].start == pytest.approx(12.5)
    assert shifted.words[1].end == pytest.approx(13.5)


def test_offset_segment_no_words():
    """Segment without words still gets shifted timestamps."""
    seg = make_seg(words=None, start=5.0, end=8.0)
    shifted = _offset_segment(seg, 100.0)
    assert shifted.start == pytest.approx(105.0)
    assert shifted.end == pytest.approx(108.0)
    assert shifted.words is None


def test_offset_segment_preserves_other_fields():
    """Non-timestamp fields are preserved through the offset."""
    seg = make_seg(words=[w(" hi", 0.0, 1.0, 0.95)], text=" hi",
                   avg_logprob=-0.3, no_speech_prob=0.05)
    shifted = _offset_segment(seg, 5.0)
    assert shifted.text == " hi"
    assert shifted.avg_logprob == pytest.approx(-0.3)
    assert shifted.no_speech_prob == pytest.approx(0.05)
    assert shifted.words[0].probability == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# _transcribe_with_retry
# ---------------------------------------------------------------------------


def _mock_progress():
    """Return a mock Progress whose console.print is a no-op."""
    progress = MagicMock()
    progress.console.print = MagicMock()
    return progress


def _collect_retry(audio, model, kwargs=None, *, language=None, progress=None):
    """Run _transcribe_with_retry and split results into kept and discarded lists.

    Returns (kept, discarded) where kept is a list of Segments yielded with
    reason=None, and discarded is a list of (Segment, reason) tuples.
    """
    if kwargs is None:
        kwargs = {}
    if progress is None:
        progress = _mock_progress()
    kept = []
    discarded = []
    for seg, reason in _transcribe_with_retry(audio, model, kwargs, language=language, progress=progress):
        if reason is None:
            kept.append(seg)
        else:
            discarded.append((seg, reason))
    return kept, discarded


def _fake_transcribe(pass_segments: list[list[Segment]]):
    """Return a model mock whose transcribe() yields a different segment list per call.

    Each call to model.transcribe() pops the next list from pass_segments and
    returns (iter(segments), info_mock).
    """
    model = MagicMock()
    call_idx = [0]

    def side_effect(clip, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        segs = pass_segments[idx] if idx < len(pass_segments) else []
        info = MagicMock()
        return iter(segs), info

    model.transcribe.side_effect = side_effect
    return model


def test_retry_no_drift():
    """When no gap exceeds the threshold, all segments are yielded with no retry."""
    segs = [
        make_seg(words=None, start=0.0, end=5.0, text="one"),
        make_seg(words=None, start=6.0, end=10.0, text="two"),
        make_seg(words=None, start=12.0, end=15.0, text="three"),
    ]
    model = _fake_transcribe([segs])
    audio = np.zeros(20 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language=None)

    assert len(kept) == 3
    assert len(discarded) == 0
    assert model.transcribe.call_count == 1
    assert kept[0].start == pytest.approx(0.0)
    assert kept[2].end == pytest.approx(15.0)


def test_retry_resets_on_drift():
    """A gap > threshold triggers a retry; second pass fills in the missing region."""
    # Pass 1: two segments, then a 60s gap → drift detected after seg at 10.0
    pass1 = [
        make_seg(words=None, start=0.0, end=5.0, text="seg one"),
        make_seg(words=None, start=6.0, end=10.0, text="seg two"),
        make_seg(words=None, start=70.0, end=75.0, text="drifted"),  # drift: gap=60s
    ]
    # Pass 2: starts from offset=10.0, produces segments in the previously-missed region
    pass2 = [
        make_seg(words=None, start=1.0, end=5.0, text="seg three"),    # 10+1=11 in original
        make_seg(words=None, start=8.0, end=12.0, text="seg four"),   # 10+8=18 in original
    ]
    model = _fake_transcribe([pass1, pass2])
    audio = np.zeros(120 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language=None)

    assert model.transcribe.call_count == 2
    assert len(kept) == 4
    assert kept[0].start == pytest.approx(0.0)
    assert kept[1].end == pytest.approx(10.0)
    assert kept[2].start == pytest.approx(11.0)
    assert kept[3].end == pytest.approx(22.0)
    # The drifted segment is recorded
    assert len(discarded) == 1
    assert discarded[0][0].text == "drifted"
    assert discarded[0][1] == "gap"


def test_retry_skips_when_no_speech_before_gap():
    """When retry also starts with a gap, skip ahead instead of looping."""
    # Pass 1: first segment starts at 50s → gap from 0, drift detected, no yield
    pass1 = [
        make_seg(words=None, start=50.0, end=55.0, text="far away"),
    ]
    # Pass 2: starts from offset=50.0, produces normal output
    pass2 = [
        make_seg(words=None, start=0.0, end=3.0, text="seg one"),
        make_seg(words=None, start=4.0, end=8.0, text="seg two"),
    ]
    model = _fake_transcribe([pass1, pass2])
    audio = np.zeros(120 * 16_000, dtype=np.float32)
    progress = _mock_progress()

    kept, discarded = _collect_retry(audio, model, language=None, progress=progress)

    assert model.transcribe.call_count == 2
    assert len(kept) == 2
    assert kept[0].start == pytest.approx(50.0)
    assert kept[1].end == pytest.approx(58.0)
    # The skipped segment is recorded as discarded
    assert len(discarded) == 1
    assert discarded[0][1] == "gap"
    # Should have printed a "skipping" message
    skip_calls = [c for c in progress.console.print.call_args_list
                  if "no speech" in str(c)]
    assert len(skip_calls) == 1


def test_retry_empty_pass_terminates():
    """If transcribe returns no segments at all, the generator terminates cleanly."""
    model = _fake_transcribe([[]])
    audio = np.zeros(60 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language=None)

    assert kept == []
    assert discarded == []
    assert model.transcribe.call_count == 1


def test_retry_short_audio_skipped():
    """Audio shorter than 1 second produces no output."""
    model = _fake_transcribe([])
    audio = np.zeros(8000, dtype=np.float32)  # 0.5s

    kept, discarded = _collect_retry(audio, model, language=None)

    assert kept == []
    assert discarded == []
    assert model.transcribe.call_count == 0


def test_retry_multiple_drifts():
    """Multiple drift resets in sequence all recover correctly."""
    # Pass 1: one segment then drift
    pass1 = [
        make_seg(words=None, start=0.0, end=5.0, text="seg one"),
        make_seg(words=None, start=60.0, end=65.0, text="drifted"),  # drift
    ]
    # Pass 2: from offset=5.0, one segment then another drift
    pass2 = [
        make_seg(words=None, start=0.0, end=4.0, text="seg two"),
        make_seg(words=None, start=50.0, end=55.0, text="drifted again"),  # drift again
    ]
    # Pass 3: from offset=9.0, normal completion
    pass3 = [
        make_seg(words=None, start=0.0, end=3.0, text="seg three"),
    ]
    model = _fake_transcribe([pass1, pass2, pass3])
    audio = np.zeros(120 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language=None)

    assert model.transcribe.call_count == 3
    assert len(kept) == 3
    assert kept[0].start == pytest.approx(0.0)
    assert kept[1].start == pytest.approx(5.0)
    assert kept[2].start == pytest.approx(9.0)
    assert len(discarded) == 2
    assert discarded[0][0].text == "drifted"
    assert discarded[1][0].text == "drifted again"


def test_retry_offsets_words():
    """Word timestamps are offset-adjusted on retry passes."""
    pass1 = [
        make_seg(words=[w(" hi", 0.0, 2.0)], start=0.0, end=2.0, text="hi"),
        make_seg(words=None, start=50.0, end=55.0, text="drifted"),  # drift
    ]
    pass2 = [
        make_seg(words=[w(" there", 1.0, 3.0)], start=1.0, end=3.0, text="there"),
    ]
    model = _fake_transcribe([pass1, pass2])
    audio = np.zeros(60 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language=None)

    assert len(kept) == 2
    assert kept[0].words[0].start == pytest.approx(0.0)
    assert kept[1].words[0].start == pytest.approx(3.0)
    assert kept[1].words[0].end == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _detect_drift
# ---------------------------------------------------------------------------


def test_detect_drift_gap():
    """A gap exceeding _DRIFT_THRESHOLD is detected."""
    seg = make_seg(words=None, start=40.0, end=42.0, text="hello")
    assert _detect_drift(seg, 5.0, language="de", recent_texts=[]) == "gap"


def test_detect_drift_no_gap():
    """A small gap does not trigger drift."""
    seg = make_seg(words=None, start=8.0, end=10.0, text="hello")
    assert _detect_drift(seg, 5.0, language="de", recent_texts=[]) is None


def test_detect_drift_script_mismatch_cyrillic_in_german():
    """Cyrillic text in a German transcription is detected as script drift."""
    seg = make_seg(words=None, start=1.0, end=2.0, text="О, ты шесть.")
    assert _detect_drift(seg, 0.0, language="de", recent_texts=[]) == "script"


def test_detect_drift_script_mismatch_cjk_in_german():
    """CJK text in a German transcription is detected as script drift."""
    seg = make_seg(words=None, start=1.0, end=2.0, text="ご視聴ありがとうございました")
    assert _detect_drift(seg, 0.0, language="de", recent_texts=[]) == "script"


def test_detect_drift_script_mismatch_korean_in_english():
    """Korean text in an English transcription is detected as script drift."""
    seg = make_seg(words=None, start=1.0, end=2.0, text="다음 영상에서 만나요")
    assert _detect_drift(seg, 0.0, language="en", recent_texts=[]) == "script"


def test_detect_drift_script_mismatch_mixed_in_latin():
    """CJK characters embedded in Latin text are detected."""
    seg = make_seg(words=None, start=1.0, end=2.0, text="Die Frau hat ja こんにちは ein Problem.")
    assert _detect_drift(seg, 0.0, language="de", recent_texts=[]) == "script"


def test_detect_drift_no_script_check_for_non_latin_language():
    """Non-Latin script in a non-Latin language (e.g. Japanese) is not flagged."""
    seg = make_seg(words=None, start=1.0, end=2.0, text="ご視聴ありがとうございました")
    assert _detect_drift(seg, 0.0, language="ja", recent_texts=[]) is None


def test_detect_drift_no_script_check_for_none_language():
    """When language is None, script check is skipped."""
    seg = make_seg(words=None, start=1.0, end=2.0, text="ご視聴ありがとうございました")
    assert _detect_drift(seg, 0.0, language=None, recent_texts=[]) is None


def test_detect_drift_latin_in_latin_language_ok():
    """Normal Latin text in a Latin language is not flagged."""
    seg = make_seg(words=None, start=1.0, end=2.0, text="Büro ist wie Achterbahnfahren.")
    assert _detect_drift(seg, 0.0, language="de", recent_texts=[]) is None


def test_detect_drift_echo():
    """Same text appearing 3+ times in recent history triggers echo detection."""
    seg = make_seg(words=None, start=5.0, end=6.0, text="Ich bin gut gegangen.")
    recent = ["Ich bin gut gegangen.", "Ich bin gut gegangen.", "Ich bin gut gegangen."]
    assert _detect_drift(seg, 4.0, language="de", recent_texts=recent) == "echo"


def test_detect_drift_echo_below_threshold():
    """Same text appearing fewer than 3 times does not trigger echo."""
    seg = make_seg(words=None, start=5.0, end=6.0, text="Ich bin gut gegangen.")
    recent = ["Ich bin gut gegangen.", "Ich bin gut gegangen."]
    assert _detect_drift(seg, 4.0, language="de", recent_texts=recent) is None


def test_detect_drift_echo_different_text():
    """Repeated text that doesn't match the current segment is not flagged."""
    seg = make_seg(words=None, start=5.0, end=6.0, text="Something new.")
    recent = ["Repeated.", "Repeated.", "Repeated."]
    assert _detect_drift(seg, 4.0, language="de", recent_texts=recent) is None


def test_detect_drift_gap_takes_priority():
    """Gap detection fires before script or echo checks."""
    seg = make_seg(words=None, start=50.0, end=52.0, text="ご視聴ありがとうございました")
    assert _detect_drift(seg, 5.0, language="de", recent_texts=[]) == "gap"


# ---------------------------------------------------------------------------
# _transcribe_with_retry — script mismatch
# ---------------------------------------------------------------------------


def test_retry_resets_on_script_mismatch():
    """Non-Latin text in a Latin-language file triggers a retry."""
    pass1 = [
        make_seg(words=None, start=0.0, end=5.0, text="Guten Tag."),
        make_seg(words=None, start=6.0, end=8.0, text="ご視聴ありがとうございました"),  # script drift
    ]
    pass2 = [
        make_seg(words=None, start=0.0, end=4.0, text="Auf Wiedersehen."),
    ]
    model = _fake_transcribe([pass1, pass2])
    audio = np.zeros(60 * 16_000, dtype=np.float32)
    progress = _mock_progress()

    kept, discarded = _collect_retry(audio, model, language="de", progress=progress)

    assert model.transcribe.call_count == 2
    assert len(kept) == 2
    assert kept[0].text == "Guten Tag."
    assert kept[1].text == "Auf Wiedersehen."
    assert kept[1].start == pytest.approx(5.0)
    assert len(discarded) == 1
    assert discarded[0][1] == "script"
    assert "ご視聴" in discarded[0][0].text


# ---------------------------------------------------------------------------
# _transcribe_with_retry — echo detection
# ---------------------------------------------------------------------------


def test_retry_resets_on_echo():
    """Same text repeated 3+ times triggers a retry.

    Echo fires when recent_texts already contains _ECHO_THRESHOLD copies,
    so the 4th occurrence is the one that triggers it (3 already in history).
    """
    pass1 = [
        make_seg(words=None, start=0.0, end=2.0, text="Normal speech."),
        make_seg(words=None, start=3.0, end=4.0, text="Ich bin gut gegangen."),
        make_seg(words=None, start=5.0, end=6.0, text="Ich bin gut gegangen."),
        make_seg(words=None, start=7.0, end=8.0, text="Ich bin gut gegangen."),
        make_seg(words=None, start=9.0, end=10.0, text="Ich bin gut gegangen."),  # 4th: 3 in history → echo
    ]
    pass2 = [
        make_seg(words=None, start=0.0, end=3.0, text="Fresh decoder output."),
    ]
    model = _fake_transcribe([pass1, pass2])
    audio = np.zeros(60 * 16_000, dtype=np.float32)
    progress = _mock_progress()

    kept, discarded = _collect_retry(audio, model, language="de", progress=progress)

    assert model.transcribe.call_count == 2
    assert len(kept) == 5  # 4 from pass1 + 1 from pass2
    assert kept[0].text == "Normal speech."
    assert kept[4].text == "Fresh decoder output."
    assert len(discarded) == 1
    assert discarded[0][1] == "echo"
    assert discarded[0][0].text == "Ich bin gut gegangen."


def test_retry_script_mismatch_as_first_segment():
    """Non-Latin text as the very first segment skips ahead (like S02E10 Korean at 1:40)."""
    pass1 = [
        make_seg(words=None, start=0.0, end=2.0, text="나이탈리아"),
    ]
    pass2 = [
        make_seg(words=None, start=0.0, end=5.0, text="Das konnte ich nicht wissen."),
    ]
    model = _fake_transcribe([pass1, pass2])
    audio = np.zeros(60 * 16_000, dtype=np.float32)
    progress = _mock_progress()

    kept, discarded = _collect_retry(audio, model, language="de", progress=progress)

    assert model.transcribe.call_count == 2
    assert len(kept) == 1
    assert kept[0].text == "Das konnte ich nicht wissen."
    assert len(discarded) == 1
    assert discarded[0][1] == "script"


def test_retry_script_mismatch_retry_also_fails():
    """When retry also produces non-Latin as first segment, skip past it."""
    pass1 = [
        make_seg(words=None, start=0.0, end=5.0, text="Normaler Text."),
        make_seg(words=None, start=6.0, end=7.0, text="고춧가루"),
    ]
    pass2 = [
        make_seg(words=None, start=2.0, end=3.0, text="고춧가루"),
    ]
    pass3 = [
        make_seg(words=None, start=0.0, end=4.0, text="Weiter gehts."),
    ]
    model = _fake_transcribe([pass1, pass2, pass3])
    audio = np.zeros(60 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language="de", progress=_mock_progress())

    assert model.transcribe.call_count == 3
    assert len(kept) == 2
    assert kept[0].text == "Normaler Text."
    assert kept[1].text == "Weiter gehts."
    assert len(discarded) == 2
    assert all(d[1] == "script" for d in discarded)


def test_retry_echo_persists_across_retry_boundary():
    """recent_texts carries across retries so echo from before a gap still counts."""
    pass1 = [
        make_seg(words=None, start=0.0, end=2.0, text="Looping text."),
        make_seg(words=None, start=3.0, end=4.0, text="Looping text."),
        make_seg(words=None, start=50.0, end=52.0, text="Far away."),  # gap drift
    ]
    pass2 = [
        make_seg(words=None, start=0.0, end=2.0, text="Looping text."),  # 3rd, yields
        make_seg(words=None, start=3.0, end=4.0, text="Looping text."),  # 4th: echo
    ]
    pass3 = [
        make_seg(words=None, start=0.0, end=3.0, text="Finally different."),
    ]
    model = _fake_transcribe([pass1, pass2, pass3])
    audio = np.zeros(120 * 16_000, dtype=np.float32)
    progress = _mock_progress()

    kept, discarded = _collect_retry(audio, model, language="de", progress=progress)

    assert model.transcribe.call_count == 3
    assert len(kept) == 4  # 2 from pass1, 1 from pass2, 1 from pass3
    assert kept[3].text == "Finally different."
    # gap + echo discards
    assert len(discarded) == 2
    reasons = {d[1] for d in discarded}
    assert "gap" in reasons
    assert "echo" in reasons


def test_retry_cyrillic_triggers_script_mismatch():
    """Cyrillic in a German file triggers script drift."""
    pass1 = [
        make_seg(words=None, start=0.0, end=5.0, text="Vielen Dank."),
        make_seg(words=None, start=6.0, end=8.0, text="О, ты шесть."),
    ]
    pass2 = [
        make_seg(words=None, start=0.0, end=3.0, text="Herr Stromberg!"),
    ]
    model = _fake_transcribe([pass1, pass2])
    audio = np.zeros(60 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language="de", progress=_mock_progress())

    assert model.transcribe.call_count == 2
    assert len(kept) == 2
    assert kept[1].text == "Herr Stromberg!"
    assert len(discarded) == 1
    assert discarded[0][1] == "script"


def test_retry_no_script_check_for_japanese_language():
    """Japanese text in a Japanese-language transcription does not trigger drift."""
    segs = [
        make_seg(words=None, start=0.0, end=5.0, text="こんにちは"),
        make_seg(words=None, start=6.0, end=10.0, text="ありがとうございます"),
        make_seg(words=None, start=12.0, end=15.0, text="さようなら"),
    ]
    model = _fake_transcribe([segs])
    audio = np.zeros(20 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language="ja", progress=_mock_progress())

    assert model.transcribe.call_count == 1
    assert len(kept) == 3
    assert len(discarded) == 0


def test_retry_script_mismatch_at_same_position_does_not_loop():
    """Script drift at the start of every retry must not loop forever.

    Reproduces the bug where retry keeps producing a non-Latin segment at
    seg.start=0.0, so offset never advances (S01E07 end-credits loop).
    After a bounded number of retries the generator must terminate.
    """
    # Every pass produces the same non-Latin segment at position 0
    bad_seg = make_seg(words=None, start=0.0, end=2.0, text="ご視聴ありがとうございました")
    passes = [[bad_seg]] * 50  # more than enough to detect a loop
    model = _fake_transcribe(passes)
    audio = np.zeros(30 * 16_000, dtype=np.float32)  # 30s

    kept, discarded = _collect_retry(audio, model, language="de", progress=_mock_progress())

    assert len(kept) == 0
    # Should have terminated, not run all 50 passes
    assert model.transcribe.call_count < 20


def test_retry_script_drift_near_end_terminates():
    """Script drift in the last few seconds of audio terminates cleanly.

    Like S01E07: good German dialogue until 23:56, then non-Latin at 24:06
    in a 24:39 file. After skipping past it, < 1s remains and we stop.
    """
    pass1 = [
        make_seg(words=None, start=0.0, end=5.0, text="Normal German."),
        make_seg(words=None, start=6.0, end=8.0, text="マンナー"),  # script drift
    ]
    # Retry from 5.0, produces non-Latin at the end
    pass2 = [
        make_seg(words=None, start=3.0, end=4.0, text="ご視聴ありがとうございました"),
    ]
    model = _fake_transcribe([pass1, pass2])
    # Audio is only 10s — after skipping to ~8.0, less than 1s remains
    audio = np.zeros(10 * 16_000, dtype=np.float32)

    kept, discarded = _collect_retry(audio, model, language="de", progress=_mock_progress())

    assert len(kept) == 1
    assert kept[0].text == "Normal German."
    # 3 calls: initial, retry from 5.0 (bad seg), final from 9.0 (empty)
    assert model.transcribe.call_count <= 3

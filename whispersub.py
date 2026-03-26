#!/usr/bin/env python3
"""Transcribe videos to ASS subtitle files using faster-whisper."""

import os
# Must be set before faster_whisper/huggingface_hub are imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows: no symlinks without Developer Mode

import argparse
import dataclasses
import importlib.metadata
import enum
import itertools
import json
import math
import re
import signal
import sys
import tempfile
import types
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Final

import av
import numpy as np
import pysubs2
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo, Word
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.rule import Rule

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

console = Console(highlight=False)

_MODEL: Final = "large-v3-turbo"  # Whisper model name


class ColourBy(enum.Enum):
    """Which per-word attribute to use for background colour coding in console output."""

    PROBABILITY = "probability"  # Whisper confidence score (default)
    DURATION = "duration"        # duration in seconds

    def __str__(self) -> str:
        return self.value


_VIDEO_EXTENSIONS: Final = frozenset({
    ".avi", ".flv", ".m4v", ".mkv", ".mov",
    ".mp4", ".mpg", ".mpeg", ".ts", ".webm", ".wmv",
})


@dataclasses.dataclass(frozen=True, kw_only=True)
class TranscribeParams:
    """Parameters forwarded verbatim to WhisperModel.transcribe()."""

    task: str = "transcribe"
    language: str | None = None
    multilingual: bool = True
    word_timestamps: bool = True
    vad_filter: bool = False
    hallucination_silence_threshold: float | None = 2


_TRANSCRIBE_PARAMS: Final = TranscribeParams()


@dataclasses.dataclass(frozen=True, kw_only=True)
class WordRecord:
    """Word with timestamp data, for serialisation into ASS Comment events.

    Mirrors faster_whisper.Word (which is a NamedTuple and therefore not
    directly usable with dataclasses.asdict).
    """

    start: float          # word start position in seconds
    end: float            # word end position in seconds
    word: str             # word text including any leading space
    probability: float    # Whisper confidence score [0, 1]

    @classmethod
    def from_word(cls, w: Word) -> "WordRecord":
        return cls(start=w.start, end=w.end, word=w.word, probability=w.probability)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AssCommentPayload:
    """Payload serialised into every whispersub Comment: event in the ASS output.

    Only one Comment is emitted per segment. All fields beyond seg_id have
    defaults so that fields added in the future do not break readers of older files.
    """

    seg_id: int                                       # 0-based index of the segment in the transcription
    words: tuple[WordRecord, ...] = dataclasses.field(default_factory=tuple)  # merged word tokens
    filtered: str | None = None                       # None = kept; otherwise the reason it was excluded (e.g. "hallucination")


# Per-channel weights for dialogue extraction from surround tracks.
# Centre carries all dialogue in professionally mixed content; the front pair
# gets a lower weight to catch tracks where dialogue bleeds into L/R.
# LFE (bass effects) and surrounds are excluded entirely.
_SURROUND_MIX_WEIGHTS: Final = types.MappingProxyType({
    "FC": 1.0,  # centre - primary dialogue channel
    "FL": 0.3,  # front left
    "FR": 0.3,  # front right
})


def _surround_mix_weights(channel_names: list[str]) -> np.ndarray:
    """Return a normalised per-channel weight vector for surround-to-mono mixing.

    Channels absent from _SURROUND_MIX_WEIGHTS (LFE, surrounds, etc.) get
    weight 0. If no known channels are present the weights fall back to a flat
    equal mix so that no audio is silenced.
    """
    weights = np.array([_SURROUND_MIX_WEIGHTS.get(ch, 0.0) for ch in channel_names], dtype=np.float32)
    if weights.sum() == 0:
        weights = np.ones(len(channel_names), dtype=np.float32)
    return weights / weights.sum()


# Frequently hallucinated strings to filter out.  Matching is case-insensitive
# and ignores leading/trailing whitespace and punctuation so that minor Whisper
# variations (e.g. trailing period vs none) don't slip through.
_KNOWN_HALLUCINATIONS: Final = frozenset({
    "субтитры создавал dimatorzok",   # https://github.com/openai/whisper/discussions/2372
    "субтитры подогнал «симон»",      # "Subtitles by Simon" (Russian attribution)
    "продолжение следует",            # "To be continued" (Russian)
    "다음 영상에서 만나요",             # "See you in the next video" (Korean YouTube outro)
    "한글자막 by 박진희",              # "Korean subtitles by Park Jinhee" (Korean attribution)
    "ご視聴ありがとうございました",      # "Thank you for watching" (Japanese YouTube outro)
})


def _normalize_text(text: str) -> str:
    """Lower-case and strip whitespace + sentence-final punctuation for comparison."""
    return text.strip().lower().rstrip(".,!?;:。！？")


def is_hallucination(text: str) -> bool:
    """Return True if *text* matches a known Whisper hallucination."""
    return _normalize_text(text) in _KNOWN_HALLUCINATIONS

# ---------------------------------------------------------------------------
# GPU / CUDA helpers
# ---------------------------------------------------------------------------


def _register_nvidia_dll_directories() -> None:
    """Add NVIDIA pip-package DLL dirs to the Windows DLL search path.

    When CUDA libraries are installed via ``pip install nvidia-cublas-cu12`` etc.
    they land in ``site-packages/nvidia/*/bin/`` which is not on PATH.  Python 3.8+
    exposes ``os.add_dll_directory()`` so we register those dirs here before
    CTranslate2 tries to load them.
    """
    import site
    search_roots: list[str] = []
    try:
        search_roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        search_roots.append(user_site)
    for sp in search_roots:
        nvidia_dir = Path(sp) / "nvidia"
        if nvidia_dir.is_dir():
            for pkg_dir in nvidia_dir.iterdir():
                dll_dir = pkg_dir / "bin"
                if dll_dir.is_dir():
                    os.add_dll_directory(str(dll_dir))


# File discovery
# ---------------------------------------------------------------------------


def find_videos(root: Path) -> Iterator[Path]:
    """Recursively find all video files under root."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS:
            yield p


def collect_videos(inputs: list[str]) -> list[Path]:
    """Resolve inputs and return a deduplicated sorted list of video files.

    Files are included directly; directories are scanned recursively.
    Exits if any path does not exist.
    """
    videos: set[Path] = set()
    for i in inputs:
        p = Path(i).resolve()
        if not p.exists():
            console.print(f"[red]Error:[/red] '{p}' does not exist.")
            sys.exit(1)
        if p.is_dir():
            videos.update(find_videos(p))
        else:
            if p.suffix.lower() not in _VIDEO_EXTENSIONS:
                console.print(f"[red]Error:[/red] '{p.name}' is not a recognised video file.")
                sys.exit(1)
            videos.add(p)
    return sorted(videos)

# ---------------------------------------------------------------------------
# Console word rendering
# ---------------------------------------------------------------------------


def confidence_colour_hex(p: float) -> str:
    """Return the RGB hex colour string for a word confidence probability."""
    if p >= 0.9:
        return "#1be91b"  # green
    if p >= 0.6:
        return "#ffff00"  # yellow
    return "#da0b0b"      # red


def duration_colour_hex(seconds: float) -> str:
    """Return the RGB hex colour string for a word duration.

    Thresholds are engineering heuristics informed by general phonetic
    knowledge of word durations in fluent speech:

    < 0.05 s  - collapsed timestamp (Whisper/alignment artefact)
                Whisper's 20 ms resolution and DTW-based forced alignment
                (e.g. WhisperX) can produce near-zero-width segments on failure.

    ≤ 0.60 s  - normal
                Conversational English (~150-200 wpm) gives mean word durations
                of ~300-400 ms; 600 ms covers the large majority of tokens.

    ≤ 1.00 s  - long but plausible (phrase-final / emphatic)
                Phrase-final lengthening and stressed multi-syllable content
                words at prosodic boundaries can approach or reach 1 s.

    > 1.00 s  - almost certainly a bad timestamp
                Single words this long are extraordinary; the aligner has most
                likely absorbed a neighbouring pause into the word span.

    Sources:
    - Yuan, Liberman & Cieri (2006), "Towards an Integrated Understanding
        of Speaking Rate in Conversation", ICSLP 2006.
        https://languagelog.ldc.upenn.edu/myl/ldc/llog/icslp06_final.pdf
    """
    if seconds < 0.05:
        return "#808080"  # grey - suspiciously short, likely collapsed timestamp
    if seconds <= 0.6:
        return "#1be91b"  # green - normal
    if seconds <= 1.0:
        return "#ffff00"  # yellow - long but plausible
    return "#da0b0b"      # red - likely a bad timestamp


# ---------------------------------------------------------------------------
# Subtitle card construction
# ---------------------------------------------------------------------------


def merge_tokens(words: list[Word]) -> list[Word]:
    """Merge continuation tokens into the preceding word.

    Whisper uses GPT-2-style byte-pair encoding: a leading space marks a new
    word, so a token without one (e.g. "'est", "'t") is a continuation of the
    previous token. Merging them prevents splits like [c] ['est] across lines
    or cards.
    """
    if not words:
        return []
    merged = [words[0]]
    for w in words[1:]:
        if not w.word.startswith(" "):
            prev = merged[-1]
            merged[-1] = Word(
                start=prev.start,
                end=w.end,
                word=prev.word + w.word,
                probability=min(prev.probability, w.probability),
            )
        else:
            merged.append(w)
    return merged


def make_line_groups(words: list[Word], max_line_width: int) -> Iterator[list[Word]]:
    """Yield groups of words that fit within max_line_width, balanced for even line lengths.

    Rather than filling each line greedily, compute a target length of
    total_chars / estimated_lines and break softly once a line reaches it.
    This prevents lopsided cards like a full first line paired with a short tail.
    """
    if not words:
        return
    total_len = sum(len(w.word) for w in words)
    n_lines = max(1, math.ceil(total_len / max_line_width))
    target = total_len / n_lines

    current: list[Word] = []
    current_len = 0
    for i, word in enumerate(words):
        w = len(word.word)
        if current and current_len + w > max_line_width:
            # Hard break: word does not fit on the current line.
            yield current
            current = [word]
            current_len = w
            continue
        current.append(word)
        current_len += w
        # Soft break: reached target and more words remain.
        if current_len >= target and i < len(words) - 1:
            yield current
            current = []
            current_len = 0
    if current:
        yield current


def make_event(card: list[list[Word]], *, name: str) -> pysubs2.SSAEvent:
    """Render a card (group of lines) as a single SSAEvent."""
    start = card[0][0].start
    end = card[-1][-1].end
    lines = ["".join(w.word for w in line).strip() for line in card]
    return pysubs2.SSAEvent(
        start=pysubs2.make_time(s=start),
        end=pysubs2.make_time(s=end),
        text=r"\N".join(lines),
        name=name,
    )


def seg_to_events(
    seg: Segment,
    *,
    seg_id: int,
    max_line_width: int,
    max_line_count: int,
) -> Iterator[pysubs2.SSAEvent]:
    """Yield one SSAEvent per subtitle card, splitting long segments across multiple cards."""
    # nsp tends to be low for all segments and is rarely informative.
    name = f"seg:{seg_id} logp:{seg.avg_logprob:.2f} nsp:{seg.no_speech_prob:.2f} cr:{seg.compression_ratio:.2f} t:{seg.temperature:.1f}"
    if not seg.words:
        yield pysubs2.SSAEvent(
            start=pysubs2.make_time(s=seg.start),
            end=pysubs2.make_time(s=seg.end),
            text=seg.text.strip(),
            name=name,
        )
        return
    card: list[list[Word]] = []
    for line in make_line_groups(seg.words, max_line_width):
        card.append(line)
        if len(card) >= max_line_count:
            yield make_event(card, name=name)
            card = []
    if card:
        yield make_event(card, name=name)


def word_to_rich_text(w: Word, colour_by: ColourBy) -> str:
    """Render a word with a Rich background colour tag based on the chosen colour mode."""
    if colour_by is ColourBy.DURATION:
        bg = duration_colour_hex(w.end - w.start)
    else:
        bg = confidence_colour_hex(w.probability)
    # Use black text on bright backgrounds (green, yellow) for legibility regardless
    # of the terminal's default foreground colour.
    fg = "black" if bg in ("#1be91b", "#ffff00") else "white"
    return f"[{fg} on {bg}]{escape(w.word)}[/]"


def seg_to_rich_text(seg: Segment, colour_by: ColourBy) -> str:
    """Render segment text with per-word Rich background colour markup."""
    if not seg.words:
        return seg.text.strip()
    return "".join(word_to_rich_text(w, colour_by) for w in seg.words)

# ---------------------------------------------------------------------------
# Transcription and output assembly
# ---------------------------------------------------------------------------


def fmt_time(seconds: float) -> str:
    """Format seconds as hh:mm:ss.t for console display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def list_audio_tracks(video: Path) -> list[str]:
    """Return a human-readable description of each audio track in the file.

    Tracks are numbered by their audio-local index (0, 1, 2, ...).
    """
    with av.open(str(video)) as container:
        tracks = []
        for audio_idx, stream in enumerate(container.streams.audio):
            ctx = stream.codec_context
            lang = stream.metadata.get("language", "und")
            title = stream.metadata.get("title", "")
            layout = ctx.layout.name if ctx.layout else f"{ctx.channels}ch"
            profile = f" {ctx.profile}" if ctx.profile else ""
            label = f"#{audio_idx} {lang} {ctx.name}{profile} {ctx.sample_rate}Hz {layout}"
            if title:
                label += f' "{title}"'
            tracks.append(label)
        return tracks


def print_tracks(progress: Progress, tracks: list[str], selected: int) -> None:
    """Print all audio tracks, highlighting the selected one."""
    for audio_idx, track in enumerate(tracks):
        if audio_idx == selected:
            progress.console.print(f"  audio track: [cyan]{track}[/cyan] [green](transcribing)[/green]")
        else:
            progress.console.print(f"  audio track: [dim]{track}[/dim]")


def extract_audio(video: Path, stream_index: int = 0, *, progress: Progress) -> np.ndarray:
    """Decode the audio track at stream_index and return a mono float32 array at 16 kHz.

    16 kHz mono float32 is required by the Whisper API when passing a numpy array.
    For surround layouts the channels are kept intact during decoding and then
    mixed down using _SURROUND_MIX_WEIGHTS, giving centre a high weight and the
    front pair a lower weight so that dialogue-in-fronts tracks still work.
    Mono and stereo sources use libav's default downmix to a single channel.

    Each decoded chunk is mixed to mono and written immediately to a temp file as
    raw float32 bytes, then loaded in one shot via np.fromfile.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    try:
        with av.open(str(video)) as container:
            stream = container.streams.audio[stream_index]
            channel_names = [ch.name for ch in stream.codec_context.layout.channels]
            is_surround = "FC" in channel_names
            target_layout = stream.codec_context.layout if is_surround else "mono"
            resampler = av.AudioResampler(format="fltp", layout=target_layout, rate=16000)
            weights = _surround_mix_weights(channel_names) if is_surround else None

            def write_chunk(frame_array: np.ndarray) -> None:
                """Mix (channels, samples) down to mono and append to tmp."""
                mono = (frame_array * weights[:, np.newaxis]).sum(axis=0) if weights is not None else frame_array[0]
                mono.tofile(tmp)

            duration = container.duration / 1_000_000 if container.duration else None
            total_label = f"{int(duration / 60)} minutes" if duration else "unknown"
            task = progress.add_task("Remuxing audio:", total=duration, total_label=total_label)
            for frame in container.decode(stream):
                progress.update(task, completed=frame.time)
                for resampled in resampler.resample(frame):
                    write_chunk(resampled.to_ndarray())
            for resampled in resampler.resample(None):
                write_chunk(resampled.to_ndarray())
            progress.remove_task(task)

        tmp.close()
        return np.fromfile(tmp.name, dtype=np.float32)
    finally:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)


_DRIFT_THRESHOLD: Final = 30
"""Seconds of gap between segments before assuming the decoder has drifted."""

_ECHO_THRESHOLD: Final = 3
"""Number of times a segment text must repeat in recent history to be considered an echo."""

# Scripts that use Latin characters.  When the detected language uses one of
# these scripts, non-Latin characters (CJK, Cyrillic, Hangul, etc.) in a
# segment are treated as a sign of decoder drift.
_LATIN_SCRIPT_LANGUAGES: Final = frozenset({
    "af", "az", "bs", "ca", "cs", "cy", "da", "de", "en", "es", "et", "eu",
    "fi", "fr", "gl", "hr", "hu", "id", "is", "it", "jw", "la", "lt", "lv",
    "mg", "ms", "mt", "nl", "nn", "no", "oc", "pl", "pt", "ro", "sk", "sl",
    "sq", "sv", "sw", "tl", "tr", "vi",
})

_NON_LATIN_RE: Final = re.compile(
    r"[\u0400-\u04ff"   # Cyrillic
    r"\u3000-\u9fff"    # CJK Unified Ideographs + symbols
    r"\u3040-\u309f"    # Hiragana
    r"\u30a0-\u30ff"    # Katakana
    r"\uac00-\ud7af]"   # Hangul
)


def _offset_segment(seg: Segment, offset: float) -> Segment:
    """Return a copy of *seg* with all timestamps shifted forward by *offset* seconds."""
    words = None
    if seg.words:
        words = [dataclasses.replace(w, start=w.start + offset, end=w.end + offset) for w in seg.words]
    return dataclasses.replace(seg, start=seg.start + offset, end=seg.end + offset, words=words)


def _detect_drift(
    seg: Segment,
    last_end: float,
    *,
    language: str | None,
    recent_texts: list[str],
) -> str | None:
    """Return a short reason string if *seg* looks like decoder drift, else None."""
    # 1. Large gap between segments
    if seg.start - last_end > _DRIFT_THRESHOLD:
        return "gap"

    text = seg.text.strip()

    # 2. Non-Latin script in a Latin-script language
    if language in _LATIN_SCRIPT_LANGUAGES and _NON_LATIN_RE.search(text):
        return "script"

    # 3. Exact text repeated too many times recently
    if recent_texts.count(text) >= _ECHO_THRESHOLD:
        return "echo"

    return None


def _transcribe_with_retry(
    audio: np.ndarray,
    model: WhisperModel,
    kwargs: dict,
    *,
    language: str | None,
    progress: Progress,
) -> Iterator[tuple[Segment, str | None]]:
    """Yield ``(segment, reason)`` tuples, retrying on drift.

    *reason* is ``None`` for normal segments or a short string (``"gap"``,
    ``"script"``, ``"echo"``) for segments that triggered a decoder reset.
    Discarded segments are still yielded (with the reason) so callers can
    record them as comments in the output file.

    Drift is detected by any of:

    - A gap > _DRIFT_THRESHOLD seconds between consecutive segments.
    - Non-Latin characters in a Latin-language transcription (script mismatch).
    - The same segment text appearing _ECHO_THRESHOLD times in recent output.

    On drift the current pass is abandoned and a fresh transcription starts from
    the last known-good position.  To avoid infinite loops, if a retry produces
    no output before the same point the gap is skipped.
    """
    sample_rate = 16_000
    total_duration = len(audio) / sample_rate
    offset = 0.0  # cumulative offset into the original audio
    recent_texts: list[str] = []  # rolling window for echo detection

    while offset < total_duration:
        start_sample = int(offset * sample_rate)
        clip = audio[start_sample:]
        if len(clip) < sample_rate:  # less than 1 s remaining
            break

        segments, _info = model.transcribe(clip, **kwargs)
        last_end = 0.0  # end of most recent yielded segment (relative to clip)
        yielded_any = False

        for seg in segments:
            reason = _detect_drift(
                seg, last_end, language=language, recent_texts=recent_texts,
            )
            if reason:
                # Yield the discarded segment so it can be logged as a comment.
                yield _offset_segment(seg, offset), reason
                drift_point = offset + last_end
                if not yielded_any:
                    # Retry produced nothing before the same point — skip ahead.
                    # For gap drift seg.start is already large (> threshold).
                    # For script/echo seg.start can be 0, so use seg.end to
                    # guarantee we advance past the bad segment.
                    advance = seg.start if reason == "gap" else seg.end
                    skip_to = offset + advance
                    progress.console.print(
                        f"  [yellow]Drift ({reason}):[/yellow] no speech in "
                        f"{fmt_time(offset)}–{fmt_time(skip_to)}, skipping"
                    )
                    offset = skip_to
                else:
                    progress.console.print(
                        f"  [yellow]Drift ({reason}):[/yellow] resetting decoder at {fmt_time(drift_point)}"
                    )
                    offset = drift_point
                break  # abandon this pass, retry from new offset
            yield _offset_segment(seg, offset), None
            last_end = seg.end
            yielded_any = True
            text = seg.text.strip()
            recent_texts.append(text)
            if len(recent_texts) > 10:
                recent_texts.pop(0)
        else:
            # Generator exhausted normally — we're done.
            break


def transcribe(video: Path, model: WhisperModel, stream_index: int, *, progress: Progress) -> tuple[Iterable[Segment], TranscriptionInfo]:
    """Decode audio from video, then transcribe and return (segments, info)."""
    audio = extract_audio(video, stream_index, progress=progress)
    kwargs = {f.name: getattr(_TRANSCRIBE_PARAMS, f.name) for f in dataclasses.fields(_TRANSCRIBE_PARAMS)}
    segments, info = model.transcribe(audio, **kwargs)
    # Wrap in the drift-retry generator; the first pass's segments feed into it
    # but we need to call transcribe fresh on retry, so we pass audio + kwargs.
    # Re-do the initial call inside the generator for uniform handling.
    return _transcribe_with_retry(audio, model, kwargs, language=info.language, progress=progress), info


def set_script_info(subs: pysubs2.SSAFile, info: TranscriptionInfo, video: Path) -> None:
    """Write Whisper transcription metadata into [Script Info]."""
    # Top language probabilities - useful for spotting code-switching in multilingual content.
    top_langs = sorted(info.all_language_probs or [], key=lambda x: x[1], reverse=True)[:5]
    lang_summary = ", ".join(f"{lang} {p:.1%}" for lang, p in top_langs)
    subs.info["X-Source"] = video.name
    subs.info["X-Transcribed-At"] = datetime.now().isoformat(timespec="seconds")
    subs.info["X-Language"] = f"{info.language} ({info.language_probability:.1%})"
    subs.info["X-Languages"] = lang_summary
    subs.info["X-Duration"] = f"{fmt_time(info.duration)}"
    subs.info["X-Model"] = _MODEL
    subs.info["X-Transcribe-Params"] = repr(_TRANSCRIBE_PARAMS)


def make_segment_comment(seg: Segment, seg_id: int, *, filtered: str | None = None) -> pysubs2.SSAEvent | None:
    """Create a Comment: ASS event carrying word-level timestamps for a segment.

    The event is given the segment's start/end positions so it sorts alongside its
    Dialogue counterpart. Players ignore Comment: events entirely; they exist
    solely so tooling can read word timestamps back from the file.
    Returns None if the segment has no word timestamps.
    """
    if not seg.words:
        return None
    payload = AssCommentPayload(
        seg_id=seg_id,
        words=tuple(WordRecord.from_word(w) for w in seg.words),
        filtered=filtered,
    )
    return pysubs2.SSAEvent(
        start=pysubs2.make_time(s=seg.start),
        end=pysubs2.make_time(s=seg.end),
        text=json.dumps(dataclasses.asdict(payload), ensure_ascii=False),
        type="Comment",
    )


def format_segment_for_console(seg: Segment, colour_by: ColourBy) -> str:
    """Format a segment as a Rich-markup string showing its timestamp and colour-coded text."""
    return f"  [dim]{fmt_time(seg.start)}→{fmt_time(seg.end)}[/dim] {seg_to_rich_text(seg, colour_by)}"


def build_subs(
    segments: Iterable[tuple[Segment, str | None]],
    *,
    font_size: int,
    limit: int | None,
    colour_by: ColourBy,
    max_line_width: int,
    max_line_count: int,
    info: TranscriptionInfo,
    video: Path,
    progress: Progress,
) -> pysubs2.SSAFile:
    """Consume (segment, drift_reason) tuples and return a populated SSAFile."""
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1280"
    subs.info["PlayResY"] = "720"
    subs.styles["Default"].fontsize = font_size
    set_script_info(subs, info, video)
    task_progress = progress.add_task("Transcribing:", total=info.duration, total_label=f"{int(info.duration / 60)} minutes")
    for seg_id, (seg, drift_reason) in enumerate(itertools.islice(segments, limit)):
        seg.words = merge_tokens(seg.words or [])
        if drift_reason:
            progress.console.print(f"{format_segment_for_console(seg, colour_by)} [bold red](drift: {drift_reason})[/]")
            if comment := make_segment_comment(seg, seg_id, filtered=f"drift:{drift_reason}"):
                subs.append(comment)
            continue
        if is_hallucination(seg.text):
            progress.console.print(f"{format_segment_for_console(seg, colour_by)} [bold red](ignoring known hallucination)[/]")
            if comment := make_segment_comment(seg, seg_id, filtered="hallucination"):
                subs.append(comment)
            continue
        if comment := make_segment_comment(seg, seg_id):
            subs.append(comment)
        events = seg_to_events(
            seg,
            seg_id=seg_id,
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )
        for event in events:
            subs.append(event)
        progress.console.print(format_segment_for_console(seg, colour_by))
        progress.update(task_progress, completed=seg.end)
    progress.remove_task(task_progress)
    return subs

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Like ArgumentDefaultsHelpFormatter but skips None and False defaults."""
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default in (None, False):
            return action.help
        return super()._get_help_string(action)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe videos to ASS subtitles.", formatter_class=_HelpFormatter, add_help=False)
    parser.add_argument("path", nargs="+", help="Video files or directories to scan recursively")
    parser.add_argument("--audio-track", type=int, metavar="N", help="Audio track index to transcribe (required when a file has multiple tracks)")
    parser.add_argument("--list-audio-tracks", action="store_true", help="Show audio tracks for all input videos, grouped by configuration, and exit")
    parser.add_argument("--force", action="store_true", help="Overwrite existing subtitle files (default: skip)")
    parser.add_argument("--keep", type=int, default=3, metavar="N", help="Number of previous subtitle versions to keep when overwriting with --force")
    parser.add_argument("--output-dir", type=Path, metavar="DIR", help="Write subtitle files here instead of alongside each video")
    parser.add_argument("--colour-by", type=ColourBy, default=ColourBy.PROBABILITY, choices=list(ColourBy), help="Per-word background colour coding in console output")
    parser.add_argument("--font-size", type=int, default=48, metavar="N", help="Font size in a 1280×720 virtual canvas (scaled to actual screen resolution by the player)")
    parser.add_argument("--max-line-count", type=int, default=2, metavar="N", help="Max subtitle lines per card")
    parser.add_argument("--max-line-width", type=int, default=36, metavar="N", help="Max characters per subtitle line")
    parser.add_argument("--max-threads", type=int, metavar="N", help="Max CPU threads (default: all cores)")
    parser.add_argument("--limit", type=int, metavar="N", help="Stop after N subtitle segments per video")
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {importlib.metadata.version('whispersub')}")
    return parser.parse_args()


def validate_audio_tracks(videos: list[Path], requested: int | None) -> None:
    """Check audio track availability for all videos before any transcription begins.

    Exits with an error if any video has no audio tracks, requires --audio-track
    but it was not provided, or has a provided --audio-track that is out of range.
    All problems are reported together so the user can fix them in one go.
    """
    errors: list[str] = []
    for video in videos:
        try:
            tracks = list_audio_tracks(video)
        except av.error.FFmpegError as exc:
            errors.append(f"{video.name}: cannot read file ({exc.strerror})")
            continue
        if not tracks:
            errors.append(f"{video.name}: no audio tracks")
        elif requested is None and len(tracks) > 1:
            track_list = "\n".join(f"  {t}" for t in tracks)
            errors.append(f"{video.name}: {len(tracks)} audio tracks - use --audio-track N to select one\n{track_list}")
        elif requested is not None and not 0 <= requested < len(tracks):
            valid = ", ".join(str(i) for i in range(len(tracks)))
            errors.append(f"{video.name}: --audio-track {requested} is out of range (valid: {valid})")
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        sys.exit(1)


def rotate_backups(dest: Path, keep: int) -> None:
    """Rotate existing subtitle file into .bak backups, keeping at most *keep*.

    ``foo.de.ass`` → ``foo.de.ass.bak`` (most recent)
    ``foo.de.ass.bak`` → ``foo.de.ass.bak.1``
    ``foo.de.ass.bak.1`` → ``foo.de.ass.bak.2``
    Backups beyond *keep* are deleted.
    """
    if not dest.exists() or keep <= 0:
        if dest.exists() and keep <= 0:
            dest.unlink()
        return
    bak = Path(str(dest) + ".bak")

    def numbered(n: int) -> Path:
        return Path(f"{bak}.{n}")

    # Delete excess numbered backups (.bak counts as slot 1, .bak.1 as slot 2, etc.)
    for n in range(max(keep - 1, 1), keep + 100):
        p = numbered(n)
        if p.exists():
            p.unlink()
        else:
            break

    # Shift numbered backups up by one
    for n in range(keep - 2, 0, -1):
        src = numbered(n)
        if src.exists():
            src.rename(numbered(n + 1))

    # .bak → .bak.1
    if bak.exists():
        if keep > 1:
            bak.rename(numbered(1))
        else:
            bak.unlink()

    # dest → .bak
    dest.rename(bak)


def process_video(
    progress: Progress,
    video: Path,
    *,
    args: argparse.Namespace,
    model: WhisperModel,
) -> None:
    """Transcribe one video file and save the ASS subtitle file."""
    tracks = list_audio_tracks(video)
    track_index = args.audio_track if args.audio_track is not None else 0
    print_tracks(progress, tracks, track_index)

    segments, info = transcribe(video, model, track_index, progress=progress)
    dest_dir = args.output_dir if args.output_dir else video.parent
    dest = dest_dir / f"{video.stem}.{info.language}.ass"
    progress.console.print(f"  detected language: [cyan]{info.language}[/cyan] ({info.language_probability:.2%})")
    progress.console.print(f"  colour by: [cyan]{args.colour_by}[/cyan]")
    progress.console.print(f"  output: [cyan]{dest}[/cyan]")
    if not args.force and dest.exists():
        progress.console.print("  [yellow]Skipping:[/yellow] subtitle already exists.")
    else:
        if args.force and dest.exists():
            rotate_backups(dest, args.keep)
        subs = build_subs(
            segments,
            font_size=args.font_size,
            limit=args.limit,
            colour_by=args.colour_by,
            max_line_width=args.max_line_width,
            max_line_count=args.max_line_count,
            info=info,
            video=video,
            progress=progress,
        )
        subs.save(str(dest))
        progress.console.print(f"  [green]Saved:[/green] {dest}")


def cmd_list_audio_tracks(videos: list[Path]) -> None:
    """Implement --list-audio-tracks: print unique track configurations with file counts."""
    configs: dict[tuple[str, ...], list[Path]] = {}
    read_errors: list[str] = []
    if len(videos) > 10:
        progress = Progress(
            TextColumn("  [dim]{task.description}[/dim]"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        )
        scan_task = progress.add_task("Scanning:", total=len(videos))
        progress.start()
    else:
        progress = None
        scan_task = None
    for video in videos:
        try:
            key = tuple(list_audio_tracks(video))
        except av.error.FFmpegError as exc:
            read_errors.append(f"{video.name}: {exc.strerror}")
        else:
            configs.setdefault(key, []).append(video)
        if progress is not None:
            progress.advance(scan_task)
    if progress is not None:
        progress.stop()
    for tracks, group in sorted(configs.items(), key=lambda x: -len(x[1])):
        first = group[0].name
        others = len(group) - 1
        first_coloured = f"[cyan]{first}[/cyan]"
        label = first_coloured if others == 0 else f"{first_coloured} and {others} other {'file' if others == 1 else 'files'}"
        console.print(f"[bold]{label}:[/bold]")
        if not tracks:
            console.print("  [dim]no audio tracks[/dim]")
        for track in tracks:
            coloured = re.sub(r"(#\d+)", r"[yellow]\1[/yellow]", track)
            console.print(f"  {coloured}")
    if read_errors:
        error_label = "1 unreadable file" if len(read_errors) == 1 else f"{len(read_errors)} unreadable files"
        console.print(f"[red]{error_label}[/red]")
        for error in read_errors:
            console.print(f"  {error}")


def _cuda_encode_works(model: WhisperModel) -> bool:
    """Return True if the model can actually run a forward pass on its device.

    Some Windows installs detect CUDA at model-load time but lack runtime
    libraries (e.g. cublas64_12.dll) that are only exercised during inference.
    A tiny detect_language smoke-test catches this early so we can fall back
    to CPU.
    """
    try:
        model.detect_language(np.zeros(16000, dtype=np.float32))
        return True
    except RuntimeError:
        return False


def load_model(max_threads: int | None) -> WhisperModel:
    """Load the Whisper model, falling back to CPU if GPU/CUDA is unavailable."""
    console.print(Rule(f"Loading model: {_MODEL}"))
    if sys.platform == "win32":
        _register_nvidia_dll_directories()
    try:
        model = WhisperModel(_MODEL, compute_type="int8", cpu_threads=max_threads or 0)
        if not _cuda_encode_works(model):
            raise RuntimeError("CUDA runtime library not found or cannot be loaded")
        return model
    except RuntimeError as exc:
        if "not found or cannot be loaded" not in str(exc):
            raise
        console.print(
            "[yellow]Warning:[/yellow] GPU/CUDA unavailable, falling back to CPU.\n"
            "For GPU support: install CUDA 12 (developer.nvidia.com/cuda-downloads)"
            " or run: [bold]pip install whispersub\\[gpu][/bold]"
        )
        return WhisperModel(_MODEL, device="cpu", compute_type="int8", cpu_threads=max_threads or 0)


def run_transcription(videos: list[Path], *, args: argparse.Namespace, model: WhisperModel) -> None:
    """Transcribe all videos, showing a progress bar."""
    progress = Progress(
        TextColumn("  [dim]{task.description}[/dim]"),
        BarColumn(),
        TextColumn(
            "[progress.percentage]{task.percentage:>3.0f}%[/progress.percentage]"
            " [dim]of[/dim] "
            "[progress.percentage]{task.fields[total_label]}[/progress.percentage]"
        ),
        console=console,
    )
    # Register before inference so this handler wins over any SIGINT handler that
    # CTranslate2 installs - ensuring the cursor is restored even when Ctrl-C fires
    # inside a C extension before Python can raise KeyboardInterrupt.
    def _sigint_handler(sig: int, frame: object) -> None:
        console.show_cursor(True)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.raise_signal(signal.SIGINT)  # Exit with 130 (128 + SIGINT).

    signal.signal(signal.SIGINT, _sigint_handler)

    with progress:
        overall_progress = progress.add_task("Overall:", total=len(videos), total_label=f"{len(videos)} videos")
        for i, video in enumerate(videos, 1):
            progress.console.print(Rule(f"{i}/{len(videos)}: {video.name}"))
            try:
                process_video(progress, video, args=args, model=model)
            except av.error.FFmpegError as exc:
                progress.console.print(f"  [red]Error:[/red] {exc.strerror} — skipping.")
            progress.update(overall_progress, advance=1)


def main() -> None:
    args = parse_args()

    if args.output_dir and not args.output_dir.is_dir():
        console.print(f"[red]Error:[/red] --output-dir '{args.output_dir}' does not exist.")
        sys.exit(1)

    videos = collect_videos(args.path)
    if not videos:
        console.print("No video files found.")
        return

    if args.list_audio_tracks:
        cmd_list_audio_tracks(videos)
        return

    validate_audio_tracks(videos, args.audio_track)
    model = load_model(args.max_threads)
    run_transcription(videos, args=args, model=model)


if __name__ == "__main__":
    main()

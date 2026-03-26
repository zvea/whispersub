# whispersub

Transcribe video files to [ASS](https://fileformats.fandom.com/wiki/SubStation_Alpha) subtitle files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Whisper detects the spoken language automatically and supports [99 languages](https://github.com/openai/whisper#available-models-and-languages), including English, Spanish, French, German, Japanese, Chinese, Arabic, Hindi, and many more.

- NVIDIA GPU acceleration with automatic CPU fallback
- Batch mode: pass multiple video files or directory trees
- Surround-sound audio extraction (dialogue-channel aware)
- Word-level timestamps with balanced line breaking
- Per-word confidence colour coding in the terminal

## Requirements

Python 3.10+. Works on Linux, Windows, and macOS. No system FFmpeg needed — PyAV bundles its own. Supports MKV, MP4, AVI, MOV, WebM, TS, and other common video formats.

## Install

```bash
pip install whispersub
```

GPU acceleration (Linux/Windows — requires an NVIDIA GPU with CUDA 12):

```bash
pip install whispersub[gpu]
```

Without `[gpu]`, whispersub falls back to CPU automatically if CUDA is unavailable.

On first run, whispersub downloads the Whisper large-v3-turbo model (~800 MB) from Hugging Face and caches it locally.

## Usage

```bash
# Single file — writes movie.en.ass alongside the video
whispersub movie.mkv

# Explicit output directory
whispersub movie.mkv --output-dir ~/subs

# Whole directory, force overwrite
whispersub /media/shows --force

# File with multiple audio tracks — inspect, then pick one
whispersub series.mkv --list-audio-tracks
whispersub series.mkv --audio-track 2
```

## Options

| Option | Default | Description |
|---|---|---|
| `--audio-track N` | — | Audio track index (required if the file has multiple tracks) |
| `--list-audio-tracks` | off | Show audio tracks for all input videos, grouped by configuration, and exit |
| `--colour-by` | `probability` | Per-word terminal colour coding: `probability` or `duration` |
| `--font-size N` | `48` | Font size (1280×720 canvas; player scales to actual resolution) |
| `--force` | off | Overwrite existing subtitle files (keeps backups) |
| `--keep N` | `3` | Number of `.bak` copies to keep when overwriting |
| `--limit N` | — | Stop after N segments per video (useful for testing) |
| `--max-line-count N` | `2` | Maximum subtitle lines per card |
| `--max-line-width N` | `36` | Maximum characters per line |
| `--max-threads N` | all cores | CPU thread limit |
| `--output-dir DIR` | alongside video | Write all subtitle files to this directory |

## Output

Subtitle files are named `<stem>.<language>.ass`, e.g. `movie.en.ass`. The detected language comes from Whisper. Output is compatible with VLC, mpv, IINA, MPC-HC, and other players that support ASS/SSA subtitles.

We chose ASS over SRT for better-looking subtitles: font sizing scales correctly to any resolution, and line breaks are balanced for readability. It also allows us to preserve word-level timing, so the file can be post-processed or reformatted without re-transcribing.

## Licence

MIT

# whispersub

Transcribe video files to [ASS](https://fileformats.fandom.com/wiki/SubStation_Alpha) subtitle files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3-turbo).

- Batch mode: pass multiple files or directories to scan recursively
- Surround-sound audio extraction (dialogue-channel aware)
- Word-level timestamps with balanced line breaking
- Per-word confidence colour coding in the terminal

## Requirements

Python 3.9+. Works on Linux, Windows, and macOS. No system FFmpeg needed — PyAV bundles its own.

## Install

```bash
pip install whispersub
```

GPU acceleration (Linux/Windows — requires an NVIDIA GPU with CUDA 12):

```bash
pip install whispersub[gpu]
```

Without `[gpu]`, whispersub falls back to CPU automatically if CUDA is unavailable.

## Usage

```bash
# Single file — writes movie.en.ass alongside the video
whispersub movie.mkv

# Explicit output directory
whispersub movie.mkv --output-dir ~/subs

# Whole directory, force overwrite
whispersub /media/shows --force

# File with multiple audio tracks — list tracks first, then pick one
whispersub series.mkv                     # error lists available tracks
whispersub series.mkv --audio-track 2
```

## Options

| Option | Default | Description |
|---|---|---|
| `--audio-track N` | auto | Audio track index (required if the file has multiple tracks) |
| `--colour-by` | `probability` | Per-word terminal colour coding: `probability` or `duration` |
| `--font-size N` | `48` | Font size (1280×720 canvas; player scales to actual resolution) |
| `--force` | off | Overwrite existing subtitle files |
| `--limit N` | — | Stop after N segments per video (useful for testing) |
| `--max-line-count N` | `2` | Maximum subtitle lines per card |
| `--max-line-width N` | `36` | Maximum characters per line |
| `--max-threads N` | all cores | CPU thread limit |
| `--output-dir DIR` | alongside video | Write all subtitle files to this directory |

## Output

Subtitle files are named `<stem>.<language>.ass`, e.g. `movie.en.ass`. The detected language comes from Whisper.

Each ASS file contains hidden `Comment:` events carrying word-level timestamp data in JSON, so the raw timing can be recovered by tooling without re-transcribing.

## Licence

MIT

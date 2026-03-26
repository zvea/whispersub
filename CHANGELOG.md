# Changelog

## v1.5.0 — 2026-03-26

- Detect wrong-script output and decoder loops as signs of drift, expanding on the gap-based detection from v1.4.0.
- Keep up to 3 backup copies (`.bak*`) when overwriting with `--force`, configurable via `--keep`.
- Expand known hallucination table with Japanese, Russian, and Korean strings found in the wild.
- Fix infinite loop when drift detection triggers repeatedly at the same position.

## v1.4.0 — 2026-03-26

- Detect decoder drift in long audio and automatically retry from a fresh state, fixing transcriptions that silently stop producing output partway through.
- Normalize hallucination matching (case-insensitive, punctuation-tolerant) and add Korean YouTube outro string.

## v1.3.1 — 2026-03-26

- Fixed CPU fallback on Windows.

## v1.3.0 — 2026-03-26

- Replaced VAD pre-filter with `hallucination_silence_threshold`, fixing dropped speech over background music/noise. May increase hallucinations over non-speech audio.

## v1.2.1 — 2026-03-23

- Fixed audio track grouping and dropped bitrate from track labels in `--list-audio-tracks` output.

## v1.2.0 — 2026-03-23

- Added `--list-audio-tracks` flag to inspect available audio tracks before processing.

## v1.1.0 — 2026-03-23

- Broken or unreadable video files are now skipped gracefully instead of crashing the run.

## v1.0.1 — 2026-03-23

- Renamed project to **whispersub** and published to PyPI (`pip install whispersub`).

## v1.0.0 — 2026-03-23

- Initial public release.

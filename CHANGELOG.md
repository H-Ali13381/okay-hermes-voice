# Changelog

All notable changes to Okay Hermes Voice are tracked here.

## [0.1.1] - 2026-05-29

### Changed
- Split the wakeword daemon, interaction router, Hermes runtime, playback, visualization, and activation archive logic into focused modules while preserving the public CLI/facade entrypoints.
- Split the large voice conversation and interaction router test files into package-level test modules.

### Fixed
- Require sustained speech before starting command capture so single-block wake-tail/beep spikes do not produce empty command recordings.
- Reap launched visualization terminal processes so closed popup terminals do not remain as zombie children of the daemon.

### Verified
- `PYTHONPATH=src ~/.hermes/hermes-agent/venv/bin/python -m pytest -q`
- `~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --smoke-test`

## [0.1.0] - 2026-05-28

### Added
- Initial always-on Okay Hermes wakeword daemon package.
- User-level systemd service installer.
- Local activation archive support.
- Popup visualization and follow-up voice conversation support.

[0.1.1]: https://github.com/H-Ali13381/okay-hermes-voice/releases/tag/v0.1.1
[0.1.0]: https://github.com/H-Ali13381/okay-hermes-voice/releases/tag/v0.1.0

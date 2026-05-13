# Contributing

Contributions are welcome. Please keep the project local-first, explicit about
privacy boundaries, and compatible with standard Hermes Agent installations.

Before opening a PR:

```bash
python -m py_compile src/okay_hermes_voice/wakeword_daemon.py src/okay_hermes_voice/voice_activation_popup.py
PYTHONPATH=src python -m pytest -q
```

Avoid committing model weights, activation audio, local config files, credentials,
or machine-specific paths.

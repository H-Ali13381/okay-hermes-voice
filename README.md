# Okay Hermes Voice

Always-on local "Okay Hermes" wakeword activation for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It listens for the Okay Hermes wakeword with a tiny ONNX model, records a spoken
request until silence, transcribes through Hermes STT, sends the request to a
warm in-process Hermes Agent, and speaks the reply through Hermes TTS.

## What it provides

- Always-on 16 kHz mono wakeword detection with ONNX Runtime.
- Low-latency warm Hermes Agent calls instead of spawning a fresh CLI process.
- Continuous voice conversation after wakeword activation.
- Explicit local close phrases such as `close`, `stop listening`, or `end voice mode`.
- Optional popup terminal visualizer with Ctrl-C cancellation of the active voice session.
- Pulse/PipeWire playback to the system default sink, a named sink, or every sink.
- Local activation archive with wake clips, command WAVs, transcripts, responses, and metadata.
- Inherits normal Hermes model/provider/toolsets by default.

## Requirements

- Linux with systemd user services.
- A working Hermes Agent install.
- A microphone visible to PortAudio/sounddevice.
- PulseAudio or PipeWire/Pulse for playback.
- Python packages: `numpy`, `onnxruntime`, `sounddevice`, `PyYAML`.
- Wakeword model weights from the upstream Okay Hermes RepCNN ONNX repo.

## Wakeword model

This repository does **not** bundle model weights. Download the model from:

https://github.com/H-Ali13381/okay-hermes-repcnn-onnx

Expected default artifact:

```text
~/.hermes/wakeword/okay-hermes-repcnn-onnx/retrained_20260510_165910_folded.onnx
```

Known SHA256 for that artifact:

```text
e705d3af445ab38666b06a1f475339ca47b3f8645e6d53d056a11db0a7a9fb19
```

Model assumptions:

- input name: `waveform`
- input shape: `(batch, 48000)`
- audio: mono float32, 16 kHz, exactly 3 seconds
- output name: `probabilities`
- recommended threshold: `0.4112943708896637`

## Install

Clone this repo, then install it into the Hermes Python environment:

```bash
git clone https://github.com/H-Ali13381/okay-hermes-voice.git
cd okay-hermes-voice
~/.hermes/hermes-agent/venv/bin/python -m pip install -e .
```

Download the wakeword model:

```bash
mkdir -p ~/.hermes/wakeword
git clone https://github.com/H-Ali13381/okay-hermes-repcnn-onnx.git   ~/.hermes/wakeword/okay-hermes-repcnn-onnx
sha256sum ~/.hermes/wakeword/okay-hermes-repcnn-onnx/retrained_20260510_165910_folded.onnx
```

Create config:

```bash
mkdir -p ~/.hermes/wakeword
cp config.example.yaml ~/.hermes/wakeword/config.yaml
```

Install and start the user service:

```bash
./scripts/install_user_service.sh
```

Or install manually:

```bash
cp systemd/hermes-wakeword.service ~/.config/systemd/user/hermes-wakeword.service
systemctl --user daemon-reload
systemctl --user enable --now hermes-wakeword.service
journalctl --user -u hermes-wakeword.service -f
```

## Usage

Say:

```text
Okay Hermes
```

Wait for the beep, then speak your request. After the response, voice mode keeps
listening for follow-up turns until you say an exact close command such as:

```text
close
stop listening
end voice mode
```

## Configuration

Edit:

```text
~/.hermes/wakeword/config.yaml
```

Important knobs:

- `threshold`: wakeword trigger probability.
- `trigger_consecutive_windows`: require multiple positive windows to reduce false accepts.
- `inference_interval_seconds`: CPU/latency tradeoff for ONNX inference.
- `speech_silence_duration_seconds`: silence cutoff after command speech.
- `hermes_provider` / `hermes_model`: blank means inherit normal Hermes config.
- `hermes_toolsets`: `null` means inherit normal CLI toolsets.
- `hermes_max_iterations`: maximum agent tool/reasoning loop iterations.
- `playback_sink`: `@DEFAULT_SINK@`, `all`, or a specific Pulse/PipeWire sink name.
- `save_activation_audio`: save local wake clips and metadata for false-positive review.

Restart after config changes:

```bash
systemctl --user restart hermes-wakeword.service
```

## Verification

```bash
okay-hermes-voice --smoke-test
okay-hermes-voice --list-devices
python -m py_compile src/okay_hermes_voice/wakeword_daemon.py src/okay_hermes_voice/voice_activation_popup.py
PYTHONPATH=src python -m pytest -q
```

## Privacy

Wakeword inference is local. STT, LLM, and TTS follow your Hermes configuration
and may use local or remote providers. Activation archiving is local, but it can
contain private speech. Do not publish `~/.hermes/wakeword/activations`.

## Attribution

- Hermes Agent: https://github.com/NousResearch/hermes-agent
- Okay Hermes RepCNN ONNX model: https://github.com/H-Ali13381/okay-hermes-repcnn-onnx

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

# Okay Hermes Voice

Talk to Hermes without touching the keyboard.

`Okay Hermes Voice` adds an always-on local wake phrase to [Hermes Agent](https://github.com/NousResearch/hermes-agent). Say "Okay Hermes", wait for the beep, speak your request, and Hermes answers out loud. After that, you can keep talking naturally until you say "close" or another close phrase.

This is a Linux voice add-on for people who already use Hermes and want a hands-free assistant on their own machine. It is not a polished app store installer yet, but the setup is meant to be copy-paste friendly.

## What you get

- Hands-free wake phrase: "Okay Hermes".
- Spoken answers using your existing Hermes text-to-speech setup.
- Follow-up conversation after the first request.
- A simple close command, such as "close", "stop listening", or "end voice mode".
- Optional popup window showing what Hermes heard and said.
- Local wakeword detection, so the always-listening part runs on your computer.
- Local activation archive for reviewing false activations.
- The same Hermes model, memory, tools, skills, and personality you normally use.

## Before you install

You need:

- Linux.
- A microphone and speakers/headphones.
- PipeWire/PulseAudio for sound playback. Most modern Linux desktops already have this.
- Hermes Agent installed and working.
- A terminal for setup.

Check Hermes first:

```bash
hermes doctor
hermes chat -q "Say hello in one short sentence."
```

If Hermes is not installed yet, start here:

https://hermes-agent.nousresearch.com/docs/getting-started/installation

## Quick install

Clone this project:

```bash
git clone https://github.com/H-Ali13381/okay-hermes-voice.git
cd okay-hermes-voice
```

Download the wakeword model:

```bash
mkdir -p ~/.hermes/wakeword
git clone https://github.com/H-Ali13381/okay-hermes-repcnn-onnx.git ~/.hermes/wakeword/okay-hermes-repcnn-onnx
```

Install and start the background service:

```bash
./scripts/install_user_service.sh
```

The installer will:

- install this package into the Hermes Python environment;
- create `~/.hermes/wakeword/config.yaml` if you do not already have one;
- install the user service as `hermes-wakeword.service`;
- start the service.

Check that it is running:

```bash
systemctl --user status hermes-wakeword.service --no-pager
```

Watch live logs if something looks wrong:

```bash
journalctl --user -u hermes-wakeword.service -f
```

## How to use it

Say:

```text
Okay Hermes
```

Wait for the beep. Then speak your request.

Examples:

```text
What time is it?
```

```text
Summarize my latest git changes.
```

```text
Turn this idea into a todo list.
```

Hermes will answer out loud. Voice mode then stays open for follow-up turns, so you can keep talking without saying the wake phrase again.

To end the voice session, say one of the close phrases:

```text
close
stop listening
end voice mode
bye
cancel
```

## Stopping or restarting

Stop listening now:

```bash
systemctl --user stop hermes-wakeword.service
```

Start listening again:

```bash
systemctl --user start hermes-wakeword.service
```

Restart after changing settings:

```bash
systemctl --user restart hermes-wakeword.service
```

Disable automatic startup:

```bash
systemctl --user disable --now hermes-wakeword.service
```

## Everyday settings

Settings live here:

```text
~/.hermes/wakeword/config.yaml
```

Most users only need a few options.

### If Hermes wakes too often

Raise the wake threshold a little:

```yaml
threshold: 0.45
```

You can also require more repeated wake detections before it triggers:

```yaml
trigger_consecutive_windows: 3
```

Restart after editing:

```bash
systemctl --user restart hermes-wakeword.service
```

### If Hermes does not wake reliably

Lower the wake threshold a little:

```yaml
threshold: 0.38
```

Do not lower it too far or random speech may trigger it.

### If replies play on the wrong device

By default, replies go to your system default audio output:

```yaml
playback_sink: '@DEFAULT_SINK@'
```

To send replies everywhere, use:

```yaml
playback_sink: all
```

For one specific PipeWire/PulseAudio sink, put that sink name here instead.

### If Hermes cuts you off too quickly

Increase the silence timeout:

```yaml
speech_silence_duration_seconds: 1.5
```

### If responses are too long

Lower the spoken response limit:

```yaml
max_spoken_response_chars: 1200
```

## Privacy

The always-listening wakeword check runs locally on your computer.

After the wake phrase is detected, your spoken request is handled by your normal Hermes speech-to-text, model, and text-to-speech configuration. If your Hermes setup uses cloud providers, those providers may receive the request text, generated response, or audio needed for transcription/speech.

Activation archives are local, but they can contain private speech. Do not publish this folder:

```text
~/.hermes/wakeword/activations
```

Also remember that voice requests can use the same Hermes tools as your normal CLI sessions. Treat this like giving your local assistant a microphone, not like installing a harmless sound widget.

## Troubleshooting

### Nothing happens when I say "Okay Hermes"

Check that the service is running:

```bash
systemctl --user status hermes-wakeword.service --no-pager
```

Check the logs:

```bash
journalctl --user -u hermes-wakeword.service -n 80 --no-pager
```

List available microphones:

```bash
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --list-devices
```

If your microphone is not the default input device, set it in your desktop sound settings first.

### The model or audio setup might be broken

Run a smoke test:

```bash
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --smoke-test
```

This loads the wakeword model and runs a tiny test inference without waiting for the wake phrase.

### Hermes hears the wake phrase, but no answer plays

Check your normal Hermes voice setup first. This project uses Hermes STT and TTS rather than shipping its own full speech stack.

Useful checks:

```bash
hermes doctor
journalctl --user -u hermes-wakeword.service -n 120 --no-pager
```

Also make sure your default audio output works in normal desktop apps.

### The popup window gets stuck

You can press Ctrl-C in the popup to cancel the active voice session. If needed, restart the service:

```bash
systemctl --user restart hermes-wakeword.service
```

## Technical details

This section is for users who want to know how it works or tune the system more deeply.

### Runtime flow

1. The daemon listens to the microphone at 16 kHz mono.
2. A 3 second rolling audio window is sent to the Okay Hermes RepCNN ONNX model.
3. When the model probability passes the configured threshold for enough windows, the daemon plays a beep.
4. The daemon records the spoken request until silence.
5. Hermes transcribes the audio through the STT provider configured in Hermes.
6. The request goes to a warm in-process Hermes Agent for lower latency than launching a new CLI process each time.
7. Hermes generates a response with the user's normal model/provider/toolsets.
8. Hermes TTS creates spoken audio.
9. The daemon plays the answer through PulseAudio/PipeWire.
10. If conversation mode is enabled, the daemon keeps listening for follow-up turns until a close phrase is heard.

### Wakeword model

This repository does not bundle model weights. Download the model from:

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

### Configuration reference

Important config options in `~/.hermes/wakeword/config.yaml`:

- `model_path`: path to the ONNX wakeword model.
- `threshold`: wakeword trigger probability.
- `trigger_consecutive_windows`: number of positive windows required before activation.
- `inference_interval_seconds`: CPU/latency tradeoff for ONNX inference.
- `speech_rms_threshold`: rough speech volume threshold while recording a request.
- `speech_start_timeout_seconds`: how long to wait for speech after the wake beep.
- `speech_silence_duration_seconds`: how long silence must last before the request is considered done.
- `max_command_seconds`: maximum length of one spoken request.
- `hermes_provider` / `hermes_model`: blank means inherit normal Hermes config.
- `hermes_toolsets`: `null` means inherit normal CLI toolsets.
- `hermes_inprocess`: use the warm in-process Hermes Agent path.
- `hermes_warm_agent`: keep the agent initialized between turns.
- `hermes_max_iterations`: maximum Hermes tool/reasoning loop iterations.
- `hermes_load_soul_identity`: load the normal Hermes identity/persona file.
- `interaction_router_enabled`: classify transcripts after STT and before the full agent.
- `interaction_router_model`: fast structured router model, defaulting to Gemini Flash Lite.
- `interaction_router_min_confidence`: below this, route conservatively to the full Hermes agent.
- `interaction_router_small_model_enabled`: allow simple/safe requests to bypass the full agent.
- `interaction_router_ack_cache_enabled`: cache short acknowledgement clips like “Got it.”
- `playback_sink`: `@DEFAULT_SINK@`, `all`, or a specific Pulse/PipeWire sink name.
- `visualization_enabled`: open the optional popup window.
- `conversation_mode_enabled`: keep listening after the first answer.
- `conversation_close_phrases`: phrases that end the voice session.
- `save_activation_audio`: save wake clips and metadata for later review.
- `activation_archive_dir`: where activation records are stored.

### Manual install

The install script is recommended, but manual installation is straightforward:

```bash
~/.hermes/hermes-agent/venv/bin/python -m pip install -e .
mkdir -p ~/.hermes/wakeword ~/.config/systemd/user
cp config.example.yaml ~/.hermes/wakeword/config.yaml
cp systemd/hermes-wakeword.service ~/.config/systemd/user/hermes-wakeword.service
systemctl --user daemon-reload
systemctl --user enable --now hermes-wakeword.service
```

If your Hermes checkout is not at `~/.hermes/hermes-agent`, set `HERMES_REPO` or `PYTHON` before running the install script:

```bash
HERMES_REPO=/path/to/hermes-agent ./scripts/install_user_service.sh
```

or:

```bash
PYTHON=/path/to/python ./scripts/install_user_service.sh
```

### Verification for contributors

```bash
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --smoke-test
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --list-devices
~/.hermes/hermes-agent/venv/bin/python -m py_compile src/okay_hermes_voice/wakeword_daemon.py src/okay_hermes_voice/voice_activation_popup.py
PYTHONPATH=src ~/.hermes/hermes-agent/venv/bin/python -m pytest -q
```

## Attribution

- Hermes Agent: https://github.com/NousResearch/hermes-agent
- Okay Hermes RepCNN ONNX model: https://github.com/H-Ali13381/okay-hermes-repcnn-onnx

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

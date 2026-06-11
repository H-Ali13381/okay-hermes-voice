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

Raise the wake threshold above the default `0.6973556280136108`:

```yaml
threshold: 0.75
```

By default, Hermes requires two consecutive positive wake windows to reduce false wakes. For lower latency, opt into one positive wake window explicitly:

```yaml
trigger_consecutive_windows: 1
```

Restart after editing:

```bash
systemctl --user restart hermes-wakeword.service
```

### If Hermes does not wake reliably

Lower the wake threshold below the default `0.6973556280136108`:

```yaml
threshold: 0.62
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

## Phase 0 latency summaries

Activation archives now include Phase 0 timing metadata for completed turns. To summarize the local archive without starting the daemon:

```bash
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --activation-summary
```

To summarize a specific archive directory:

```bash
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --activation-summary ~/.hermes/wakeword/activations
```

For scripts or benchmark comparisons, emit JSON:

```bash
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --activation-summary ~/.hermes/wakeword/activations --summary-json
```

The summary reports archive count, turn count, status/cancel counts, response-source counts, and per-stage timing statistics such as recording, STT, routing, answer generation, TTS, playback, and total turn time. If archive metadata is tagged with `benchmark_preset` and `benchmark_category`, the same metrics are grouped by preset so repeated task runs can be compared before and after later changes.

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

You can press Ctrl-C in the popup to cancel the active voice session. Cancellation is wired into the Hermes execution layer: the daemon calls `AIAgent.interrupt()` for the warm in-process agent, which propagates to active tools and subagents, and the subprocess fallback is launched in its own process group so it can be terminated as a whole. If needed, restart the service:

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
6. The interaction router classifies the transcript and schedules any short acknowledgement clip (for example “Okay, I’m on it.”) asynchronously, so the full Hermes agent can start immediately instead of waiting for the clip to finish.
7. The request goes to a warm in-process Hermes Agent for lower latency than launching a new CLI process each time.
8. Hermes generates a response with the user's normal model/provider/toolsets.
9. Hermes TTS creates spoken audio.
10. The daemon plays the answer through PulseAudio/PipeWire.
11. If conversation mode is enabled, the daemon keeps listening for follow-up turns until a close phrase is heard.

### Native PipeWire listener

The user service now starts the native C listener directly for the always-on path. Python is not in `ExecStart` and is not resident while the machine is just waiting for the wakeword.

Native source:

```text
native/okay-hermes-wake-listener.c
```

Installed binary:

```text
~/.hermes/wakeword/bin/okay-hermes-wake-listener
```

The listener uses direct PipeWire `pw_stream` capture plus the ONNX Runtime C API. The realtime `.process` callback dequeues the PipeWire buffer, does only cheap F32 downmix/resampling into a 16 kHz mono ring buffer, queues the buffer back, and returns. A normal worker thread snapshots the model's 3-second 16 kHz mono input, runs `OrtRun`, applies threshold/consecutive-window gating, and launches the short-lived Python activation handler only after detection.

Build it locally with:

```bash
PYTHON=~/.hermes/hermes-agent/venv/bin/python \
  ./native/build_wake_listener.sh --output ~/.hermes/wakeword/bin/okay-hermes-wake-listener
```

Run a model-only self-test without opening a microphone stream:

```bash
~/.hermes/wakeword/bin/okay-hermes-wake-listener \
  --model ~/.hermes/wakeword/okay-hermes-repcnn-onnx/wakeword.onnx \
  --self-test
```

Run a short direct PipeWire capture test without launching Python on activation:

```bash
~/.hermes/wakeword/bin/okay-hermes-wake-listener \
  --model ~/.hermes/wakeword/okay-hermes-repcnn-onnx/wakeword.onnx \
  --duration-seconds 5 \
  --verbose
```

On activation, the service uses:

```text
okay_hermes_voice.native_activation_handler
```

That Python process is intentionally short-lived: it records the command, transcribes, invokes Hermes, speaks the answer, then exits so STT/Hermes/CUDA/plugin memory is released instead of staying in the wake listener.

### Wakeword model

This repository does not bundle model weights. Download the model from:

https://github.com/H-Ali13381/okay-hermes-repcnn-onnx

Expected default artifact:

```text
~/.hermes/wakeword/okay-hermes-repcnn-onnx/wakeword.onnx
```

Known SHA256 for that artifact:

```text
f022856f17916f5c7b2d8041f44308f889f292d92be12b71e1fe3ee49bd0a0fc
```

Model assumptions:

- input name: `waveform`
- input shape: `(batch, time)`; use 48,000 samples per row for the 3-second default window
- audio: mono float32, 16 kHz, exactly 3 seconds
- output name: `score` (the daemon discovers the model output name dynamically)
- recommended threshold: `0.6973556280136108`

### Configuration reference

Important config options in `~/.hermes/wakeword/config.yaml`:

- `model_path`: path to the ONNX wakeword model.
- `threshold`: wakeword trigger probability.
- `trigger_consecutive_windows`: number of positive windows required before activation.
- `inference_interval_seconds`: CPU/latency tradeoff for ONNX inference.
- `wake_audio_backend`: legacy Python daemon capture backend; ignored by the native systemd service path.
- `wake_audio_device`: ALSA/PipeWire device name for the legacy `arecord` backend, default `default`.
- `native_listener_bin`: installed C wake listener path used by the systemd launcher.
- `native_pipewire_target`: optional PipeWire target object/node name or id for the native listener.
- `native_listener_verbose`: print native listener scores to stderr for debugging.
- `cooldown_seconds`: normal delay before listening again after a completed activation.
- `cancel_cooldown_seconds`: delay before listening again after popup Ctrl-C cancellation; `0.0` re-arms immediately.
- `speech_rms_threshold`: rough speech volume threshold while recording a request.
- `speech_start_timeout_seconds`: how long to wait for speech after the wake beep.
- `speech_silence_duration_seconds`: how long silence must last before the request is considered done.
- `max_command_seconds`: maximum length of one spoken request.
- `hermes_provider` / `hermes_model`: blank means inherit normal Hermes config.
- `hermes_toolsets`: `null` means inherit normal CLI toolsets.
- `hermes_inprocess`: use the warm in-process Hermes Agent path.
- `hermes_warm_agent`: keep the agent initialized between turns.
- `hermes_max_iterations`: maximum Hermes tool/reasoning loop iterations.
- `hermes_cancel_poll_seconds`: how often the daemon checks popup cancellation while Hermes is running.
- `hermes_interrupt_wait_seconds`: grace period after interrupt/SIGTERM before dropping the warm agent or escalating subprocess shutdown.
- `hermes_load_soul_identity`: load the normal Hermes identity/persona file.
- `interaction_router_enabled`: classify transcripts after STT and before the full agent.
- `interaction_router_model`: fast structured router model, defaulting to Gemini Flash Lite.
- `interaction_router_min_confidence`: below this, route conservatively to the full Hermes agent.
- `interaction_router_small_model_enabled`: allow simple/safe requests to bypass the full agent.
- `interaction_router_ack_cache_enabled`: cache short acknowledgement clips like “Okay, I’m on it.” Cached files preserve the TTS provider's audio extension, and acknowledgements are played asynchronously before full-agent work.
- `beep_enabled`: play short local beeps for wake/listening/error feedback.
- `stt_provider`: `hermes` keeps the normal Hermes STT stack; `nemotron_en_streaming` enables NVIDIA Nemotron English-only cache-aware streaming ASR; `parakeet_unified_streaming` enables NVIDIA Parakeet Unified English streaming ASR.
- `nemotron_model_name`: Hugging Face model id for the Nemotron provider, default `nvidia/nemotron-speech-streaming-en-0.6b`.
- `nemotron_model_path`: optional local `.nemo` checkpoint path; blank loads by model name through NeMo.
- `nemotron_device`: `auto`, `cuda`, or `cpu`.
- `nemotron_att_context_size`: Nemotron streaming lookahead context, default `[70, 13]` for the English model.
- `nemotron_cudnn_enabled`: enable/disable cuDNN for Nemotron CUDA inference; default `false` avoids the Torch CUDA 13 cuDNN sublibrary mismatch seen on this host.
- `nemotron_live_streaming`: when Nemotron is selected, feed microphone blocks to ASR during command recording and use that final live transcript instead of running post-WAV STT.
- `parakeet_model_name`: Hugging Face model id for the Parakeet provider, default `nvidia/parakeet-unified-en-0.6b`.
- `parakeet_left_context_secs`, `parakeet_chunk_secs`, `parakeet_right_context_secs`: chunked streaming context. `chunk + right` is the theoretical ASR latency.
- `parakeet_live_streaming`: when Parakeet is selected, feed microphone blocks to ASR during command recording and use that final live transcript instead of running post-WAV STT.
- `transcript_only_mode`: for shadow/benchmark daemons, stop after STT and archive the transcript without routing to Hermes, TTS, or playback.
- `prewarm_stt_on_start`: load the active STT path during daemon startup.
- `playback_sink`: `@DEFAULT_SINK@`, `all`, or a specific Pulse/PipeWire sink name.
- `visualization_enabled`: open the optional popup window.
- `visualization_launch_grace_seconds`: how long to wait for a terminal launch to fail before trying the next candidate.
- `conversation_mode_enabled`: keep listening after the first answer.
- `conversation_close_phrases`: phrases that end the voice session.
- `save_activation_audio`: save wake clips and metadata for later review.
- `activation_archive_dir`: where activation records are stored.

### Nemotron English streaming STT

The default STT provider remains Hermes' configured STT stack. To test NVIDIA's
English-only streaming ASR, install NeMo in the same environment that runs the
daemon, then switch only the wakeword config:

```bash
~/.hermes/hermes-agent/venv/bin/python -m pip install Cython packaging
~/.hermes/hermes-agent/venv/bin/python -m pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
```

```yaml
stt_provider: nemotron_en_streaming
nemotron_model_name: nvidia/nemotron-speech-streaming-en-0.6b
nemotron_device: auto
nemotron_att_context_size: [70, 13]
nemotron_cudnn_enabled: false
nemotron_live_streaming: true
```

With `nemotron_live_streaming: true`, command capture starts a live Nemotron
session before opening the microphone stream, feeds accepted mic blocks into
NeMo as speech is recorded, and uses that final live transcript directly. The
post-WAV `transcribe_command()` path remains as a fallback when live streaming is
disabled or produces no transcript. The provider still uses NeMo's
`CacheAwareStreamingAudioBuffer` and `conformer_stream_step`; it does not call
offline `model.transcribe`.

### Parakeet Unified English streaming STT

Parakeet Unified is the higher-quality English NVIDIA streaming candidate. It
uses a chunked RNNT streaming loop with configurable left/chunk/right context:

```yaml
stt_provider: parakeet_unified_streaming
parakeet_model_name: nvidia/parakeet-unified-en-0.6b
parakeet_device: auto
parakeet_left_context_secs: 2.0
parakeet_chunk_secs: 0.56
parakeet_right_context_secs: 0.56
parakeet_cudnn_enabled: false
parakeet_live_streaming: true
```

With the default context, theoretical ASR latency is about 1.12s
(`chunk + right`). Command capture starts a live Parakeet session before opening
the microphone stream, feeds accepted mic blocks during recording, and uses the
final live transcript directly when available.

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
~/.hermes/hermes-agent/venv/bin/okay-hermes-voice --activation-summary ~/.hermes/wakeword/activations --summary-json
~/.hermes/hermes-agent/venv/bin/python -m py_compile src/okay_hermes_voice/activation_archive.py src/okay_hermes_voice/wakeword_daemon.py src/okay_hermes_voice/voice_activation_popup.py
PYTHONPATH=src ~/.hermes/hermes-agent/venv/bin/python -m pytest -q
```

## Attribution

- Hermes Agent: https://github.com/NousResearch/hermes-agent
- Okay Hermes RepCNN ONNX model: https://github.com/H-Ali13381/okay-hermes-repcnn-onnx

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

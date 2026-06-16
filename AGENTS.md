# AGENTS.md

Guidance for AI agents working in this repository.

## Project summary

`okay-hermes-voice` is a Linux voice add-on for Hermes Agent. It provides a local "Okay Hermes" wake phrase, captures the user's spoken request, transcribes it, routes the request through Hermes Agent, and plays the answer out loud.

Core goals:

- Keep always-on idle listening local and lightweight.
- Preserve normal Hermes Agent behavior: same model config, memory, tools, skills, and personality.
- Prefer native/event-driven Linux integration over polling subprocesses.
- Keep runtime behavior practical and measurable: verify CPU, memory, service status, and tests after daemon/tray changes.

Primary runtime path:

1. Native wake listener detects the wake phrase.
2. Python activation handler or warm activation server receives the activation payload.
3. Activation flow records/transcribes speech, launches optional visualization, routes the request, calls Hermes, and plays TTS.
4. Conversation mode remains active until a close phrase is heard.

## Repository map

- `src/okay_hermes_voice/wakeword_daemon.py`
  - Main Python CLI entrypoint: `okay-hermes-voice`.
  - Owns daemon orchestration only: config, warmup, wake loop, activation handoff, cooldown.

- `src/okay_hermes_voice/daemon_config.py`
  - Config defaults, YAML loading, logging, signal handling.
  - User config lives at `~/.hermes/wakeword/config.yaml`.

- `src/okay_hermes_voice/audio/`
  - Wakeword capture/inference, recording, WAV helpers, transcription provider wrappers, audio device helpers.
  - `audio/wake/` contains wake capture/session/inference mechanics.
  - `audio/nemotron/` and `audio/parakeet/` contain NVIDIA streaming ASR integrations.

- `src/okay_hermes_voice/activation_flow.py`
  - Public facade for activation handling.
  - Implementation lives under `src/okay_hermes_voice/activation/flow/`.

- `src/okay_hermes_voice/activation/flow/`
  - Voice-session orchestration: setup, routing, turn input, timing, visualization, cancellation, outcomes.

- `src/okay_hermes_voice/activation_archive/` and `src/okay_hermes_voice/activation/archive/summary/`
  - Activation audio/metadata persistence and latency summary reporting.

- `src/okay_hermes_voice/hermes_runtime.py`
  - Warm/in-process Hermes runtime integration.

- `src/okay_hermes_voice/hermes_subprocess/`
  - Subprocess-based Hermes execution, output parsing, cancellation, termination.

- `src/okay_hermes_voice/interaction_router.py`, `interaction_clients/`, `interaction_routes/`, `voice_routing/`
  - Fast interaction routing, close phrase handling, acknowledgement generation/playback, optional small-model path.
  - Keep router responsibilities split: classification in `interaction_clients/router_classification.py`, provider-client caching in `interaction_clients/router_client_cache.py`, startup prewarm in `interaction_clients/router_prewarm.py`, and daemon orchestration in `native_activation_server.py`.
  - Router prewarm resolves/caches the provider client only; do not add dummy classification calls at startup unless explicitly measuring token/API tradeoffs.

- `src/okay_hermes_voice/playback/response/`
  - Beeps, TTS playback, sink selection, process waiting/termination.

- `src/okay_hermes_voice/visualization/`
  - Optional terminal popup state, launcher, rendering pipeline, input handling.
  - `voice_activation_popup.py` is the installed CLI entrypoint facade.

- `src/okay_hermes_voice/native_activation_handler.py`
  - Short-lived Python handler launched by the native wake listener after an activation.
  - Can forward to the warm activation server when enabled.

- `src/okay_hermes_voice/native_activation_server.py`
  - Warm Unix-socket activation server that keeps STT/Hermes initialized for lower latency.
  - Writes the `.ready` marker watched by the native tray.

- `native/okay-hermes-wake-listener.c`
  - Native always-on wake listener. This is the low-idle-residency path used by the user systemd service.

- `native/build_wake_listener.sh`
  - Builds the native C wake listener into `~/.hermes/wakeword/bin/okay-hermes-wake-listener` during service install.

- `native/wakeword-tray/`
  - Native Qt/KDE system tray control for the wakeword services.
  - Uses Qt Widgets, QtDBus, QFileSystemWatcher, and PulseAudioQt.
  - Shows red/off, yellow/starting, green/on, and gray/no-microphone states.
  - Right-click menu is intentionally minimal: `Turn ON`, `Turn OFF`, `Exit`.

- `systemd/hermes-wakeword.service`
  - User service for the native wake listener.

- `scripts/install_user_service.sh`
  - Installs the Python package into the Hermes venv, builds the native listener, installs/enables the user service.

- `scripts/install_wakeword_tray.sh`
  - Builds and installs the native tray binary to `~/.local/bin/okay-hermes-wakeword-tray` and adds a desktop autostart entry.

- `tests/`
  - Main test suite.
  - `tests/voice_conversation/` covers daemon/config/audio/activation/native listener/tray behavior.
  - `tests/interaction_router/` covers routing decisions and client behavior.

## Architecture conventions

This repo intentionally follows a root-to-leaf structure:

- Root modules tell the application story or preserve public facades.
- Implementation mechanics live in semantic subpackages.
- Avoid generic `utils.py`/`helpers.py` style junk drawers.
- Keep package facades thin. If a facade starts owning loops, retries, parsing, device work, or rendering, move that logic deeper.
- Structure tests enforce that implementation leaf files do not accumulate many top-level functions. See `tests/voice_conversation/test_source_structure.py`.

When adding Python code:

- Prefer semantic leaf modules under the relevant package.
- Preserve public import paths through explicit facades when needed.
- Update or add focused tests in the same change.
- Do not add tracked code that depends on untracked generated files.

When adding native Linux integration:

- Prefer event-driven APIs over periodic polling.
- Keep subprocess spawning out of idle paths.
- Measure CPU/memory if changing always-on components.
- For router prewarm changes, verify both `tests/voice_conversation/test_source_structure.py` and `tests/interaction_router/` so latency features do not collapse into oversized leaf modules.
- Keep systemd service names and installed binary paths stable unless the user explicitly asks for a migration.

## Native tray notes

The native tray is deliberately lean and desktop-standard:

- Build files:
  - `native/wakeword-tray/CMakeLists.txt`
  - `native/wakeword-tray/main.cpp`
  - `native/wakeword-tray/tray_state.h` for pure, testable state helpers.
- Install command:
  - `scripts/install_wakeword_tray.sh`
- Installed binary:
  - `~/.local/bin/okay-hermes-wakeword-tray`
- Autostart file:
  - `~/.config/autostart/okay-hermes-wakeword-tray.desktop`
- Controlled units:
  - `hermes-wakeword.service`
  - `hermes-voice-handler.service`
- Ready marker:
  - `~/.hermes/wakeword/native-handler.ready`

Current implementation direction:

- systemd state and start/stop use QtDBus.
- marker readiness uses QFileSystemWatcher.
- microphone/default-source changes use PulseAudioQt.
- No recurring `systemctl is-active`, `wpctl`, or poll timer should be reintroduced without a strong reason.

## Known non-goals / accepted out-of-scope edge cases

These are known possible hardening items. They are intentionally out of scope unless the user reports the symptom or explicitly asks to harden rare desktop/session failures.

### 1. Re-subscribing after user D-Bus or user-systemd resets

The tray records that systemd D-Bus signal subscriptions succeeded. If the user D-Bus connection or the user systemd manager restarts underneath the tray, those signal matches could be dropped while the tray still believes they are active.

What could happen:

- The tray icon may stop noticing service state changes made outside the tray.
- State may look stale until another refresh path happens or the tray is restarted.

Why this is accepted:

- User D-Bus and user systemd normally live for the whole login session.
- If they restart, many desktop components are already in a degraded state.
- The failure mode is stale UI, not data loss or runaway CPU.

Do not add complex reconnection machinery for this unless there is a real observed stale-state bug.

### 2. QFileSystemWatcher add failures for the ready marker

The tray creates/watches the ready-marker directory and file, but does not aggressively recover from every possible watcher registration failure.

What could happen:

- If inotify limits, permissions, or filesystem issues prevent the watcher from being added, the tray may miss `native-handler.ready` creation/deletion.
- The icon could stay yellow/starting even though the handler is ready.

Why this is accepted:

- This is unlikely on the target desktop.
- Inotify exhaustion or home-directory permission problems would affect other desktop/dev tools too.
- The service can still work; the risk is stale UI.

Do not over-engineer this path unless the tray gets stuck yellow in practice.

### 3. Strict microphone availability detection

The tray currently treats an unavailable/non-ready PulseAudioQt default source as no microphone and shows the gray state.

What could happen:

- During unusual audio-server startup/hotplug timing, the tray may briefly gray out or disable `Turn ON` even if a usable mic appears moments later.

Why this is accepted for now:

- The target setup is KDE/PipeWire where PulseAudioQt default-source updates are reliable.
- The tray refreshes on audio source/server changes.
- A tri-state `AudioUnknown` model would add complexity that is not justified until false-gray states are observed.

## Common commands

Use the Hermes venv Python when available on this machine:

```bash
/home/neos/.hermes/hermes-agent/venv/bin/python -m pytest tests/voice_conversation -q
```

General test commands:

```bash
python -m pytest
python -m pytest tests/voice_conversation -q
python -m pytest tests/interaction_router -q
python -m pytest tests/voice_conversation/test_wakeword_tray.py -q
```

Native tray build:

```bash
cmake -S native/wakeword-tray -B build/wakeword-tray -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build/wakeword-tray --config Release
```

Install/restart tray locally:

```bash
scripts/install_wakeword_tray.sh
systemctl --user restart okay-hermes-wakeword-tray.service || \
  systemd-run --user --unit=okay-hermes-wakeword-tray --collect ~/.local/bin/okay-hermes-wakeword-tray
```

Install/restart wakeword service:

```bash
scripts/install_user_service.sh
systemctl --user restart hermes-wakeword.service
journalctl --user -u hermes-wakeword.service -f
```

Useful verification before claiming completion:

```bash
git diff --check
git status --short --branch --untracked-files=all
python -m pytest tests/voice_conversation -q
```

For native tray changes, also verify the live status notifier when practical:

```bash
busctl --user get-property org.kde.StatusNotifierWatcher /StatusNotifierWatcher \
  org.kde.StatusNotifierWatcher RegisteredStatusNotifierItems
```

## Coding and review discipline

- Keep scope tight. Do not bundle unrelated refactors with bug fixes.
- Prefer simple, explicit state over clever abstractions.
- Errors should not pass silently in runtime/service paths unless explicitly intended.
- Use source-string tests only as lightweight architecture guardrails; do not mistake them for full behavioral coverage.
- For always-on components, avoid wakeup churn and child-process probes.
- For native code, build after every meaningful change.
- For Python code, run targeted tests plus the relevant full subset.
- Do not commit unless explicitly asked.

## Files and artifacts to avoid editing/committing

Do not edit or commit generated/transient files:

- `build/`
- `.pytest_cache/`
- `__pycache__/`
- local logs under `~/.hermes/logs/`
- local wakeword config under `~/.hermes/wakeword/config.yaml`
- installed binaries under `~/.local/bin/`
- desktop autostart output under `~/.config/autostart/`

Keep repository changes in tracked source, tests, scripts, docs, and config examples only.

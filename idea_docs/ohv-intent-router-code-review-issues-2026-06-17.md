# OHV Intent/Router Active Issues

Date: 2026-06-17
Last updated: 2026-06-17 20:09 EDT
Repo: `okay-hermes-voice`

Independent Claude Opus review artifact:
`idea_docs/claude-opus-code-review-20260617-155248.txt`

## Verification refresh

Latest verification pass: 2026-06-17 after router/default/STT/classification fixes.

Commands run while updating this list:

- `PYTHONPATH=src uv run --with omegaconf python -m pytest tests/interaction_router tests/voice_conversation -q`
  - Result: `156 passed in 7.44s`
- `uv run python -m pytest tests/interaction_router tests/voice_conversation/test_routing_and_acks.py tests/voice_conversation/test_default_stt_provider.py -q`
  - Result: `44 passed in 0.13s`
- `PYTHONPATH=src uv run python - <<'PY' ... import okay_hermes_voice.voice_routing / playback.response ... PY`
  - Result: `okay_hermes_voice.voice_routing: ok`, `okay_hermes_voice.playback.response: ok`
- Live classification smoke after classification/prompting fix:
  - `hello` -> `simple small_model none 1.0 local_simple_chat`, route `small_model / router_small_model`
  - `how are you` -> `simple small_model none 1.0 local_simple_chat`, route `small_model / router_small_model`
  - `tell me a fun fact` -> `simple small_model none 0.95 simple_safe_chat`, route `small_model / router_small_model`
  - `what is recursion` -> `simple small_model none 0.9 simple_explanation`, route `small_model / router_small_model`
- OHV services restarted after classification/prompting fix:
  - `hermes-voice-handler.service`: active
  - `hermes-wakeword.service`: active
  - `app-okay-hermes-wakeword-tray@autostart.service`: active

## Pruned as solved after verification

These issues are no longer active in the current working tree.

1. Default STT does not use Parakeet
   - `src/okay_hermes_voice/daemon_config.py` now defaults `stt_provider` to `parakeet_unified_streaming`.
   - `tests/voice_conversation/test_default_stt_provider.py` covers the default.

2. Ack playback waits for router classification
   - `src/okay_hermes_voice/voice_routing/request_route.py` starts a provisional cached `GOT_IT` before planning when `loop_ack_until_cancelled=True`.
   - `src/okay_hermes_voice/activation/flow/routing.py` passes `loop_ack_until_cancelled=True`.
   - Covered by `test_route_transcribed_request_schedules_provisional_ack_before_planning`.

3. Route taxonomy does not match planned `local | chat | tools | agent`
   - Runtime targets still use compatibility enums.
   - `VoiceRequestPlan.route_lane` exposes canonical product/training lanes.
   - Covered by `test_voice_request_plan_exposes_canonical_product_route_lanes`.

4. Warm activation server stale guard misses in-flight activations
   - Previously audited as already covered: warm server is serial (`listen(1)` + `_serve_once`), and queued activations detected during an active session are dropped after the session finishes by comparing `detected_at` against `last_session_finished_at + cooldown`.
   - Existing test: `test_native_activation_server_drops_queued_activation_from_active_session`.

5. `IMMEDIATE_ONLY` is not executable in `answer_routed_request`
   - `answer_routed_request()` now explicitly handles `RouteTarget.IMMEDIATE_ONLY` and does not call Hermes.
   - Covered by `test_answer_routed_request_immediate_only_bypasses_heavy_hermes_call`.

6. Close phrase logic has multiple divergent sources of truth
   - Shared source now lives in `src/okay_hermes_voice/close_phrases.py`.
   - Daemon config, local classification, route choice, and voice normalization now use/re-export the shared normalization/phrase set.
   - Covered by close-phrase route-selection and local-classification tests.

7. Small-model fast path times out too aggressively
   - Added `small_model_timeout_seconds` separate from router classification timeout.
   - `answer_with_small_model()` now uses `cfg.small_model_timeout_seconds`.
   - Covered by config and small-model client tests.

8. Simple safe requests still use GPT-5.5/heavy Hermes
   - Root cause was `interaction_router_small_model_enabled=False` by default.
   - Defaults now enable the small-model path:
     - `InteractionRouterConfig.small_model_enabled = True`
     - `DEFAULT_CONFIG["interaction_router_small_model_enabled"] = True`
     - daemon-config fallback defaults to `True`
   - Covered by default-config routing and answer-path tests.

9. Router prompt/classification sends obvious pleasantries to `heavy_agent`
   - Router prompt now explicitly documents `small_model` as the fast LLM answer path for simple safe chat.
   - Deterministic local classification now handles obvious pleasantries without resolving a remote router provider.
   - Covered by prompt tests and `test_classify_request_handles_obvious_local_simple_chat_without_provider`.

10. Test/dependency setup is brittle for router/ack subset without full Hermes environment
   - Import brittleness around `tools.*` was reduced with lazy imports/facades.
   - Verified `uv run python -m pytest tests/interaction_router tests/voice_conversation/test_routing_and_acks.py tests/voice_conversation/test_default_stt_provider.py -q` -> `44 passed` without manual Hermes `PYTHONPATH`.
   - Verified package import smoke with `PYTHONPATH=src`.

11. Router failure branches lack direct coverage
   - Added fake-client coverage for timeout `TypeError` retry, non-timeout `TypeError`, generic exception, invalid response shape, and invalid JSON response.
   - Covered in `tests/interaction_router/test_client_classification.py`.

## Active Issues

### 1. HIGH — No structured function/capability catalog for timer/tool/local routing

Evidence:

- Live OHV request from `~/.hermes/logs/okay-hermes-wakeword.log` before the latest prompt work:
  - Transcript: `i am currently testing routing can you start a five minute time`
  - Router result: `lane=tools target=heavy_agent ack=got_it reason=low_router_confidence confidence=0.00 latency=1.039s`
  - Router reason: `User is asking to start a timer, which is a direct function call to a timer API.`
- The prompt now has examples and better small-chat rules, but there is still no structured catalog/data model for capabilities such as timer, voice session control, media, browser/search, tool-backed actions, or heavy-agent handoff.

Impact:

- The router can recognize “this is a timer/tool-ish thing” but cannot map it to a stable function contract.
- Future BERT/local routing would need to relearn product taxonomy from prose instead of consuming a structured catalog.
- Timer/tool requests can still fall through to heavy Hermes instead of a bounded tool/local path.

Fix:

- Add a structured capability catalog consumed by the prompt builder and local deterministic router, for example:
  - `timer.start(duration, label)` -> `tools` or future local timer handler
  - `voice.close()` -> `local`
  - `voice.cancel_or_barge_in()` -> `local`
  - `chat.answer_briefly(...)` -> `chat`
  - `media.play/search(...)` -> `tools`
  - `browser.open/search(...)` -> `tools`
  - `agent.deep_task(...)` -> `agent`
- Include compatibility mapping from canonical lanes to current runtime route targets.
- Keep the catalog structured enough to feed BERT/ONNX/local classifier training later.

Test:

- Prompt-builder tests assert catalog contents, timer examples, lane mapping, and non-execution guardrails.
- Local/router tests assert `start a five minute timer` does not become low-confidence heavy-agent by default.
- Prompt-injection transcript tests confirm transcript text cannot override router instructions.

### 2. HIGH — Router remains remote LLM fallback-first for most non-deterministic turns

Evidence:

- `classify_request()` now checks local deterministic cases first.
- Local deterministic coverage currently includes close phrases and obvious simple chat/pleasantries.
- Most other turns still fall through to the LLM-backed `DEFAULT_INTENT_ENGINE`.

Impact:

- Most routed voice turns still send transcript to a remote provider before response handling.
- This conflicts with the local-first/BERT/table-router direction.
- Network latency remains in the router stage after transcript finalization.

Fix:

- Expand the local first-stage router using the same capability catalog:
  - close/cancel/session control
  - timer commands
  - deterministic tool/media/browser patterns
  - simple chat patterns beyond fixed pleasantry phrases when safe
- Keep the LLM router as fallback/teacher for ambiguous turns until the BERT/local model exists.

Test:

- Assert known deterministic commands classify without calling `resolve_provider_client`:
  - close/cancel
  - start timer
  - simple local chat
  - obvious media/browser/tool request when confidence is deterministic

### 3. HIGH — No dedicated route-decision telemetry dataset for BERT distillation

Evidence:

- Runtime logs and route metadata include transcript/decision-ish fields, but there is no dedicated append-only training dataset/export path.
- The earlier Mermaid telemetry arrow was explicitly inferred/planned, not existing code.

Impact:

- The LLM router cannot reliably become a teacher for future BERT distillation without structured examples.
- Later training work will depend on ad hoc log scraping or manual reconstruction.

Fix:

- Add privacy-conscious JSONL route telemetry behind a config flag.
- Capture stable structured fields:
  - timestamp/session/turn id where available
  - transcript or transcript hash depending on privacy mode
  - route target and canonical lane
  - request complexity
  - tool/memory/external-data flags
  - confidence and reason
  - selected execution source (`small_model`, `heavy_agent`, `immediate_only`, etc.)
  - fallback outcome/error category
- Add a correction/outcome hook later for supervised labels.

Test:

- Unit test: one routed request writes exactly one schema-valid JSONL row.
- Privacy-mode test: raw transcript is omitted when configured.
- Failure-mode test: fallback/error categories are logged without crashing the voice turn.

### 4. HIGH — No voice barge-in path

Evidence:

- Cancellation currently comes from visualization/session flags.
- Playback cancellation polls `cancel_check`.
- No audio/VAD/wakeword interrupt path is wired for playback.

Impact:

- User cannot interrupt a long spoken answer by speaking.
- Popup/session cancellation works, but voice assistant UX still lacks natural barge-in.

Fix:

- During playback, run a lightweight VAD/wakeword interrupt listener.
- On detected barge-in, set the same cancellation signal used by playback and Hermes execution.
- Define policy first: wakeword-only interrupt vs any speech/VAD interrupt.

Test:

- Simulated audio interrupt cancels playback and propagates through the same cancellation path.
- False-positive/noise case does not cancel playback under threshold.

### 5. MEDIUM — Heavy Hermes path is synchronous; no voice-level background handoff

Evidence:

- `answer_routed_request()` directly calls `ask_hermes_turn()` for heavy routes.
- `hermes_runtime.py` runs the Hermes turn synchronously.
- `hermes_timeout_seconds` remains long (`900`).

Impact:

- One heavy task can block the whole voice session for up to 15 minutes.
- No “I’ll work on that and report back” behavior yet.
- Conflicts with the desired Hermes handoff/background-delegation model.

Fix:

- Add a heavy-task handoff path:
  - immediate ack
  - background Hermes/delegate task
  - return voice session to idle/listening
  - deliver completion later via chosen channel: notification, TTS, log, popup, or chat
- Requires product decision on delivery UX.

Test:

- Long-running fake Hermes task transitions to background outcome before voice timeout.
- Completion callback/delivery is recorded and does not block a new voice session.

### 6. MEDIUM — Need live dogfood evidence after classification/prompting fix

Evidence:

- Unit and smoke tests prove expected route decisions.
- The daemon was restarted after the fix.
- There is not yet a captured live spoken request log proving `how are you`/`tell me a fun fact` now avoids GPT-5.5 in the running daemon.

Impact:

- Tests can pass while live service import/config/runtime path is still wrong.
- This was exactly the failure mode for `small_model_enabled` before restart.

Fix:

- Speak these after restart:
  - `how are you`
  - `tell me a fun fact`
  - `what is recursion`
- Check `~/.hermes/logs/okay-hermes-wakeword.log` / journal for:
  - `target=small_model` or `source=small_model` where logged
  - no immediate `agent.conversation_loop: model=gpt-5.5` for those turns

Test:

- Manual live dogfood log capture, then optionally add a script/check that extracts route/source/model lines around the activation timestamp.

## Recommended priority order

1. Add the structured capability catalog and timer-routing contract.
2. Expand deterministic local first-stage router using that catalog.
3. Add route-decision telemetry for future BERT/local-router distillation.
4. Dogfood live spoken simple-chat turns and capture logs proving small-model runtime behavior.
5. Design/implement voice barge-in policy.
6. Design/implement heavy-task background handoff.
7. Split/checkpoint the current dirty working tree into coherent commits before making more broad changes.

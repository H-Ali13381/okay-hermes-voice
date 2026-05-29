from __future__ import annotations

import signal
import sys
import time
import types

from okay_hermes_voice import hermes_runtime as runtime


def test_configured_hermes_toolsets_preserves_explicit_wakeword_override():
    assert runtime.configured_hermes_toolsets({"hermes_toolsets": ["web"]}) == ["web"]

def test_configured_hermes_toolsets_inherits_cli_platform_tools_when_unset(monkeypatch):
    from hermes_cli import config as hermes_config
    from hermes_cli import tools_config

    fake_cfg = {"platform_toolsets": {"cli": ["terminal", "file", "skills"]}}
    monkeypatch.setattr(hermes_config, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda cfg, platform: set(cfg["platform_toolsets"][platform]))

    assert runtime.configured_hermes_toolsets({"hermes_toolsets": None}) == ["file", "skills", "terminal"]
    assert runtime.configured_hermes_toolsets({"hermes_toolsets": ""}) == ["file", "skills", "terminal"]

def test_warm_hermes_agent_loads_soul_identity_without_project_context(monkeypatch):
    created_kwargs = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            created_kwargs.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        types.SimpleNamespace(
            resolve_runtime_provider=lambda requested=None, target_model=None: {
                "provider": "openai-codex",
                "api_key": "token",
                "base_url": "https://example.invalid",
                "api_mode": "codex_responses",
                "credential_pool": None,
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(
            _create_session_db_for_oneshot=lambda: object(),
            _oneshot_clarify_callback=lambda *_args, **_kwargs: "pick a default",
        ),
    )
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAIAgent))
    runtime._HERMES_AGENT_CACHE.clear()

    runtime.get_warm_hermes_agent(
        {"hermes_max_iterations": 90, "hermes_load_soul_identity": True},
        provider=None,
        model="gpt-5.5",
        toolsets=["skills"],
    )

    assert created_kwargs["skip_context_files"] is True
    assert created_kwargs["load_soul_identity"] is True
    assert created_kwargs["max_iterations"] == 90
    runtime._HERMES_AGENT_CACHE.clear()

def test_ask_hermes_turn_preserves_voice_conversation_history(monkeypatch):
    calls = {}

    class FakeAgent:
        def run_conversation(self, prompt, conversation_history=None, persist_user_message=None):
            calls["prompt"] = prompt
            calls["conversation_history"] = list(conversation_history or [])
            calls["persist_user_message"] = persist_user_message
            return {
                "final_response": "second answer",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": persist_user_message},
                    {"role": "assistant", "content": "second answer"},
                ],
            }

    monkeypatch.setattr(runtime, "configured_hermes_runtime_selection", lambda cfg: (None, "gpt-5.5"))
    monkeypatch.setattr(runtime, "configured_hermes_toolsets", lambda cfg: ["skills"])
    monkeypatch.setattr(runtime, "get_warm_hermes_agent", lambda cfg, provider, model, toolsets: FakeAgent())

    previous_history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    response, updated_history = runtime.ask_hermes_turn(
        {
            "hermes_inprocess": True,
            "hermes_warm_agent": True,
            "hermes_prompt_prefix": "VOICE PREFIX",
        },
        "follow-up question",
        previous_history,
    )

    assert response == "second answer"
    assert calls["prompt"] == "VOICE PREFIX\n\nTranscript:\nfollow-up question"
    assert calls["conversation_history"] == previous_history
    assert calls["persist_user_message"] == "follow-up question"
    assert updated_history[-2:] == [
        {"role": "user", "content": "follow-up question"},
        {"role": "assistant", "content": "second answer"},
    ]

def test_ask_hermes_turn_interrupts_warm_agent_when_cancel_requested(monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.interrupted = False
            self.interrupt_message = None

        def run_conversation(self, prompt, conversation_history=None, persist_user_message=None):
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline and not self.interrupted:
                time.sleep(0.01)
            return {
                "final_response": "late answer",
                "messages": list(conversation_history or []),
                "interrupted": self.interrupted,
            }

        def interrupt(self, message=None):
            self.interrupted = True
            self.interrupt_message = message

    fake_agent = FakeAgent()
    monkeypatch.setattr(runtime, "configured_hermes_runtime_selection", lambda cfg: (None, "gpt-5.5"))
    monkeypatch.setattr(runtime, "configured_hermes_toolsets", lambda cfg: ["skills"])
    monkeypatch.setattr(runtime, "get_warm_hermes_agent", lambda cfg, provider, model, toolsets: fake_agent)

    response, history = runtime.ask_hermes_turn(
        {
            "hermes_inprocess": True,
            "hermes_warm_agent": True,
            "hermes_cancel_poll_seconds": 0.01,
            "hermes_interrupt_wait_seconds": 1.0,
        },
        "please do a long task",
        [{"role": "user", "content": "before"}],
        cancel_check=lambda: True,
    )

    assert response is None
    assert history == [{"role": "user", "content": "before"}]
    assert fake_agent.interrupted is True
    assert fake_agent.interrupt_message == "Voice session cancelled"

def test_ask_hermes_turn_kills_subprocess_group_when_cancel_requested(monkeypatch):
    popen_kwargs = {}
    killpg_calls = []

    class FakeProc:
        pid = 43210
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -signal.SIGTERM
            return self.returncode

        def communicate(self):
            return "partial stdout", "partial stderr"

    fake_proc = FakeProc()

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return fake_proc

    monkeypatch.setattr(runtime, "configured_hermes_runtime_selection", lambda cfg: (None, None))
    monkeypatch.setattr(runtime, "configured_hermes_toolsets", lambda cfg: None)
    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Hermes subprocess must be cancellable Popen, not run")),
    )
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))

    response, history = runtime.ask_hermes_turn(
        {
            "hermes_inprocess": False,
            "hermes_bin": "hermes",
            "hermes_source": "wakeword",
            "hermes_timeout_seconds": 30,
            "hermes_cancel_poll_seconds": 0.01,
            "hermes_interrupt_wait_seconds": 1.0,
        },
        "run a long shell command",
        [],
        cancel_check=lambda: True,
    )

    assert response is None
    assert history == []
    assert popen_kwargs["start_new_session"] is True
    assert killpg_calls == [(fake_proc.pid, signal.SIGTERM)]

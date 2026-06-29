from __future__ import annotations

from okay_hermes_voice.interaction_router import build_router_messages


def test_build_router_messages_mentions_json_and_forbids_solving():
    messages = build_router_messages("inspect the repo and fix tests")
    joined = "\n".join(message["content"] for message in messages)

    assert "JSON" in joined
    assert "Do not solve" in joined
    assert "inspect the repo and fix tests" in joined


def test_build_router_messages_biases_simple_safe_chat_to_small_model():
    messages = build_router_messages("how are you")
    joined = "\n".join(message["content"] for message in messages)

    assert "small_model" in joined
    assert "heavy_agent" in joined
    assert "pleasantries" in joined
    assert "fun facts" in joined
    assert "tool_risk" in joined
    assert "Do not choose heavy_agent for simple safe chat" in joined


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, content: str):
        self.completions = FakeCompletions(content)


class FakeClient:
    def __init__(self, content: str):
        self.chat = FakeChat(content)

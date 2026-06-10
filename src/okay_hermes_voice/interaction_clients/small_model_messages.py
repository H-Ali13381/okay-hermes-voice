"""Prompt messages for the fast small-model answer path."""
from __future__ import annotations


def build_small_model_messages(transcript: str) -> list[dict[str, str]]:
    system = (
        "You are the fast response path for a local voice assistant. "
        "Answer only simple, safe requests that require no tools, memory, or current external data. "
        "Keep the spoken answer concise. If the request needs tools, memory, external data, "
        "or side effects, say: I need the full agent for that."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": transcript}]


__all__ = ["build_small_model_messages"]

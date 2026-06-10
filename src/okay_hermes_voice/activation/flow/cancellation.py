"""Cancellation boundary for an activation session."""
from __future__ import annotations

from typing import Any, Dict, List

from .services import ActivationFlowServices


class ActivationCancellation:
    def __init__(
        self,
        deps: ActivationFlowServices,
        visual_state: Any,
        activation_archive: Any,
        archive_turns: List[Dict[str, Any]],
    ):
        self.deps = deps
        self.visual_state = visual_state
        self.activation_archive = activation_archive
        self.archive_turns = archive_turns

    def requested(self) -> bool:
        return bool(self.deps.is_visualization_cancel_requested(self.visual_state))

    def stop_if_cancelled(self) -> bool:
        if not self.requested():
            return False
        self.deps.finish_cancelled_voice_session(
            self.visual_state,
            self.activation_archive,
            self.archive_turns,
            self.deps.visualization_cancel_reason(self.visual_state),
        )
        return True

"""Activation flow runner."""
from __future__ import annotations

from typing import Any, Dict, List

from .cancellation import ActivationCancellation
from .constants import VOICE_SESSION_CANCELLED, VOICE_SESSION_COMPLETED
from .services import ActivationFlowServices
from .outcomes import TurnOutcomeHandler
from .routing import RoutedTurnHandler
from .session_setup import start_activation_session
from .turn_input import TurnInputHandler
from ...voice_routing import has_pending_heavy_delegation, pop_completed_heavy_delegation


class ActivationFlowRunner:
    def __init__(self, deps: ActivationFlowServices, cfg: Dict[str, Any], activation: Any):
        self.deps = deps
        self.cfg = cfg
        self.session = start_activation_session(deps, cfg, activation)
        self.conversation_enabled = bool(cfg.get("conversation_mode_enabled", True))
        self.max_turns = max(1, int(cfg.get("conversation_max_turns") or 50))
        self.archive_turns: List[Dict[str, Any]] = []
        self.turn_timings: List[Dict[str, Any]] = []
        self.hermes_history: List[Dict[str, Any]] = []
        self.cancellation = ActivationCancellation(
            deps,
            self.session.visual_state,
            self.session.archive,
            self.archive_turns,
        )
        self.turn_input = TurnInputHandler(
            deps,
            cfg,
            self.session.visual_state,
            self.session.archive,
            self.session.activation_detected_at,
            self.cancellation.requested,
        )
        self.outcomes = TurnOutcomeHandler(
            deps,
            cfg,
            self.session.visual_state,
            self.session.archive,
            self.archive_turns,
            self.turn_timings,
            self.cancellation.requested,
            self.cancellation.stop_if_cancelled,
        )
        self.routed_turns = RoutedTurnHandler(
            deps,
            cfg,
            self.session.visual_state,
            self.session.archive,
            self.archive_turns,
            self.turn_timings,
            self.cancellation.requested,
            self.cancellation.stop_if_cancelled,
        )

    def run(self) -> str:
        turn_index = 1
        while not self.deps.stop.is_set() and turn_index <= self.max_turns:
            completed_delegation = pop_completed_heavy_delegation()
            if completed_delegation is not None:
                result, self.hermes_history, continue_conversation = self.routed_turns.speak_delegated_completion(
                    turn_index,
                    completed_delegation.response,
                    completed_delegation.history,
                    self.conversation_enabled,
                )
                if not continue_conversation:
                    return result
                turn_index += 1
                continue

            self.cfg["_heavy_agent_delegation_pending"] = has_pending_heavy_delegation()
            first_turn = turn_index == 1
            turn_started = self.deps.time.monotonic()
            turn_timing: Dict[str, Any] = {"turn": turn_index}
            self.turn_input.publish_recording_start(turn_index, first_turn)
            if self.cancellation.stop_if_cancelled():
                return VOICE_SESSION_CANCELLED

            turn_input = self.turn_input.record_and_transcribe(turn_index, turn_timing)
            if self.cancellation.stop_if_cancelled():
                return VOICE_SESSION_CANCELLED
            if turn_input is None:
                result = self.outcomes.no_recording(turn_index, first_turn, self.conversation_enabled)
                if result:
                    return result
                continue
            if self.cancellation.stop_if_cancelled():
                return VOICE_SESSION_CANCELLED
            if not turn_input.transcript:
                result = self.outcomes.no_transcript(
                    turn_index,
                    first_turn,
                    self.conversation_enabled,
                    turn_input.archived_command_path,
                    turn_input.command_path,
                )
                if result:
                    return result
                continue

            if self.cfg.get("transcript_only_mode", False):
                return self.outcomes.transcript_only(
                    turn_index,
                    turn_input.transcript,
                    turn_started,
                    turn_timing,
                    turn_input.archived_command_path,
                    turn_input.command_path,
                )

            if self.conversation_enabled and self.deps.is_close_transcript(turn_input.transcript, self.cfg):
                return self.outcomes.close_command(
                    turn_index,
                    turn_input.transcript,
                    turn_started,
                    turn_timing,
                    turn_input.archived_command_path,
                    turn_input.command_path,
                )

            result, self.hermes_history, continue_conversation = self.routed_turns.answer(
                turn_index,
                turn_input.transcript,
                turn_started,
                turn_timing,
                turn_input.archived_command_path,
                turn_input.command_path,
                self.hermes_history,
                self.conversation_enabled,
            )
            if not continue_conversation:
                return result
            turn_index += 1

        if not self.deps.stop.is_set():
            self.deps.update_visualization_state(
                self.session.visual_state,
                status="done",
                message=f"Voice conversation hit the safety limit of {self.max_turns} turns.",
                error="Say the wakeword again to start a new voice session.",
            )
            self.deps.update_activation_archive_metadata(
                self.session.archive,
                status="max_turns_reached",
                close_reason="max_turns",
                turns=self.archive_turns,
            )
        return VOICE_SESSION_COMPLETED


def handle_activation_impl(deps: ActivationFlowServices, cfg: Dict[str, Any], activation: Any) -> str:
    return ActivationFlowRunner(deps, cfg, activation).run()

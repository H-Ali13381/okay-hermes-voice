from __future__ import annotations

from okay_hermes_voice import voice_routing as routing


def test_close_transcript_matches_only_explicit_session_close_commands():
    cfg = {
        "conversation_close_phrases": [
            "close",
            "close voice mode",
            "close conversation",
            "stop listening",
            "end conversation",
        ]
    }

    assert routing.is_close_transcript("close", cfg)
    assert routing.is_close_transcript("Okay Hermes, close voice mode.", cfg)
    assert routing.is_close_transcript("Hermes stop listening", cfg)
    assert routing.is_close_transcript("end conversation", cfg)

    assert not routing.is_close_transcript("close the browser", cfg)
    assert not routing.is_close_transcript("can you close the window after this", cfg)
    assert not routing.is_close_transcript("what is the closest coffee shop", cfg)

def test_command_recording_config_for_followup_disables_start_timeout_without_mutating_original():
    cfg = {
        "speech_start_timeout_seconds": 15.0,
        "conversation_followup_start_timeout_seconds": 0.0,
    }

    first = routing.command_recording_config_for_turn(cfg, turn_index=1)
    followup = routing.command_recording_config_for_turn(cfg, turn_index=2)

    assert first["speech_start_timeout_seconds"] == 15.0
    assert followup["speech_start_timeout_seconds"] == 0.0
    assert cfg["speech_start_timeout_seconds"] == 15.0
    assert followup is not cfg

from __future__ import annotations

from pathlib import Path

from okay_hermes_voice.interaction_router import ACK_TEXT, AckTemplate, AcknowledgementCache


def test_ack_cache_generates_missing_audio_once(tmp_path):
    generated: list[str] = []

    def fake_tts(text: str, out_path: Path) -> Path:
        generated.append(text)
        out_path.write_bytes(b"fake audio")
        return out_path

    cache = AcknowledgementCache(tmp_path, tts_generator=fake_tts, audio_player=lambda path: True)

    first = cache.ensure(AckTemplate.GOT_IT)
    second = cache.ensure(AckTemplate.GOT_IT)

    assert ACK_TEXT[AckTemplate.GOT_IT] == "Okay, I’m on it."
    assert first == second
    assert first.exists()
    assert generated == ["Okay, I’m on it."]

def test_ack_cache_preserves_provider_suffix_and_ignores_mislabeled_wav(tmp_path):
    generated: list[str] = []
    stale = tmp_path / "got_it.wav"
    stale.write_bytes(b"OggS\x00mislabeled opus cache")

    def fake_tts(text: str, out_path: Path) -> Path:
        generated.append(text)
        actual_path = out_path.with_suffix(".ogg")
        actual_path.write_bytes(b"OggS\x00fresh opus cache")
        return actual_path

    cache = AcknowledgementCache(tmp_path, tts_generator=fake_tts, audio_player=lambda path: True)

    first = cache.ensure(AckTemplate.GOT_IT)
    second = cache.ensure(AckTemplate.GOT_IT)

    assert first == tmp_path / "got_it.ogg"
    assert second == first
    assert first.read_bytes().startswith(b"OggS")
    assert generated == ["Okay, I’m on it."]

def test_ack_cache_play_uses_existing_audio(tmp_path):
    played: list[Path] = []
    path = tmp_path / "got_it.wav"
    path.write_bytes(b"fake audio")

    cache = AcknowledgementCache(
        tmp_path,
        tts_generator=lambda text, out_path: out_path,
        audio_player=lambda p: played.append(p) or True,
    )

    assert cache.play(AckTemplate.GOT_IT) is True
    assert played == [path]

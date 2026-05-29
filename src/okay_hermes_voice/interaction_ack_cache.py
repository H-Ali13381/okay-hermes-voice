"""Cached acknowledgement audio for the voice interaction router."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .interaction_types import AckTemplate

ACK_TEXT: dict[AckTemplate, str] = {
    AckTemplate.GOT_IT: "Okay, I’m on it.",
    AckTemplate.CHECKING: "Let me check.",
    AckTemplate.THINKING: "I’ll think through that.",
    AckTemplate.LOOKING_THAT_UP: "I’ll look that up.",
    AckTemplate.WORKING: "Working on it.",
}


ACK_AUDIO_SUFFIXES = (".ogg", ".opus", ".wav", ".mp3", ".flac", ".m4a", ".aac")


class AcknowledgementCache:
    """Generate and play short acknowledgement clips without a router/agent call."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        tts_generator: Callable[[str, Path], Path],
        audio_player: Callable[[Path], bool],
    ) -> None:
        self.cache_dir = cache_dir.expanduser()
        self.tts_generator = tts_generator
        self.audio_player = audio_player

    def _candidate_paths(self, template_id: AckTemplate) -> list[Path]:
        stem = self.cache_dir / template_id.value
        candidates = [stem.with_suffix(suffix) for suffix in ACK_AUDIO_SUFFIXES]
        if self.cache_dir.exists():
            for path in sorted(self.cache_dir.glob(f"{template_id.value}.*")):
                if path not in candidates:
                    candidates.append(path)
        return candidates

    @staticmethod
    def _usable_audio_file(path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size <= 0:
                return False
            header = path.read_bytes()[:12]
        except OSError:
            return False
        suffix = path.suffix.lower()
        if suffix == ".wav" and header.startswith(b"OggS"):
            return False
        if suffix in {".ogg", ".opus"} and header.startswith(b"RIFF"):
            return False
        return True

    def ensure(self, template_id: AckTemplate) -> Path:
        if template_id is AckTemplate.NONE:
            raise ValueError("AckTemplate.NONE has no audio file")
        text = ACK_TEXT[template_id]
        for path in self._candidate_paths(template_id):
            if self._usable_audio_file(path):
                return path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        preferred_path = self.cache_dir / f"{template_id.value}.wav"
        generated_path = Path(self.tts_generator(text, preferred_path)).expanduser()
        if self._usable_audio_file(generated_path):
            return generated_path
        if self._usable_audio_file(preferred_path):
            return preferred_path
        for path in self._candidate_paths(template_id):
            if self._usable_audio_file(path):
                return path
        raise RuntimeError(f"Acknowledgement TTS did not create a usable audio file for {template_id.value}")

    def play(self, template_id: AckTemplate) -> bool:
        if template_id is AckTemplate.NONE:
            return False
        return self.audio_player(self.ensure(template_id))

"""Generate cached acknowledgement TTS audio."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def _generate_ack_tts(text: str, out_path: Path) -> Path:
    """Generate one acknowledgement clip and copy it into the ack cache."""
    from . import text_to_speech_tool
    result_raw = text_to_speech_tool(text)
    try:
        result = json.loads(result_raw)
    except Exception as exc:
        raise RuntimeError(f"TTS returned non-JSON for acknowledgement: {result_raw!r}") from exc
    if not result.get("success") or not result.get("file_path"):
        raise RuntimeError(f"TTS failed for acknowledgement: {result}")
    source = Path(str(result["file_path"])).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = out_path.with_suffix(source.suffix or out_path.suffix)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target


__all__ = ["_generate_ack_tts"]

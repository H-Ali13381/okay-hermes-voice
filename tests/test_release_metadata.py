from __future__ import annotations

import re
import tomllib
from pathlib import Path

import okay_hermes_voice


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_project_metadata():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert okay_hermes_voice.__version__ == version


def test_changelog_tracks_current_version():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{version}]" in changelog
    assert f"releases/tag/v{version}" in changelog

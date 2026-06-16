from __future__ import annotations

import ast
from pathlib import Path


MAX_LEAF_FUNCTIONS = 2

# Entrypoints and orchestration spines are allowed to tell a larger story at the
# package root. Implementation leaves are not.
STRUCTURE_EXEMPTIONS = {
    Path("okay_hermes_voice/daemon_config.py"),
    Path("okay_hermes_voice/hermes_agent_cache.py"),
    Path("okay_hermes_voice/hermes_runtime.py"),
    Path("okay_hermes_voice/interaction_types.py"),
    Path("okay_hermes_voice/native_activation_handler.py"),
    Path("okay_hermes_voice/native_activation_server.py"),
    Path("okay_hermes_voice/wakeword_daemon.py"),
}


def test_src_implementation_files_do_not_collect_many_top_level_functions():
    src_root = Path(__file__).parents[2] / "src"
    oversized = {}

    for path in sorted(src_root.rglob("*.py")):
        relative = path.relative_to(src_root)
        if path.name == "__init__.py" or relative in STRUCTURE_EXEMPTIONS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(functions) > MAX_LEAF_FUNCTIONS:
            oversized[str(relative)] = functions

    assert oversized == {}

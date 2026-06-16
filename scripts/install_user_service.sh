#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hermes_home="${HERMES_HOME:-$HOME/.hermes}"
hermes_repo="${HERMES_REPO:-$hermes_home/hermes-agent}"
python_bin="${PYTHON:-$hermes_repo/venv/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Hermes venv Python not found at $python_bin" >&2
  echo "Set PYTHON=/path/to/python or HERMES_REPO=/path/to/hermes-agent." >&2
  exit 1
fi

mkdir -p "$hermes_home/wakeword" "$hermes_home/logs" "$HOME/.config/systemd/user"
if "$python_bin" -m pip --version >/dev/null 2>&1; then
  "$python_bin" -m pip install -e "$repo_dir"
elif command -v uv >/dev/null 2>&1; then
  uv pip install --python "$python_bin" -e "$repo_dir"
else
  echo "Neither pip in $python_bin nor uv is available for package installation." >&2
  echo "Install pip in the Hermes venv or install uv, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "$hermes_home/wakeword/config.yaml" ]]; then
  cp "$repo_dir/config.example.yaml" "$hermes_home/wakeword/config.yaml"
  echo "Created $hermes_home/wakeword/config.yaml"
else
  echo "Keeping existing $hermes_home/wakeword/config.yaml"
fi

PYTHON="$python_bin" "$repo_dir/native/build_wake_listener.sh" --output "$hermes_home/wakeword/bin/okay-hermes-wake-listener"

cp "$repo_dir/systemd/hermes-wakeword.service" "$HOME/.config/systemd/user/hermes-wakeword.service"
systemctl --user daemon-reload
systemctl --user enable --now hermes-wakeword.service
systemctl --user status hermes-wakeword.service --no-pager --lines=20

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
"$python_bin" -m pip install -e "$repo_dir"

if [[ ! -f "$hermes_home/wakeword/config.yaml" ]]; then
  cp "$repo_dir/config.example.yaml" "$hermes_home/wakeword/config.yaml"
  echo "Created $hermes_home/wakeword/config.yaml"
else
  echo "Keeping existing $hermes_home/wakeword/config.yaml"
fi

cp "$repo_dir/systemd/hermes-wakeword.service" "$HOME/.config/systemd/user/hermes-wakeword.service"
systemctl --user daemon-reload
systemctl --user enable --now hermes-wakeword.service
systemctl --user status hermes-wakeword.service --no-pager --lines=20

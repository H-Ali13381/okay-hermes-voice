#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-${HERMES_REPO:-$HOME/.hermes/hermes-agent}/venv/bin/python}"
output="$repo_dir/native/okay-hermes-wake-listener"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output="$2"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$python_bin" ]]; then
  echo "Python not found: $python_bin" >&2
  exit 1
fi

ort_capi_dir="$($python_bin - <<'PY'
from pathlib import Path
import onnxruntime as ort
print(Path(ort.__file__).parent / 'capi')
PY
)"
ort_lib="$ort_capi_dir/libonnxruntime.so.1.26.0"
if [[ ! -f "$ort_lib" ]]; then
  echo "ONNX Runtime shared library not found: $ort_lib" >&2
  exit 1
fi

out_dir="$(dirname "$output")"
out_lib_dir="$out_dir/lib"
mkdir -p "$out_dir" "$out_lib_dir"
cp "$ort_lib" "$out_lib_dir/libonnxruntime.so.1.26.0.tmp"
mv -f "$out_lib_dir/libonnxruntime.so.1.26.0.tmp" "$out_lib_dir/libonnxruntime.so.1.26.0"
ln -sfn libonnxruntime.so.1.26.0 "$out_lib_dir/libonnxruntime.so.1"

tmp_output="$out_dir/.okay-hermes-wake-listener.$$.tmp"
trap 'rm -f "$tmp_output"' EXIT
cc -std=c11 -O2 -Wall -Wextra -Werror \
  -I"$repo_dir/native/include" \
  "$repo_dir/native/okay-hermes-wake-listener.c" \
  $(pkg-config --cflags --libs libpipewire-0.3) \
  "$out_lib_dir/libonnxruntime.so.1.26.0" \
  -Wl,-rpath,'$ORIGIN/lib' \
  -lm -pthread \
  -o "$tmp_output"
chmod 755 "$tmp_output"
mv -f "$tmp_output" "$output"
trap - EXIT

echo "$output"

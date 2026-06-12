#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/native/wakeword-tray"
BUILD_DIR="$REPO_ROOT/build/wakeword-tray"
BIN_PATH="$HOME/.local/bin/okay-hermes-wakeword-tray"
AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/okay-hermes-wakeword-tray.desktop"

mkdir -p "$BUILD_DIR" "$(dirname "$BIN_PATH")" "$AUTOSTART_DIR"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build "$BUILD_DIR" --config Release
install -m 0755 "$BUILD_DIR/okay-hermes-wakeword-tray" "$BIN_PATH"

cat > "$AUTOSTART_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Okay Hermes Wakeword Tray
Comment=Persistent tray control for the Okay Hermes wakeword listener
Exec=$BIN_PATH
Icon=audio-input-microphone
Terminal=false
Categories=Utility;
StartupNotify=false
X-GNOME-Autostart-enabled=true
EOF
chmod +x "$AUTOSTART_FILE"

if command -v desktop-file-validate >/dev/null 2>&1; then
  desktop-file-validate "$AUTOSTART_FILE"
fi

echo "Installed native Okay Hermes wakeword tray:"
echo "  $BIN_PATH"
echo "  $AUTOSTART_FILE"
echo "Start it now with: $BIN_PATH &"

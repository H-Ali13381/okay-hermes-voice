from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
TRAY_DIR = REPO_ROOT / "native" / "wakeword-tray"
CPP_PATH = TRAY_DIR / "main.cpp"
INSTALLER_PATH = REPO_ROOT / "scripts" / "install_wakeword_tray.sh"


def test_native_tray_is_cpp_qt_not_python_runtime():
    assert CPP_PATH.exists()
    source = CPP_PATH.read_text(encoding="utf-8")

    assert "#include <QSystemTrayIcon>" in source
    assert "#include <QProcess>" in source
    assert "python" not in source.lower()
    assert "PyQt" not in source


def test_tray_menu_is_minimal_right_click_contract():
    source = CPP_PATH.read_text(encoding="utf-8")

    assert '"Turn ON"' in source
    assert '"Turn OFF"' in source
    assert '"Exit"' in source
    assert '"Restart"' not in source
    assert '"Status:"' not in source


def test_tray_has_switching_spinner_state_for_start_and_stop():
    source = CPP_PATH.read_text(encoding="utf-8")

    assert "setSwitchingState" in source
    assert "switchingTimer" in source
    assert "spinnerFrame" in source
    assert "QColor(234, 179, 8)" in source  # yellow loading/closing state
    assert 'setSwitchingState("Starting wakeword daemon' in source
    assert 'setSwitchingState("Stopping wakeword daemon' in source


def test_tray_stays_yellow_while_wakeword_is_active_but_handler_is_not_ready():
    source = CPP_PATH.read_text(encoding="utf-8")

    assert "enum class DaemonState" in source
    assert "DaemonState::Starting" in source
    assert "readyPath()" in source
    assert "QFileInfo(readyPath()).exists()" in source
    assert "journalctl" not in source
    assert "wakeword active but handler is still warming" in source


def test_installer_builds_native_binary_and_autostart_execs_it_directly():
    installer = INSTALLER_PATH.read_text(encoding="utf-8")

    assert "cmake" in installer
    assert "okay-hermes-wakeword-tray" in installer
    assert "python -m okay_hermes_voice.wakeword_tray" not in installer
    assert "PYTHONPATH" not in installer
    assert "Exec=$BIN_PATH" in installer

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

REPO_ROOT = Path(__file__).parents[2]
TRAY_DIR = REPO_ROOT / "native" / "wakeword-tray"
CPP_PATH = TRAY_DIR / "main.cpp"
STATE_HEADER_PATH = TRAY_DIR / "tray_state.h"
INSTALLER_PATH = REPO_ROOT / "scripts" / "install_wakeword_tray.sh"


def tray_source() -> str:
    return CPP_PATH.read_text(encoding="utf-8")


def tray_code() -> str:
    parts = [tray_source()]
    if STATE_HEADER_PATH.exists():
        parts.append(STATE_HEADER_PATH.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_native_tray_is_cpp_qt_not_python_runtime():
    assert CPP_PATH.exists()
    source = tray_code()

    assert "QSystemTrayIcon" in source
    assert "QProcess" not in source
    assert "python" not in source.lower()
    assert "PyQt" not in source


def test_tray_menu_is_minimal_right_click_contract():
    source = tray_code()

    assert '"Turn ON"' in source
    assert '"Turn OFF"' in source
    assert '"Exit"' in source
    assert '"Restart"' not in source
    assert '"Status:"' not in source


def test_tray_has_switching_spinner_state_for_start_and_stop():
    source = tray_code()

    assert "setSwitchingState" in source
    assert "switchingTimer" in source
    assert "spinnerFrame" in source
    assert "Starting wakeword daemon" in source
    assert "Stopping wakeword daemon" in source


def test_tray_stays_yellow_while_wakeword_is_active_but_handler_is_not_ready():
    source = tray_code()

    assert "enum class DaemonState" in source
    assert "DaemonState::Starting" in source
    assert "readyPath()" in source
    assert "QFileInfo(readyPath()).exists()" in source
    assert "journalctl" not in source
    assert "wakeword active but handler is still warming" in source


def test_tray_turns_gray_when_no_microphone_or_capture_frames_are_available():
    source = tray_code()

    assert "DaemonState::NoMicrophone" in source
    assert "CaptureHealth::Unhealthy" in source
    assert "captureStatusChanged" in source
    assert "defaultSourceChanged" in source
    assert "sourceAdded" in source
    assert "no microphone available" in source
    assert "turnOnAction->setEnabled(false)" in source


def test_tray_state_is_event_driven_not_periodically_polled():
    source = tray_code()

    assert "QDBusConnection" in source
    assert "QFileSystemWatcher" in source
    assert "PulseAudioQt" in source
    assert "pollTimer" not in source
    assert '"is-active"' not in source
    assert '"wpctl"' not in source
    assert "systemdStateChanged" in source
    assert "readyMarkerChanged" in source
    assert "audioStateChanged" in source


def test_tray_build_links_event_driven_backends():
    cmake = (TRAY_DIR / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "Qt6 REQUIRED COMPONENTS Widgets DBus" in cmake
    assert "KF6PulseAudioQt" in cmake
    assert "Qt6::DBus" in cmake
    assert "KF6::PulseAudioQt" in cmake


def test_systemd_signal_setup_retries_failed_connections():
    source = tray_code()

    assert "systemdRetryTimer" in source
    assert "scheduleSystemdWatcherRetry" in source
    assert "QDBusPendingReply<void>" in source
    assert "isValid()" in source
    assert "QDBusConnection::sessionBus().connect" in source
    assert "unitSignalPaths.append(path)" in source
    assert "scheduleSystemdWatcherRetry();" in source


def test_systemd_calls_are_async_not_gui_thread_blocking():
    source = tray_code()

    assert "QDBusPendingCallWatcher" in source
    assert "asyncCall(" in source
    assert "requestUnitStates" in source
    assert "requestUnitState" in source
    assert "runSystemdCommandsAsync" in source
    assert "QDBusPendingReply<QDBusObjectPath>" in source
    assert ".call(" not in source


def test_audio_server_signals_reconnect_after_context_ready_changes():
    source = tray_code()

    assert "reconnectAudioServerSignals" in source
    assert "watchedAudioServer" in source
    assert "defaultSourceConnection" in source
    assert "serverUpdatedConnection" in source
    assert "QObject::disconnect(defaultSourceConnection)" in source
    assert "PulseAudioQt::Context::stateChanged" in source
    assert "reconnectAudioServerSignals();" in source


def test_tray_pure_state_helpers_compile_and_match_contract(tmp_path):
    if not shutil.which("pkg-config") or not shutil.which("c++"):
        pytest.skip("native C++ helper test requires pkg-config and c++")
    pkg = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "Qt6Core"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pkg.returncode != 0:
        pytest.skip("Qt6Core pkg-config metadata is not available")

    assert STATE_HEADER_PATH.exists()
    test_cpp = tmp_path / "tray_state_test.cpp"
    test_cpp.write_text(
        textwrap.dedent(
            """
            #include <cassert>
            #include <QString>
            #include "@STATE_HEADER_PATH@"

            int main() {
                using okay_hermes_tray::CaptureHealth;
                using okay_hermes_tray::DaemonState;
                assert(okay_hermes_tray::systemdUnitObjectPath(QStringLiteral("hermes-wakeword.service")) == QStringLiteral("/org/freedesktop/systemd1/unit/hermes_2dwakeword_2eservice"));
                assert(okay_hermes_tray::systemdUnitObjectPath(QStringLiteral("hermes-voice-handler.service")) == QStringLiteral("/org/freedesktop/systemd1/unit/hermes_2dvoice_2dhandler_2eservice"));
                assert(okay_hermes_tray::captureHealthFromStatusText(QStringLiteral("healthy")) == CaptureHealth::Healthy);
                assert(okay_hermes_tray::captureHealthFromStatusText(QStringLiteral(" healthy\\n")) == CaptureHealth::Healthy);
                assert(okay_hermes_tray::captureHealthFromStatusText(QStringLiteral("unhealthy")) == CaptureHealth::Unhealthy);
                assert(okay_hermes_tray::captureHealthFromStatusText(QStringLiteral("starting")) == CaptureHealth::Unknown);
                assert(okay_hermes_tray::captureHealthFromStatusText(QStringLiteral("")) == CaptureHealth::Unknown);
                assert(okay_hermes_tray::captureHealthFromStatusText(QStringLiteral("garbage")) == CaptureHealth::Unknown);
                struct StateCase {
                    bool microphoneAvailable;
                    CaptureHealth captureHealth;
                    bool wakewordActive;
                    bool handlerActive;
                    bool handlerReady;
                    DaemonState expected;
                };

                const StateCase cases[] = {
                    {false, CaptureHealth::Unknown, false, false, false, DaemonState::NoMicrophone},
                    {false, CaptureHealth::Healthy, true, true, true, DaemonState::NoMicrophone},
                    {false, CaptureHealth::Unhealthy, true, true, true, DaemonState::NoMicrophone},
                    {true, CaptureHealth::Unknown, false, false, false, DaemonState::Off},
                    {true, CaptureHealth::Unhealthy, false, false, false, DaemonState::Off},
                    {true, CaptureHealth::Unhealthy, true, false, false, DaemonState::NoMicrophone},
                    {true, CaptureHealth::Unhealthy, true, true, true, DaemonState::NoMicrophone},
                    {true, CaptureHealth::Unknown, false, true, false, DaemonState::Starting},
                    {true, CaptureHealth::Healthy, false, true, true, DaemonState::Starting},
                    {true, CaptureHealth::Unknown, true, false, false, DaemonState::Starting},
                    {true, CaptureHealth::Healthy, true, false, true, DaemonState::Starting},
                    {true, CaptureHealth::Unknown, true, true, false, DaemonState::Starting},
                    {true, CaptureHealth::Healthy, true, true, false, DaemonState::Starting},
                    {true, CaptureHealth::Unknown, true, true, true, DaemonState::Starting},
                    {true, CaptureHealth::Healthy, true, true, true, DaemonState::On},
                };

                for (const auto& item : cases) {
                    assert(okay_hermes_tray::stateFromInputs(
                        item.microphoneAvailable,
                        item.captureHealth,
                        item.wakewordActive,
                        item.handlerActive,
                        item.handlerReady) == item.expected);
                }
                return 0;
            }
            """
        ).replace("@STATE_HEADER_PATH@", str(STATE_HEADER_PATH)),
        encoding="utf-8",
    )
    binary = tmp_path / "tray_state_test"
    command = ["c++", "-std=c++17", "-fPIC", str(test_cpp), "-o", str(binary), *pkg.stdout.split()]
    subprocess.run(command, text=True, capture_output=True, check=True)
    subprocess.run([str(binary)], check=True)


def test_tray_cmake_configures_and_builds_in_temp_dir(tmp_path):
    if not shutil.which("cmake"):
        pytest.skip("native tray build test requires cmake")

    build_dir = tmp_path / "wakeword-tray-build"
    configure_command = [
        "cmake",
        "-S",
        str(TRAY_DIR),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if shutil.which("ninja"):
        configure_command.extend(["-G", "Ninja"])

    subprocess.run(configure_command, text=True, capture_output=True, check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert (build_dir / "okay-hermes-wakeword-tray").exists()


def test_installer_builds_native_binary_and_autostart_execs_it_directly():
    installer = INSTALLER_PATH.read_text(encoding="utf-8")

    assert "cmake" in installer
    assert "okay-hermes-wakeword-tray" in installer
    assert "python -m okay_hermes_voice.wakeword_tray" not in installer
    assert "PYTHONPATH" not in installer
    assert "Exec=$BIN_PATH" in installer

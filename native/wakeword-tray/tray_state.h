#pragma once

#include <QChar>
#include <QLatin1Char>
#include <QString>
#include <QStringView>

namespace okay_hermes_tray {

enum class DaemonState { Off, Starting, On, NoMicrophone };
enum class CaptureHealth { Unknown, Healthy, Unhealthy };

inline CaptureHealth captureHealthFromStatusText(QStringView text) {
    const QString status = text.toString().trimmed();
    if (status == QStringLiteral("healthy")) {
        return CaptureHealth::Healthy;
    }
    if (status == QStringLiteral("unhealthy")) {
        return CaptureHealth::Unhealthy;
    }
    return CaptureHealth::Unknown;
}

inline QString systemdUnitObjectPath(QStringView unit) {
    QString escaped;
    escaped.reserve(unit.size() * 3);
    for (const QChar ch : unit) {
        if (ch.isLetterOrNumber()) {
            escaped.append(ch);
        } else {
            escaped.append(QStringLiteral("_%1").arg(static_cast<uint>(ch.unicode()), 2, 16, QLatin1Char('0')));
        }
    }
    return QStringLiteral("/org/freedesktop/systemd1/unit/") + escaped;
}

inline DaemonState stateFromInputs(
    bool microphoneAvailable,
    CaptureHealth captureHealth,
    bool wakewordActive,
    bool handlerActive,
    bool handlerReady) {
    if (!microphoneAvailable) {
        return DaemonState::NoMicrophone;
    }
    if (wakewordActive && captureHealth == CaptureHealth::Unhealthy) {
        return DaemonState::NoMicrophone;
    }
    if (wakewordActive && captureHealth == CaptureHealth::Unknown) {
        return DaemonState::Starting;
    }
    if (wakewordActive && handlerActive && handlerReady) {
        return DaemonState::On;
    }
    if (wakewordActive || handlerActive) {
        return DaemonState::Starting;
    }
    return DaemonState::Off;
}

}  // namespace okay_hermes_tray

#pragma once

#include <QChar>
#include <QLatin1Char>
#include <QString>
#include <QStringView>

namespace okay_hermes_tray {

enum class DaemonState { Off, Starting, On, NoMicrophone };

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

inline DaemonState stateFromInputs(bool microphoneAvailable, bool wakewordActive, bool handlerActive, bool handlerReady) {
    if (!microphoneAvailable) {
        return DaemonState::NoMicrophone;
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

#include <QAction>
#include <QApplication>
#include <QColor>
#include <QDBusConnection>
#include <QDBusError>
#include <QDBusInterface>
#include <QDBusObjectPath>
#include <QDBusPendingCall>
#include <QDBusPendingCallWatcher>
#include <QDBusPendingReply>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QIcon>
#include <QMenu>
#include <QMetaObject>
#include <QObject>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QStringList>
#include <QSystemTrayIcon>
#include <QTimer>
#include <QVariant>
#include <QVariantMap>

#include <PulseAudioQt/Context>
#include <PulseAudioQt/Server>
#include <PulseAudioQt/Source>

#include "tray_state.h"

#include <memory>

namespace {

constexpr const char* kWakewordUnit = "hermes-wakeword.service";
constexpr const char* kHandlerUnit = "hermes-voice-handler.service";
constexpr const char* kReadyMarkerRelativePath = ".hermes/wakeword/native-handler.ready";
constexpr const char* kCaptureStatusRelativePath = ".hermes/wakeword/native-listener.capture-status";
constexpr const char* kSystemdService = "org.freedesktop.systemd1";
constexpr const char* kSystemdManagerPath = "/org/freedesktop/systemd1";
constexpr const char* kSystemdManagerInterface = "org.freedesktop.systemd1.Manager";
constexpr const char* kSystemdUnitInterface = "org.freedesktop.systemd1.Unit";
constexpr const char* kDBusPropertiesInterface = "org.freedesktop.DBus.Properties";
constexpr int kIconSize = 64;

QIcon stateIcon(const QColor& color, int spinnerFrame = -1) {
    QPixmap pixmap(kIconSize, kIconSize);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(17, 24, 39));
    painter.drawEllipse(2, 2, 60, 60);
    painter.setBrush(color);
    painter.drawEllipse(7, 7, 50, 50);

    painter.setBrush(Qt::white);
    painter.drawRoundedRect(24, 12, 16, 28, 8, 8);
    QPen micPen(Qt::white, 5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    painter.setPen(micPen);
    painter.setBrush(Qt::NoBrush);
    painter.drawArc(18, 22, 28, 25, 180 * 16, 180 * 16);
    painter.drawLine(32, 44, 32, 53);
    painter.drawLine(24, 53, 40, 53);

    if (spinnerFrame >= 0) {
        QPen spinnerPen(Qt::white, 6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        painter.setPen(spinnerPen);
        const int start = (spinnerFrame * 40) % 360;
        painter.drawArc(8, 8, 48, 48, start * 16, 95 * 16);
    }

    return QIcon(pixmap);
}

QString readyPath() {
    return QDir::home().filePath(kReadyMarkerRelativePath);
}

QString readyDirPath() {
    return QFileInfo(readyPath()).absolutePath();
}

QString captureStatusPath() {
    return QDir::home().filePath(kCaptureStatusRelativePath);
}

QString captureStatusDirPath() {
    return QFileInfo(captureStatusPath()).absolutePath();
}

}  // namespace

using okay_hermes_tray::CaptureHealth;
using okay_hermes_tray::DaemonState;

class TrayController : public QObject {
    Q_OBJECT

public:
    explicit TrayController(QApplication& app)
        : QObject(&app),
          app(app),
          tray(new QSystemTrayIcon()),
          menu(new QMenu()),
          switchingTimer(new QTimer(this)),
          systemdRetryTimer(new QTimer(this)),
          readyWatcher(new QFileSystemWatcher(this)),
          captureStatusWatcher(new QFileSystemWatcher(this)) {
        turnOnAction = menu->addAction("Turn ON");
        turnOffAction = menu->addAction("Turn OFF");
        menu->addSeparator();
        exitAction = menu->addAction("Exit");

        tray->setContextMenu(menu.get());
        tray->setIcon(stateIcon(QColor(234, 179, 8), 0));
        tray->setToolTip("Okay Hermes wakeword: loading…");

        QObject::connect(turnOnAction, &QAction::triggered, this, &TrayController::startDaemon);
        QObject::connect(turnOffAction, &QAction::triggered, this, &TrayController::stopDaemon);
        QObject::connect(exitAction, &QAction::triggered, &app, &QApplication::quit);
        QObject::connect(switchingTimer, &QTimer::timeout, this, &TrayController::advanceSpinner);
        systemdRetryTimer->setSingleShot(true);
        systemdRetryTimer->setInterval(3000);
        QObject::connect(systemdRetryTimer, &QTimer::timeout, this, &TrayController::setupSystemdWatchers);
        QObject::connect(readyWatcher, &QFileSystemWatcher::directoryChanged, this, &TrayController::readyMarkerChanged);
        QObject::connect(readyWatcher, &QFileSystemWatcher::fileChanged, this, &TrayController::readyMarkerChanged);
        QObject::connect(captureStatusWatcher, &QFileSystemWatcher::directoryChanged, this, &TrayController::captureStatusChanged);
        QObject::connect(captureStatusWatcher, &QFileSystemWatcher::fileChanged, this, &TrayController::captureStatusChanged);

        setupSystemdWatchers();
        requestUnitStates();
        setupReadyWatcher();
        setupCaptureStatusWatcher();
        setupAudioWatcher();
        setSwitchingState("Loading wakeword daemon state…");
        QTimer::singleShot(150, this, &TrayController::refreshState);
    }

    int run() {
        if (!QSystemTrayIcon::isSystemTrayAvailable()) {
            qCritical("No system tray is available in this desktop session.");
            return 1;
        }
        tray->show();
        return app.exec();
    }

private Q_SLOTS:
    void refreshState() {
        if (systemdCommandInFlight) {
            return;
        }
        const DaemonState state = daemonState();
        if (state == DaemonState::NoMicrophone) {
            switchingTimer->stop();
            turnOnAction->setEnabled(false);
            turnOffAction->setEnabled(anyControlledUnitActive());
            tray->setToolTip("Okay Hermes wakeword: no microphone available / no capture frames");
            tray->setIcon(stateIcon(QColor(107, 114, 128)));
            return;
        }
        if (state == DaemonState::Starting) {
            turnOnAction->setEnabled(false);
            turnOffAction->setEnabled(false);
            tray->setToolTip("Okay Hermes wakeword: wakeword active but handler is still warming…");
            if (!switchingTimer->isActive()) {
                switchingTimer->start(150);
            }
            tray->setIcon(stateIcon(QColor(234, 179, 8), spinnerFrame));
            return;
        }

        switchingTimer->stop();
        const bool active = state == DaemonState::On;
        turnOnAction->setEnabled(!active);
        turnOffAction->setEnabled(active);
        tray->setIcon(stateIcon(active ? QColor(34, 197, 94) : QColor(239, 68, 68)));
        tray->setToolTip(active ? "Okay Hermes wakeword: ON" : "Okay Hermes wakeword: OFF");
    }

    void systemdStateChanged() {
        const bool wakewordConnected = connectUnitSignals(kWakewordUnit);
        const bool handlerConnected = connectUnitSignals(kHandlerUnit);
        if (!wakewordConnected || !handlerConnected) {
            scheduleSystemdWatcherRetry();
        }
        requestUnitStates();
    }

    void systemdJobRemoved(uint, const QDBusObjectPath&, const QString&, const QString&) {
        systemdStateChanged();
    }

    void systemdUnitChanged(const QString&, const QDBusObjectPath&) {
        systemdStateChanged();
    }

    void systemdPropertiesChanged(const QString&, const QVariantMap&, const QStringList&) {
        systemdStateChanged();
    }

    void readyMarkerChanged() {
        setupReadyWatcher();
        refreshState();
    }

    void captureStatusChanged() {
        setupCaptureStatusWatcher();
        refreshState();
    }

    void audioStateChanged() {
        refreshState();
    }

    void advanceSpinner() {
        ++spinnerFrame;
        tray->setIcon(stateIcon(QColor(234, 179, 8), spinnerFrame));
    }

private:
    void startDaemon() {
        if (!microphoneAvailable()) {
            refreshState();
            return;
        }
        setSwitchingState("Starting wakeword daemon…");
        runSystemdCommandsAsync("StartUnit", {kHandlerUnit, kWakewordUnit});
    }

    void stopDaemon() {
        setSwitchingState("Stopping wakeword daemon…");
        runSystemdCommandsAsync("StopUnit", {kWakewordUnit, kHandlerUnit});
    }

    void setSwitchingState(const QString& tooltip) {
        spinnerFrame = 0;
        turnOnAction->setEnabled(false);
        turnOffAction->setEnabled(false);
        tray->setToolTip(tooltip);
        tray->setIcon(stateIcon(QColor(234, 179, 8), spinnerFrame));
        switchingTimer->start(150);
    }

    QDBusInterface systemdManager() const {
        return QDBusInterface(kSystemdService, kSystemdManagerPath, kSystemdManagerInterface, QDBusConnection::sessionBus());
    }

    void requestUnitStates() {
        requestUnitState(kWakewordUnit);
        requestUnitState(kHandlerUnit);
    }

    void requestUnitState(const QString& unit) {
        QDBusPendingCall pending = systemdManager().asyncCall("GetUnit", unit);
        auto* watcher = new QDBusPendingCallWatcher(pending, this);
        QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, unit](QDBusPendingCallWatcher* finished) {
            QDBusPendingReply<QDBusObjectPath> reply(*finished);
            finished->deleteLater();
            if (!reply.isValid() || reply.value().path().isEmpty()) {
                setCachedUnitActive(unit, false);
                refreshState();
                return;
            }
            requestUnitActiveState(unit, reply.value().path());
        });
    }

    void requestUnitActiveState(const QString& unit, const QString& path) {
        QDBusInterface properties(kSystemdService, path, kDBusPropertiesInterface, QDBusConnection::sessionBus());
        QDBusPendingCall pending = properties.asyncCall("Get", kSystemdUnitInterface, "ActiveState");
        auto* watcher = new QDBusPendingCallWatcher(pending, this);
        QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, unit](QDBusPendingCallWatcher* finished) {
            QDBusPendingReply<QVariant> reply(*finished);
            finished->deleteLater();
            setCachedUnitActive(unit, reply.isValid() && reply.value().toString() == "active");
            refreshState();
        });
    }

    void setCachedUnitActive(const QString& unit, bool active) {
        if (unit == QLatin1String(kWakewordUnit)) {
            wakewordActive = active;
        } else if (unit == QLatin1String(kHandlerUnit)) {
            handlerActive = active;
        }
    }

    bool handlerIsReady() const {
        return handlerActive && QFileInfo(readyPath()).exists();
    }

    bool microphoneAvailable() const {
        auto* context = PulseAudioQt::Context::instance();
        if (!context || context->state() != PulseAudioQt::Context::State::Ready || !context->server()) {
            return false;
        }
        auto* source = context->server()->defaultSource();
        if (!source) {
            return false;
        }
        const QVariantMap properties = source->pulseProperties();
        const QString mediaClass = properties.value(QStringLiteral("media.class")).toString();
        const QString deviceClass = properties.value(QStringLiteral("device.class")).toString();
        return mediaClass == QStringLiteral("Audio/Source") && deviceClass != QStringLiteral("monitor") && !source->isVirtualDevice();
    }

    CaptureHealth captureHealth() const {
        QFile file(captureStatusPath());
        if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return CaptureHealth::Unknown;
        }
        const QString status = QString::fromUtf8(file.read(64));
        return okay_hermes_tray::captureHealthFromStatusText(status);
    }

    bool anyControlledUnitActive() const {
        return wakewordActive || handlerActive;
    }

    DaemonState daemonState() const {
        return okay_hermes_tray::stateFromInputs(microphoneAvailable(), captureHealth(), wakewordActive, handlerActive, handlerIsReady());
    }

    void runSystemdCommandsAsync(const QString& method, const QStringList& units) {
        if (systemdCommandInFlight) {
            return;
        }
        systemdCommandInFlight = true;
        runSystemdCommandAt(method, units, 0);
    }

    void runSystemdCommandAt(const QString& method, const QStringList& units, int index) {
        if (index >= units.size()) {
            systemdCommandInFlight = false;
            requestUnitStates();
            return;
        }

        QDBusPendingCall pending = systemdManager().asyncCall(method, units.at(index), QStringLiteral("replace"));
        auto* watcher = new QDBusPendingCallWatcher(pending, this);
        QObject::connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, method, units, index](QDBusPendingCallWatcher* finished) {
            QDBusPendingReply<QDBusObjectPath> reply(*finished);
            finished->deleteLater();
            if (!reply.isValid()) {
                const QDBusError error = reply.error();
                tray->showMessage(
                    "Okay Hermes wakeword",
                    QStringLiteral("systemd %1 failed for %2: %3").arg(method, units.at(index), error.message()),
                    QSystemTrayIcon::Warning,
                    4000);
                systemdCommandInFlight = false;
                requestUnitStates();
                return;
            }
            runSystemdCommandAt(method, units, index + 1);
        });
    }

    void scheduleSystemdWatcherRetry() {
        if (!systemdRetryTimer->isActive()) {
            systemdRetryTimer->start();
        }
    }

    void setupSystemdWatchers() {
        bool allConnected = true;
        QDBusPendingCall subscribePending = systemdManager().asyncCall("Subscribe");
        auto* subscribeWatcher = new QDBusPendingCallWatcher(subscribePending, this);
        QObject::connect(subscribeWatcher, &QDBusPendingCallWatcher::finished, this, [this](QDBusPendingCallWatcher* finished) {
            QDBusPendingReply<void> subscribeReply(*finished);
            finished->deleteLater();
            if (!subscribeReply.isValid()) {
                scheduleSystemdWatcherRetry();
            }
        });

        auto bus = QDBusConnection::sessionBus();
        if (!jobRemovedSignalConnected) {
            jobRemovedSignalConnected = bus.connect(kSystemdService, kSystemdManagerPath, kSystemdManagerInterface, "JobRemoved", this, SLOT(systemdJobRemoved(uint,QDBusObjectPath,QString,QString)));
        }
        if (!unitNewSignalConnected) {
            unitNewSignalConnected = bus.connect(kSystemdService, kSystemdManagerPath, kSystemdManagerInterface, "UnitNew", this, SLOT(systemdUnitChanged(QString,QDBusObjectPath)));
        }
        if (!unitRemovedSignalConnected) {
            unitRemovedSignalConnected = bus.connect(kSystemdService, kSystemdManagerPath, kSystemdManagerInterface, "UnitRemoved", this, SLOT(systemdUnitChanged(QString,QDBusObjectPath)));
        }

        allConnected = allConnected && jobRemovedSignalConnected && unitNewSignalConnected && unitRemovedSignalConnected;
        allConnected = connectUnitSignals(kWakewordUnit) && allConnected;
        allConnected = connectUnitSignals(kHandlerUnit) && allConnected;

        if (!allConnected) {
            scheduleSystemdWatcherRetry();
        } else {
            systemdRetryTimer->stop();
        }
    }

    bool connectUnitSignals(const QString& unit) {
        const QString path = okay_hermes_tray::systemdUnitObjectPath(unit);
        if (unitSignalPaths.contains(path)) {
            return true;
        }
        const bool connected = QDBusConnection::sessionBus().connect(
            kSystemdService,
            path,
            kDBusPropertiesInterface,
            "PropertiesChanged",
            this,
            SLOT(systemdPropertiesChanged(QString,QVariantMap,QStringList)));
        if (!connected) {
            return false;
        }
        unitSignalPaths.append(path);
        return true;
    }

    void setupReadyWatcher() {
        const QString dir = readyDirPath();
        QDir().mkpath(dir);
        if (!readyWatcher->directories().contains(dir)) {
            readyWatcher->addPath(dir);
        }
        const QString marker = readyPath();
        if (QFileInfo(marker).exists() && !readyWatcher->files().contains(marker)) {
            readyWatcher->addPath(marker);
        }
    }

    void setupCaptureStatusWatcher() {
        const QString dir = captureStatusDirPath();
        QDir().mkpath(dir);
        if (!captureStatusWatcher->directories().contains(dir)) {
            captureStatusWatcher->addPath(dir);
        }
        const QString marker = captureStatusPath();
        if (QFileInfo(marker).exists() && !captureStatusWatcher->files().contains(marker)) {
            captureStatusWatcher->addPath(marker);
        }
    }

    void setupAudioWatcher() {
        auto* context = PulseAudioQt::Context::instance();
        QObject::connect(context, &PulseAudioQt::Context::stateChanged, this, [this] {
            reconnectAudioServerSignals();
            audioStateChanged();
        });
        QObject::connect(context, &PulseAudioQt::Context::sourceAdded, this, [this](PulseAudioQt::Source*) { audioStateChanged(); });
        QObject::connect(context, &PulseAudioQt::Context::sourceRemoved, this, [this](PulseAudioQt::Source*) { audioStateChanged(); });
        reconnectAudioServerSignals();
    }

    void reconnectAudioServerSignals() {
        auto* context = PulseAudioQt::Context::instance();
        auto* server = (context && context->state() == PulseAudioQt::Context::State::Ready) ? context->server() : nullptr;
        if (server == watchedAudioServer) {
            return;
        }

        QObject::disconnect(defaultSourceConnection);
        QObject::disconnect(serverUpdatedConnection);
        defaultSourceConnection = QMetaObject::Connection();
        serverUpdatedConnection = QMetaObject::Connection();
        watchedAudioServer = server;

        if (!watchedAudioServer) {
            return;
        }

        defaultSourceConnection = QObject::connect(
            watchedAudioServer,
            &PulseAudioQt::Server::defaultSourceChanged,
            this,
            [this](PulseAudioQt::Source*) { audioStateChanged(); });
        serverUpdatedConnection = QObject::connect(watchedAudioServer, &PulseAudioQt::Server::updated, this, &TrayController::audioStateChanged);
    }

    QApplication& app;
    std::unique_ptr<QSystemTrayIcon> tray;
    std::unique_ptr<QMenu> menu;
    QAction* turnOnAction = nullptr;
    QAction* turnOffAction = nullptr;
    QAction* exitAction = nullptr;
    QTimer* switchingTimer = nullptr;
    QTimer* systemdRetryTimer = nullptr;
    QFileSystemWatcher* readyWatcher = nullptr;
    QFileSystemWatcher* captureStatusWatcher = nullptr;
    QStringList unitSignalPaths;
    bool jobRemovedSignalConnected = false;
    bool unitNewSignalConnected = false;
    bool unitRemovedSignalConnected = false;
    bool wakewordActive = false;
    bool handlerActive = false;
    bool systemdCommandInFlight = false;
    PulseAudioQt::Server* watchedAudioServer = nullptr;
    QMetaObject::Connection defaultSourceConnection;
    QMetaObject::Connection serverUpdatedConnection;
    int spinnerFrame = 0;
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("Okay Hermes Wakeword Tray");
    QApplication::setQuitOnLastWindowClosed(false);
    PulseAudioQt::Context::setApplicationId(QStringLiteral("okay-hermes-wakeword-tray"));

    TrayController controller(app);
    return controller.run();
}

#include "main.moc"

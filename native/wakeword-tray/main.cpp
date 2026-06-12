#include <QAction>
#include <QApplication>
#include <QColor>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QIcon>
#include <QMenu>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QProcess>
#include <QStringList>
#include <QSystemTrayIcon>
#include <QTimer>

#include <array>
#include <memory>

namespace {

constexpr const char* kWakewordUnit = "hermes-wakeword.service";
constexpr const char* kHandlerUnit = "hermes-voice-handler.service";
constexpr const char* kReadyMarkerRelativePath = ".hermes/wakeword/native-handler.ready";
constexpr int kIconSize = 64;

enum class DaemonState { Off, Starting, On };

bool commandSucceeds(const QString& program, const QStringList& args, int timeoutMs = 1500) {
    QProcess process;
    process.start(program, args);
    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        process.waitForFinished(250);
        return false;
    }
    return process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0;
}

bool unitIsActive(const QString& unit) {
    return commandSucceeds("systemctl", {"--user", "is-active", "--quiet", unit});
}

QString readyPath() {
    return QDir::home().filePath(kReadyMarkerRelativePath);
}

bool handlerIsReady() {
    if (!unitIsActive(kHandlerUnit)) {
        return false;
    }
    return QFileInfo(readyPath()).exists();
}

DaemonState daemonState() {
    const bool wakewordActive = unitIsActive(kWakewordUnit);
    const bool handlerActive = unitIsActive(kHandlerUnit);

    if (wakewordActive && handlerActive && handlerIsReady()) {
        return DaemonState::On;
    }
    if (wakewordActive || handlerActive) {
        return DaemonState::Starting;
    }
    return DaemonState::Off;
}

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

class TrayController {
public:
    explicit TrayController(QApplication& app)
        : app(app), tray(new QSystemTrayIcon()), menu(new QMenu()), pollTimer(new QTimer()), switchingTimer(new QTimer()) {
        turnOnAction = menu->addAction("Turn ON");
        turnOffAction = menu->addAction("Turn OFF");
        menu->addSeparator();
        exitAction = menu->addAction("Exit");

        tray->setContextMenu(menu.get());
        tray->setIcon(stateIcon(QColor(234, 179, 8), 0));
        tray->setToolTip("Okay Hermes wakeword: loading…");

        QObject::connect(turnOnAction, &QAction::triggered, [&] { startDaemon(); });
        QObject::connect(turnOffAction, &QAction::triggered, [&] { stopDaemon(); });
        QObject::connect(exitAction, &QAction::triggered, [&] { app.quit(); });
        QObject::connect(pollTimer.get(), &QTimer::timeout, [&] { refreshState(); });
        QObject::connect(switchingTimer.get(), &QTimer::timeout, [&] { advanceSpinner(); });

        setSwitchingState("Loading wakeword daemon state…");
        QTimer::singleShot(150, [&] { refreshState(); });
        pollTimer->start(2000);
    }

    int run() {
        if (!QSystemTrayIcon::isSystemTrayAvailable()) {
            qCritical("No system tray is available in this desktop session.");
            return 1;
        }
        tray->show();
        return app.exec();
    }

private:
    void startDaemon() {
        setSwitchingState("Starting wakeword daemon…");
        runSystemctl({"--user", "start", kHandlerUnit, kWakewordUnit});
    }

    void stopDaemon() {
        setSwitchingState("Stopping wakeword daemon…");
        runSystemctl({"--user", "stop", kWakewordUnit, kHandlerUnit});
    }

    void runSystemctl(const QStringList& args) {
        if (activeProcess) {
            return;
        }
        activeProcess = std::make_unique<QProcess>();
        QObject::connect(activeProcess.get(), &QProcess::finished, [this](int exitCode, QProcess::ExitStatus exitStatus) {
            const QString stderrText = QString::fromLocal8Bit(activeProcess->readAllStandardError()).trimmed();
            activeProcess.reset();
            if (exitStatus != QProcess::NormalExit || exitCode != 0) {
                tray->showMessage("Okay Hermes wakeword", stderrText.isEmpty() ? "systemctl failed" : stderrText, QSystemTrayIcon::Warning, 4000);
            }
            refreshState();
        });
        activeProcess->start("systemctl", args);
    }

    void refreshState() {
        if (activeProcess) {
            return;
        }
        const DaemonState state = daemonState();
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

    void setSwitchingState(const QString& tooltip) {
        spinnerFrame = 0;
        turnOnAction->setEnabled(false);
        turnOffAction->setEnabled(false);
        tray->setToolTip(tooltip);
        tray->setIcon(stateIcon(QColor(234, 179, 8), spinnerFrame));
        switchingTimer->start(150);
    }

    void advanceSpinner() {
        ++spinnerFrame;
        tray->setIcon(stateIcon(QColor(234, 179, 8), spinnerFrame));
    }

    QApplication& app;
    std::unique_ptr<QSystemTrayIcon> tray;
    std::unique_ptr<QMenu> menu;
    QAction* turnOnAction = nullptr;
    QAction* turnOffAction = nullptr;
    QAction* exitAction = nullptr;
    std::unique_ptr<QTimer> pollTimer;
    std::unique_ptr<QTimer> switchingTimer;
    std::unique_ptr<QProcess> activeProcess;
    int spinnerFrame = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("Okay Hermes Wakeword Tray");
    QApplication::setQuitOnLastWindowClosed(false);

    TrayController controller(app);
    return controller.run();
}

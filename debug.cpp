#include "debug.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include <QDateTime>
#include <QTextCodec>

#ifdef DEBUG_TIMING
QTime __debugTimer;
unsigned int __lastTime;
unsigned int __totalTime;
#endif

QStringList __dbg_classes;
QStringList __dbg_classes_info;
QStringList __dbg_classes_log;
QStringList __dbg_classes_logv;

void changeDebug(QString debug, int defaultLevel)
{
#if QT_VERSION < 0x050000
	QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));
	QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
#endif
#ifdef DEBUG_TIMING
	__debugTimer.start();
	__lastTime = __totalTime = 0;
#endif
	__dbg_classes.clear();
	__dbg_classes_info.clear();
	__dbg_classes_log.clear();
	__dbg_classes_logv.clear();
	QStringList debugees;
	__dbg_classes << "__na__";
	__dbg_classes_info << "__na__";
	__dbg_classes_log << "__na__";
	__dbg_classes_logv << "__na__";
	debugees = debug.split(",");
	foreach (const QString d, debugees) {
		if (d.isEmpty())
			continue;
		QString className;
		int level = defaultLevel;
		if (d.contains(":")) {
			QStringList pair = d.split(":");
			className = pair[0];
			level = pair[1].toInt();
		} else
			className = d;
		if (className == "*") {
			if (level > 0)
				__dbg_classes.clear();
			if (level > 1)
				__dbg_classes_info.clear();
			if (level > 2)
				__dbg_classes_log.clear();
			if (level > 3)
				__dbg_classes_logv.clear();
			break;
		}
		if (level > 0)
			__dbg_classes << className;
		if (level > 1)
			__dbg_classes_info << className;
		if (level > 2)
			__dbg_classes_log << className;
		if (level > 3)
			__dbg_classes_logv << className;
	}
	if (__dbg_classes.size() > 1)
		__dbg_classes.removeFirst();
	if (__dbg_classes_info.size() > 1)
		__dbg_classes_info.removeFirst();
	if (__dbg_classes_log.size() > 1)
		__dbg_classes_log.removeFirst();
	if (__dbg_classes_logv.size() > 1)
		__dbg_classes_logv.removeFirst();
	if (__dbg_classes.size())
		qDebug() << "debug classes:" << __dbg_classes << __dbg_classes_info << __dbg_classes_log << __dbg_classes_logv;
}

void initDebug()
{
	QString env = QString(getenv("DEBUG"));
	changeDebug(env);
}

/* Obtain a backtrace and print it to stdout. */
void print_trace(void)
{
	void *array[10];
	size_t size;
	char **strings;
	size_t i;

	size = backtrace (array, 10);
	strings = backtrace_symbols (array, size);

	printf ("Obtained %zd stack frames.\n", size);

	for (i = 0; i < size; i++)
		printf ("%s\n", strings[i]);

	free (strings);
}

#include <QFile>
#include <QMutex>
#include <QUdpSocket>

static QMutex mutex;
QString __dbg_file_log_dir = "/www/pages";
static QFile debugFile;
static void messageHandlerFile(QtMsgType type, const char *msg)
{
	switch (type) {
	case QtDebugMsg:
	case QtWarningMsg:
	case QtCriticalMsg:
	case QtFatalMsg:
		mutex.lock();
		debugFile.write(msg);
		debugFile.write("\n", 1);
		mutex.unlock();
		break;
	}
}

QString __dbg__network_addr;
int __dbg_debugging_mode = 0;
static QUdpSocket *debugUdpSocket = NULL;
static void messageHandlerNetwork(QtMsgType type, const char *msg)
{
	switch (type) {
	case QtDebugMsg:
	case QtWarningMsg:
	case QtCriticalMsg:
	case QtFatalMsg:
		mutex.lock();
		debugUdpSocket->write(msg);
		debugUdpSocket->write("\n");
		mutex.unlock();
		break;
	}
}

void setDebuggingMode(int mode, QString networkAddr)
{
	__dbg__network_addr = networkAddr;
	__dbg_debugging_mode = mode;
	/*if (mode == 3) {
		debugWebSockTarget = webserver;
		webserver->allocateLogBuffer(1024 * 1024);
	} else {
		webserver->releaseLogBuffer();
	}*/
	if (mode == 2) {
		if (!debugUdpSocket) {
			debugUdpSocket = new QUdpSocket;
			debugUdpSocket->bind(0);
		}
		debugUdpSocket->disconnectFromHost();
		if (networkAddr.contains(":"))
			debugUdpSocket->connectToHost(QHostAddress(networkAddr.split(":")[0]), networkAddr.split(":")[1].toInt());
		else
			debugUdpSocket->connectToHost(QHostAddress(networkAddr), 19999);
	} else if (mode == 1) {
		if (!debugFile.isOpen()) {
			debugFile.setFileName(QString("%1/debug_%2.log").arg(__dbg_file_log_dir)
								  .arg(QDateTime::currentDateTime().toString("ddMMyy_hhmmss")));
			debugFile.open(QIODevice::WriteOnly);
		}
	} else {
		/*if (debugUdpSocket) {
			delete debugUdpSocket;
			debugUdpSocket = NULL;
		}*/
		mutex.lock();
		if (debugFile.isOpen())
			debugFile.close();
		mutex.unlock();
	}
#if QT_VERSION < 0x050000
	if (mode == 0)
		qInstallMsgHandler(0);
	else if (mode == 1)
		qInstallMsgHandler(messageHandlerFile);
	else if (mode == 2)
		qInstallMsgHandler(messageHandlerNetwork);
	/*else if (mode == 3)
		qInstallMsgHandler(messageHandlerWebSock);*/
#endif
}

extern "C" {

intptr_t __thread _trstack[1024];
intptr_t __thread _trstack2[1024];
intptr_t __thread _trpos = 0;

void __attribute__ ((no_instrument_function)) __cyg_profile_func_enter  (void *this_fn, void *call_site);
void __attribute__ ((no_instrument_function)) __cyg_profile_func_exit  (void *this_fn, void *call_site);
void __attribute__ ((no_instrument_function)) init_instrument();
int __attribute__ ((no_instrument_function)) main(int argc, char *argv[]);

void __cyg_profile_func_enter(void *this_fn, void *call_site)
{
	_trstack[_trpos] = (intptr_t)this_fn;
	_trstack2[_trpos++] = (intptr_t)call_site;
	_trpos &= 0x3ff;
}

void __cyg_profile_func_exit(void *this_fn, void *call_site)
{
	_trpos--;
}

}

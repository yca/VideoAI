#include "mainwindow.h"
#include "snippets.h"

#include <QApplication>

#include <stdio.h>

static void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
	QByteArray localMsg = msg.toLocal8Bit();
	fprintf(stderr, "%s\n", localMsg.constData());
	return;
	switch (type) {
	case QtDebugMsg:
		fprintf(stderr, "Debug: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
		break;
	case QtWarningMsg:
		fprintf(stderr, "Warning: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
		break;
	case QtCriticalMsg:
		fprintf(stderr, "Critical: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
		break;
	case QtFatalMsg:
		fprintf(stderr, "Fatal: %s (%s:%u, %s)\n", localMsg.constData(), context.file, context.line, context.function);
		abort();
	}
}

int main(int argc, char *argv[])
{
	qInstallMessageHandler(myMessageOutput);

	//Snippets::voc2007();
	//Snippets::vocpyr2linearsvm();
	Snippets::toVOCKit("/home/caglar/myfs/source-codes/personal/build_x86/videoai/vocsvm/L2_5000_s8/");
	//Snippets::caltech1();
	//Snippets::pyr2linearsvm("data/svm_train.txt", "data/svm_test.txt");
	return 0;

	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	return a.exec();
}

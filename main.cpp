#include "mainwindow.h"
#include "snippets.h"

#include <QDebug>
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
#include "imps/caltechbench.h"
int main(int argc, char *argv[])
{
	qInstallMessageHandler(myMessageOutput);

	//CaltechBench::createImageFeatures();
	//CaltechBench::createDictionary(512, 1000);
	//return 0;

	int flags = 0x3; /* 1: pyramids, 2: rbow 3: both */
	int K = 512;
	QString dname = QString("data/dict_%1.bin").arg(K);

	//CaltechBench::createImageIds(dname);

	//CaltechBench::createImageDescriptors("data/dict_128.bin");
	CaltechBench::createImageDescriptors2(dname, 2, flags);

	//CaltechBench::exportForLibSvm();
	CaltechBench::exportForLibSvmMulti(flags);

	//Snippets::oxfordCreateSoft();
	//Snippets::oxfordMakeDensePyramids();
	//Snippets::oxfordSpatialRerank();
	//Snippets::oxfordRunQueriesPar();
	//Snippets::oxfordTemp();
	//Snippets::oxfordRunQueries();
	//Snippets::oxfordRunQuery();
	//Snippets::oxfordRerankAll();
	//qDebug() << Snippets::oxfordRerank(0);
	/*QList<int> qs;
	for (int i = 0; i < 55; i++)
		qs << i;
	Snippets::oxfordRerank(qs);*/
	return 0;

	int step = 0, L = 2, H = 0;
	//Snippets::vocpyr2linearsvm("data/pyramids_5000_L2_s2.dat", "L2_5000_s2_np", 1);

#ifdef minimized
	Snippets::voc2007Min(step, L, H);
	QString sub = QString("L%1_5000_s%2_h%3_min").arg(L).arg(step).arg(H);
	QString pyramidData = QString("data/pyramids_min_5000_L%1H%2_s%3.dat").arg(L).arg(H).arg(step);
	Snippets::vocTrain(pyramidData, sub, 1, 2.8);
	Snippets::vocPredict(pyramidData, sub, 1);
	Snippets::vocAP(sub);
#else
	Snippets::voc2007(step, L, H);
	QString sub = QString("L%1_5000_s%2_h%3").arg(L).arg(step).arg(H);
	QString pyramidData = QString("data/pyramids_5000_L%1H%2_s%3.dat").arg(L).arg(H).arg(step);
	Snippets::vocTrain(pyramidData, sub, 1, 2.8);
	Snippets::vocPredict(pyramidData, sub, 1);
	Snippets::vocAP(sub);
#endif
	//Snippets::toVOCKit(QString("/home/caglar/myfs/source-codes/personal/build_x86/videoai/vocsvm/%1/").arg(sub));

	//Snippets::caltech1();
	//Snippets::pyr2linearsvm("data/svm_train.txt", "data/svm_test.txt");

	return 0;

	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	return a.exec();
}

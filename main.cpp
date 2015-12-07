#include "mainwindow.h"
#include "common.h"
#include "snippets.h"
#include "imps/caltechbench.h"
#include "lmm/classificationpipeline.h"

#include <QDir>
#include <QDebug>
#include <QApplication>

#include <stdio.h>

static const QMap<QString, QString> parseArgs(int &argc, char **argv)
{
	QMap<QString, QString> args;
	args.insert("app", QString::fromLatin1(argv[0]));
	QStringList files;
	QString cmd;
	for (int i = 1; i < argc; i++) {
		QString arg = QString::fromLatin1(argv[i]);
		cmd.append(arg).append(" ");
		if (!arg.startsWith("--") && !arg.startsWith("-")) {
			files << arg;
			continue;
		}
		QString pars;
		if (i + 1 < argc)
			pars = QString::fromLatin1(argv[i + 1]);
		if (!pars.startsWith("--") && !pars.startsWith("-")) {
			i++;
			args.insert(arg, pars);
			cmd.append(pars).append(" ");
		} else
			args.insert(arg, "");
	}
	args.insert("__app__", QString::fromLatin1(argv[0]));
	args.insert("__cmd__", cmd);
	args.insert("__files__", files.join("\n"));
	return args;
}

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
#include "common.h"
#define parDbg(_x) qDebug() << #_x << pars._x
#define setParInt(_x) if (flds[0] == #_x) { pars._x = flds[1].toInt(); parDbg(_x); }
#define setParInt64(_x) if (flds[0] == #_x) { pars._x = flds[1].toLongLong(); parDbg(_x); }
#define setParDbl(_x) if (flds[0] == #_x) { pars._x = flds[1].toDouble(); parDbg(_x); }
#define setParStr(_x) if (flds[0] == #_x) { pars._x = flds[1]; parDbg(_x); }
#define setParEnum(_x, _y) if (flds[0] == #_x) { pars._x = (_y)flds[1].toInt(); parDbg(_x); }

#define setAllPars() \
	setParEnum(ft, ClassificationPipeline::ftype); \
	setParInt(xStep); \
	setParInt(yStep); \
	setParInt(exportData); \
	setParInt(threads); \
	setParInt(K); \
	setParInt(dictSubSample); \
	setParInt(useExisting); \
	setParInt(createDict); \
	setParStr(dataPath); \
	setParDbl(gamma); \
	setParInt(trainCount); \
	setParInt(testCount); \
	setParInt(L); \
	setParInt64(maxMemBytes); \
	setParInt(maxFeaturesPerImage); \
	setParInt(useExistingTrainSet); \
	setParStr(datasetPath); \
	setParStr(datasetName); \

static int pipelineImp(const QMap<QString, QString> &args, int argc, char *argv[])
{
	QApplication a(argc, argv);
	QDir::setCurrent(a.applicationDirPath());
	ClassificationPipeline *pl;
	if (args.contains("--conf")) {
		assert(QFile::exists(args["--conf"]));
		QStringList lines = Common::importText(args["--conf"]);
		ClassificationPipeline::parameters pars;
		foreach (QString line, lines) {
			if (!line.contains("="))
				continue;
			line.trimmed();
			line.remove(";");
			line.remove("\"");
			line.remove("pars.");
			QStringList flds = line.split("=");
			flds[0] = flds[0].trimmed();
			flds[1] = flds[1].trimmed();
			if (flds[1] == "true")
				flds[1] = "1";
			if (flds[1] == "false")
				flds[1] = "0";
			if (flds[1] == "FEAT_SIFT")
				flds[1] = "0";
			if (flds[1] == "FEAT_SURF")
				flds[1] = "1";
			setAllPars();
		}
		/* check command line parameters */
		QMapIterator<QString, QString> mi(args);
		while (mi.hasNext()) {
			mi.next();
			QStringList flds;
			QString key = mi.key();
			key.remove("--");
			flds << key;
			flds << mi.value().trimmed();
			setAllPars();
		}
		pl = new ClassificationPipeline(pars);
	} else
		pl = new ClassificationPipeline;
	pl->start();
	return a.exec();
}

static void accTemp()
{
	QString tmp = "";
	QDir d(QString("/home/caglar/myfs/source-codes/personal/build_x86/videoai/dataset2/%1/").arg(tmp));
	QStringList results = d.entryList(QStringList() << "*.res", QDir::Files);
	QHash<QString, float> best;
	QHash<QString, QStringList> bestPars;
	for (int i = 0; i < results.size(); i++) {
		QString rfile = d.filePath(results[i]);
		QString pfile = d.filePath(results[i]).replace(".res", ".txt").replace("_train_", "_test_");
		QStringList flds = results[i].remove(".res").split("_");
		flds.removeFirst();
		flds.removeFirst();
		//qDebug() << flds[2] << flds[3] << flds[4] << flds[5] << flds[6];
		QHash<int, float> perClassAcc;
		float acc = Snippets::getAcc(rfile, pfile, perClassAcc);
		if (acc > best[flds[0]]) {
			best[flds[0]] = acc;
			bestPars[flds[0]] = flds;
		}
		qDebug() << perClassAcc;
	}
	qDebug() << best << bestPars;
}

int main(int argc, char *argv[])
{
	qInstallMessageHandler(myMessageOutput);

	QMap<QString, QString> args = parseArgs(argc, argv);
	if (args["__app__"].contains("pipeline"))
		return pipelineImp(args, argc, argv);

	accTemp();
	/*Snippets::getAcc("/home/caglar/myfs/source-codes/personal/build_x86/videoai/dataset2/svm_train_ftype0_K2048_step3_L2_gamma0.01.res",
					"/home/caglar/myfs/source-codes/personal/build_x86/videoai/dataset2/svm_test_ftype1_K2048_step3_L2_gamma0.01.txt");
					//"/home/caglar/myfs/source-codes/personal/build_x86/videoai/dataset2/cats");*/
	return 0;

	//CaltechBench::createImageFeatures();
	//CaltechBench::createDictionary(512, 1000);

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

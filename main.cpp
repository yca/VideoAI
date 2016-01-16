#include "mainwindow.h"
#include "common.h"
#include "snippets.h"
#include "imps/caltechbench.h"

#include "lmm/bowpipeline.h"
#include "lmm/cnnpipeline.h"
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

#if QT_VERSION >= 0x050000
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
#endif

#ifdef HAVE_LMM
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
	setParEnum(cl, ClassificationPipeline::cltype); \
	setParInt(imFlags); \
	setParStr(fileListTxt); \
	setParStr(trainListTxt); \
	setParStr(testListTxt); \
	setParStr(lmdbFeaturePath); \
	setParStr(cnnFeatureLayer); \
	setParInt(debug); \
	setParInt(spatialSize); \
	setParInt(homkermap); \
	setParStr(caffeBaseDir); \
	setParStr(caffeDeployProto); \
	setParStr(caffeModelFile); \
	setParStr(caffeImageMeanProto); \
	setParInt(targetCaffeModel); \
	setParInt(featureMergingMethod); \
	setParInt(dataAug); \
	setParInt(rotationDegree); \
	setParStr(cnnFeatureLayerType); \
	setParInt(runId); \

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
			if (line.startsWith("#"))
				continue;
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
			if (flds[1] == "FEAT_CNN")
				flds[1] = "2";
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
		if (pars.createDict || pars.cl == ClassificationPipeline::CLASSIFY_BOW)
			pl = new BowPipeline(pars);
		else
			pl = new CnnPipeline(pars);
	} else
		pl = new BowPipeline;
	pl->init();
	pl->start();
	return a.exec();
}
#else
static int pipelineImp(const QMap<QString, QString> &args, int argc, char *argv[])
{
	return 0;
}

#endif

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
#include "datasetmanager.h"
static int makeForDigits()
{
	QStringList lines = Common::importText("/home/amenmd/myfs/source-codes/personal/build/videoai/dataset2/train_set.txt");
	DatasetManager ds;
	ds.addDataset("caltech", "/home/amenmd/myfs/tasks/video_analysis/dataset/101_ObjectCategories/");
	QStringList images = ds.dataSetImages("caltech");
	QDir d("/home/amenmd/myfs/tasks/video_analysis/dataset");
	d.mkpath("for_digits/caltech_train");
	d.mkpath("for_digits/caltech_test");
	for (int i = 0; i < lines.size(); i++) {
		const QString line = lines[i];
		if (line.trimmed().isEmpty())
			continue;
		QString image = images[i];
		QString cat = image.split("/")[8];
		d.mkpath(QString("for_digits/caltech_train/%1").arg(cat));
		d.mkpath(QString("for_digits/caltech_test/%1").arg(cat));
		QStringList flds = line.split(":");
		if (flds[1].toInt())
			QFile::copy(image, d.filePath(QString("for_digits/caltech_train/%1/%2").arg(cat).arg(QFileInfo(image).fileName())));
		else if (flds[2].toInt())
			QFile::copy(image, d.filePath(QString("for_digits/caltech_test/%1/%2").arg(cat).arg(QFileInfo(image).fileName())));
	}
	return 0;
}

#include "caffe/caffecnn.h"
#include <QElapsedTimer>
static int cnnClassify()
{
	DatasetManager ds;
	ds.addDataset("caltech", "/home/amenmd/myfs/tasks/video_analysis/dataset/101_ObjectCategories/");
	QStringList images = ds.dataSetImages("caltech");

	QString cbase = "/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/";
	CaffeCnn c;
	c.load(cbase + "models/bvlc_reference_caffenet/deploy.prototxt",
		   cbase + "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel",
		   cbase + "data/ilsvrc12/imagenet_mean.binaryproto",
		   cbase + "data/ilsvrc12/synset_words.txt");

	QElapsedTimer t; t.start();
	foreach (const QString &image, images) {
		qDebug() << image << t.restart();
		QStringList cats = c.classify(image, 100);
		Common::exportText(cats.join("\n"), QString(image).replace(".jpg", ".cnn"));
	}
	return 0;
}
static int cnnExtract()
{
	DatasetManager ds;
	ds.addDataset("caltech", "/home/amenmd/myfs/tasks/video_analysis/dataset/101_ObjectCategories/");
	QStringList images = ds.dataSetImages("caltech");

	QString cbase = "/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/";
	CaffeCnn c;
	c.load(cbase + "models/bvlc_reference_caffenet/deploy.prototxt",
		   cbase + "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel",
		   cbase + "data/ilsvrc12/imagenet_mean.binaryproto",
		   cbase + "data/ilsvrc12/synset_words.txt");

	//c.extractFeatures("/home/amenmd/myfs/tasks/video_analysis/dataset/101_ObjectCategories/schooner/image_0001.jpg", "fc7");
	//c.extractFeatures("/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/examples/_temp/features/", "fc7");
	return 0;
}

#include "vlfeat/vlfeat.h"
static int calcPearson()
{
	DatasetManager ds;
	ds.addDataset("caltech", "/home/amenmd/myfs/tasks/video_analysis/dataset/101_ObjectCategories/");
	QStringList images = ds.dataSetImages("caltech");
	QStringList cats = ds.datasetCategories("caltech");
	QStringList lines = Common::importText("/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/data/ilsvrc12/synset_words.txt");
	QStringList icats;
	foreach (const QString &line, lines) {
		QStringList flds = line.split(" ");
		if (flds.size() < 2)
			continue;
		icats << flds.first().trimmed();
	}

	/* now create train/test set */
	QString trainSetFileName = QString("%1/train_set.txt")
			.arg("/home/amenmd/myfs/source-codes/personal/build/videoai/dataset2");
	lines = Common::importText(trainSetFileName);
	QHash<QString, int> useForTrain;
	QHash<QString, int> useForTest;
	for (int i = 0; i < lines.size(); i++) {
		const QString &line = lines[i];
		QStringList vals = line.split(":");
		if (vals.size() != 3)
			continue;
		useForTrain.insert(images[i], vals[1].toInt());
		useForTest.insert(images[i], vals[2].toInt());
	}
	assert(useForTrain.size() == images.size());

	Mat trainData(0, icats.size(), CV_32F);
	Mat testData(0, icats.size(), CV_32F);
	Mat trainLabels(0, 1, CV_32F);
	Mat testLabels(0, 1, CV_32F);
	foreach (const QString &image, images) {
		if (!useForTrain[image] && !useForTest[image])
			continue;
		Mat m = Mat::zeros(1, trainData.cols, trainData.type());
		QStringList lines = Common::importText(QString(image).replace(".jpg", ".cnn"));
		foreach (const QString &line, lines) {
			QStringList flds = line.split(" ");
			if (flds.size() < 3)
				continue;
			float val = flds.last().toFloat();
			int col = icats.indexOf(flds.first().trimmed());
			assert(col >= 0);
			m.at<float>(0, col) = val;
		}
		int cat1 = cats.indexOf(ds.getCategory(image));
		qDebug() << image << useForTrain[image] << useForTest[image] << cat1;
		Mat l = Mat::ones(1, 1, CV_32F) * cat1;
		if (useForTrain[image]) {
			trainData.push_back(m);
			trainLabels.push_back(l);
		} else {
			testData.push_back(m);
			testLabels.push_back(l);
		}
	}

	VlFeat::exportToSvm(trainData, trainLabels, "dataset3/train.txt", 0.1);
	VlFeat::exportToSvm(testData, testLabels, "dataset3/test.txt", 0.1);

	return 0;
}

static int calculateAcc(const QMap<QString, QString> &args)
{
	QHash<int, float> perClassAcc;
	QString results = args["--results"];
	QString test = args["--test-data"];
	ffDebug() << Snippets::getAcc(results, test, perClassAcc);
	//Snippets::toVOCKit(results);
	//Snippets::getAP(results, test);
	return 0;
}

#include "svm/liblinear.h"
static int mergeSvmFiles(const QMap<QString, QString> &args)
{
	if (!args["--input2"].contains(","))
		return LibLinear::merge(args["--input1"], args["--input2"], args["--output"], args["--fcnt"].toInt());
	QString base = args["--input1"];
	QStringList filenames = args["--input2"].split(",");
	QStringList outputs = args["--output"].split(",");
	QStringList counts = args["--fcnt"].split(",");
	int off = counts[0].toInt();
	int K = filenames[0].toInt();
	for (int i = 1; i < filenames.size(); i++) {
		ffDebug() << base.arg(K) << base.arg(filenames[i]) << base.arg(outputs[i - 1]) << off;
		//if (i == 5)
		LibLinear::merge(base.arg(K), base.arg(filenames[i]), base.arg(outputs[i - 1]), off);
		off += counts[i].toInt();
		K = outputs[i - 1].toInt();
	}
	return 0;
}

static int lateFusionSvm(const QMap<QString, QString> &args)
{
	QString base = args["--input1"];
	QStringList filenames = args["--input2"].split(",");
	QString output = args["--output"];
	QList<Mat> counts;
	for (int i = 0; i < filenames.size(); i++) {
		ffDebug() << base.arg(filenames[i]);

		//QList<int> truth = getPredictInputs(base.arg(filenames[i]));
		QList<int> results;
		QStringList lines = Common::importText(base.arg(filenames[i]).replace(".txt", ".results").replace("svm_test", "svm_train"));
		foreach (QString line, lines)
			if (!line.trimmed().isEmpty())
				results << line.toInt();
		assert(results.size());

		if (counts.size() == 0)
			while (counts.size() != results.size())
				counts << Mat::zeros(1, 10240, CV_32F);
		assert(counts.size() == results.size());
		for (int j = 0; j < results.size(); j++)
			//counts[j][results[j]]++;
			counts[j].at<float>(0, results[j])++;
	}
	QStringList lines;
	for (int i = 0; i < counts.size(); i++) {
		double min, max;
		Point mini, maxi;
		minMaxLoc(counts[i], &min, &max, &mini, &maxi);
		lines << QString::number(maxi.x);
	}
	lines << "";
	Common::exportText(lines.join("\n"), output);
	return 0;
}

static QList<int> getPredictInputs(const QString &filename)
{
	QList<int> list;
	QFile f(filename);
	f.open(QIODevice::ReadOnly);
	while (!f.atEnd()) {
		QByteArray l = f.readLine();
		list << l.mid(0, l.indexOf(" ")).toInt();
	}
	return list;
}

static int calculateAccVideo(const QMap<QString, QString> &args)
{
	/* import results first */
	QList<int> truth = getPredictInputs(args["--test-data"]);
	QList<int> results;
	QStringList lines = Common::importText(args["--results"]);
	foreach (QString line, lines)
		if (!line.trimmed().isEmpty())
			results << line.toInt();

	/* now load test mapping */
	DatasetManager ds;
	ds.addUCF101(args["--dataset-path"], "/home/amenmd/myfs/tasks/video_analysis/dataset/ucf/ucfTrainTestlist");
	QStringList images = ds.dataSetImages("ucf101");
	QHash<QString, int> tthash;
	QStringList testList = Common::importText(args["--test-list"]);
	testList.removeLast();
	for (int j = 0; j < testList.size(); j++) {
		QStringList flds = testList[j].trimmed().split(" ");
		QString name = flds[0].remove(".avi");
		if (name.isEmpty())
			continue;
		tthash.insert(name.split("/").last(), 2);
	}
	QStringList testImages;
	QList<int> testIndexes;
	for (int i = 0; i < images.size(); i++) {
		QFileInfo fi(images[i]);
		QStringList flds = fi.baseName().split("_");
		flds.removeLast();
		if (tthash.contains(flds.join("_"))) {
			testImages << images[i];
			testIndexes << i;
		}
	}

	QHash<QString, QHash<int, int> > counts;
	QHash<QString, int> videoTruths;
	for (int i = 0; i < testImages.size(); i++) {
		QFileInfo fi(testImages[i]);
		QString bname = fi.baseName();
		QString videoName = bname.left(bname.size() - 6);
		assert(!videoName.isEmpty());
		assert(videoName.startsWith("v_"));
		assert(results[i]);
		counts[videoName][results[i]]++;
		int realCat = ds.getDatasetCategory(bname.split("_")[1]);
		truth[i] = realCat;
		if (!videoTruths.contains(videoName))
			videoTruths[videoName] = truth[i];
		assert(videoTruths[videoName] == truth[i]);
	}
	assert(counts.size() == testList.size());
	QHashIterator <QString, QHash<int, int> > hi(counts);
	QHash<QString, int> videoPredicts;
	while (hi.hasNext()) {
		hi.next();
		const QHash<int, int> hash = hi.value();
		QHashIterator<int, int> hj(hash);
		int max = 0;
		int maxLabel = 0;
		while (hj.hasNext()) {
			hj.next();
			if (hj.value() > max) {
				max = hj.value();
				maxLabel = hj.key();
			}
		}
		assert(max);
		assert(maxLabel);
		videoPredicts[hi.key()] = maxLabel;
	}
	QHashIterator<QString, int> hj(videoPredicts);
	int correct = 0;
	while (hj.hasNext()) {
		hj.next();
		if (hj.value() == videoTruths[hj.key()])
			correct++;
	}
	qDebug() << (float)correct / videoPredicts.size();

	return 0;
#if 0

#if 0
	DatasetManager ds;
	ds.addDataset("tmp", args["--dataset-path"]);
	QStringList images = ds.dataSetImages("tmp");
#endif

	//QHash<QString, int> tthash;
	QStringList cats;
	QStringList testList = Common::importText(args["--test-list"]);
	for (int j = 0; j < testList.size(); j++) {
		if (testList[j].trimmed().isEmpty())
			continue;
		QStringList flds = testList[j].trimmed().split(" ");
		QString name = flds[0].remove(".avi");
		if (name.isEmpty())
			continue;
		//tthash.insert(name.split("/").last(), 2);

		ffDebug() << name;
	}

#if 0
	QList<int> labels;
	QStringList testImages;
	for (int i = 0; i < images.size(); i++) {
		QFileInfo fi(images[i]);
		QStringList flds = fi.baseName().split("_");
		flds.removeLast();
		int val = tthash[flds.join("_")];
		if (val == 2) {
			labels << cats.indexOf(fi.dir().dirName()) + 1;
			testImages << fi.baseName();
		}
	}

	QHash<int, float> perClassAcc;
	QString resultsFile = args["--results"];
	QString predictInputs = args["--test-data"];

	//ffDebug() << Snippets::getAcc(results, test, perClassAcc);
	QList<int> results;
	QStringList lines = Common::importText(resultsFile);
	foreach (QString line, lines)
		results << line.toInt();
	QList<int> truth = getPredictInputs(predictInputs);
	QHash<int, int> ctotal, ccorrect;
	int total = 0, correct = 0;
	for (int i = 0; i < results.size(); i++) {
		if (i >= truth.size())
			break;
		if (results[i] == truth[i])
			correct++;
		total++;

		int cl = truth[i];
		ctotal[cl]++;
		if (results[i] == truth[i])
			ccorrect[cl]++;
	}
	QHashIterator<int, int> hi(ctotal);
	double accpc = 0;
	while (hi.hasNext()) {
		hi.next();
		double acc = (double)ccorrect[hi.key()] / hi.value() * 100;
		perClassAcc.insert(hi.key(), acc);
		accpc +=  acc / ctotal.size();
	}

	QString last;
	QHash<int, int> counts;
	int clipLabel = 0;
	int clipTotal = 0, clipCorrect = 0;
	for (int i = 0; i < labels.size(); i++) {
		QStringList flds = testImages[i].split("_");
		flds.removeLast();
		QString clipName = flds.join("_");
		counts[results[i]]++;
		if (last.isEmpty())
			clipLabel = truth[i];
		else if (last != clipName) {
			/* find most voted */
			QHashIterator<int, int> hi(counts);
			int max = 0;
			int maxLabel = 0;
			while (hi.hasNext()) {
				hi.next();
				if (hi.value() > max) {
					max = hi.value();
					maxLabel = hi.key();
				}
			}

			assert(clipLabel);
			assert(maxLabel);
			/* accumulate */
			if (clipLabel == maxLabel)
				clipCorrect++;
			clipTotal++;

			/* reset */
			clipLabel = truth[i];
			counts.clear();
		} else {
			qDebug() << clipLabel << truth[i] << testImages[i] << i;
			qDebug() << labels.size() << results.size() << truth.size() << results.first() << results.last();
			assert(clipLabel == truth[i]);
		}
		last = clipName;
	}
	ffDebug() << labels.size() << results.size() << truth.size() << accpc << (double)clipCorrect / clipTotal;
#endif
	return 0;
#endif
}

static int diff2Svms()
{
	QFile f1("/home/amenmd/myfs/source-codes/personal/build/videoai/work_bowcnn_ucf/svm_train_ftype2_K4_step3_L2_gamma0.txt");
	QFile f2("/home/amenmd/myfs/source-codes/personal/build/videoai/dataset_ucf_fc7/svm_train_ftype2_K512_step3_L2_gamma0.txt");
	//QFile f1("/home/amenmd/myfs/source-codes/personal/build/videoai/work_bowcnn_ucf/svm_test_ftype2_K4_step3_L2_gamma0.txt");
	//QFile f2("/home/amenmd/myfs/source-codes/personal/build/videoai/dataset_ucf_fc7/svm_test_ftype2_K512_step3_L2_gamma0.txt");
	qDebug() << f1.open(QIODevice::ReadOnly);
	qDebug() << f2.open(QIODevice::ReadOnly);
	int line = 0;
	while (!f1.atEnd()) {
		line++;
		//qDebug() << "line" << line;
		const QString line1 = f1.readLine().trimmed();
		const QString line2 = f2.readLine().trimmed();
		if (line1.isEmpty()) {
			qDebug() << line << "is empty";
			continue;
		}
		const QStringList args1 = line1.split(" ");
		const QStringList args2 = line2.split(" ");
		if (args1.size() != args2.size()) {
			qDebug() << "size mismatch" << args1.size() << args2.size();
			continue;
		}
		for (int i = 0; i < args1.size(); i++) {
			QStringList vals1 = args1[i].split(":");
			QStringList vals2 = args2[i].split(":");
			if (vals1.size() != vals2.size()) {
				qDebug() << "feature size mismatch" << vals1.size() << vals2.size() << i;
				continue;
			}
			if (vals1.size() == 1) {
				float val1 = vals1[0].toFloat();
				float val2 = vals2[0].toFloat();
				if (val1 != val2)
					qDebug() << "class mismatch" << i << val1 << val2 << line;
				qDebug() << line << val1 << val2;
			} else if (vals1.size() == 2) {
				float val11 = vals1[0].toFloat();
				float val12 = vals1[1].toFloat();
				float val21 = vals2[0].toFloat();
				float val22 = vals2[1].toFloat();
				if (val11 != val21 || val12 != val22)
					qDebug() << "feature mismatch" << val11 << val21 << val12 << val22 << line;
			} else
				assert(0);
		}
	}

	return 0;
}

int main(int argc, char *argv[])
{
#if QT_VERION >= 0x050000
	qInstallMessageHandler(myMessageOutput);
#endif

	QMap<QString, QString> args = parseArgs(argc, argv);
	if (args["__app__"].contains("pipeline"))
		return pipelineImp(args, argc, argv);
	else if (args["__app__"].contains("accuracy") && args.contains("--video"))
		return calculateAccVideo(args);
	else if (args["__app__"].contains("accuracy"))
		return calculateAcc(args);
	else if (args["__app__"].contains("mergesvm"))
		return mergeSvmFiles(args);
	else if (args["__app__"].contains("latefusion"))
		return lateFusionSvm(args);

	CaffeCnn::printLayerInfo("/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/examples/net_surgery/VGG_ILSVRC_16_layers_deploy_s1.prototxt");
	return 0;
	return diff2Svms();

	args.insert("--conf", "/home/amenmd/myfs/tasks/video_analysis/videoai_work/pipeline.conf.ucf101.tmp");
	return pipelineImp(args, argc, argv);

	//return cnnExtract();

	QHash<int, float> perClassAcc;
	QString base = "/home/amenmd/myfs/source-codes/personal/build/videoai/dataset_ucf101_1/";
	qDebug() << Snippets::getAcc(base + "svm_train_ftype1_K2048_step3_L0_gamma0.5.results",
								 base + "svm_test_ftype1_K2048_step3_L0_gamma0.5.txt", perClassAcc);
	return 0;

	/*QHash<int, float> perClassAcc;
	qDebug() << Snippets::getAcc("/home/amenmd/myfs/source-codes/personal/build/videoai/dataset3/svm_train_ftype1_K2048_step3_L1_gamma0.5.results",
								 "/home/amenmd/myfs/source-codes/personal/build/videoai/dataset3/svm_test_ftype1_K2048_step3_L1_gamma0.5.txt", perClassAcc);
	return 0;*/

	return cnnClassify();
	return calcPearson();


	return 0;

	return makeForDigits();
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

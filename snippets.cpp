#include "snippets.h"
#include "datasetmanager.h"
#include "debug.h"
#include "common.h"

#include "opencv/opencv.h"

#include "vision/pyramids.h"

#include "vlfeat/vlfeat.h"

#include <QDir>
#include <QFile>

#include <iostream>

Snippets::Snippets(QObject *parent) :
	QObject(parent)
{
}

void Snippets::caltech1()
{
	DatasetManager dm;
	Pyramids pyr;
	pyr.setDict("/home/caglar/myfs/source-codes/personal/build_x86/videoai/data/cal1000.dict");
	dm.addDataset("caltech256", "/home/caglar/myfs/tasks/video_analysis/data/256_ObjectCategories/");
	Mat pyramids = pyr.calculatePyramids(dm.dataSetImages("caltech256"), 2, 8);
	OpenCV::exportMatrix("data/pyramids_1000_L2.dat", pyramids);
}

void Snippets::voc2007()
{
	DatasetManager dm;
	Pyramids pyr;
	pyr.setDict("/home/caglar/myfs/source-codes/personal/build_x86/videoai/data/voc5000.dict");
	dm.addDataset("voc", "/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages/");
	Mat pyramids = pyr.calculatePyramids(dm.dataSetImages("voc"), 2, 2);
	OpenCV::exportMatrix("data/pyramids_5000_L2_s2.dat", pyramids);
}

void Snippets::vocpyr2linearsvm()
{
	srand(time(NULL));
	DatasetManager *dm = new DatasetManager;
	dm->addDataset("voc", "/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages");
	QStringList images = dm->dataSetImages("voc");
	Mat pyramids = OpenCV::importMatrix("data/pyramids_5000_L2_s2.dat");
	QDir out("vocsvm");

	QStringList cats = QStringList()
		<< "aeroplane"
		<< "bicycle"
		<< "boat"
		<< "bird"
		<< "bottle"
		<< "bus"
		<< "car"
		<< "cat"
		<< "chair"
		<< "cow"
		<< "diningtable"
		<< "dog"
		<< "horse"
		<< "motorbike"
		<< "person"
		<< "pottedplant"
		<< "sheep"
		<< "sofa"
		<< "train"
		<< "tvmonitor";

	foreach (QString cat, cats) {
		fDebug("Processing %s", qPrintable(cat));
		out.mkdir(cat);
		/* training data */
		QList<QPair<int, QString> > list = dm->voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "trainval", cat);
		Mat pos(0, pyramids.cols, CV_32F);
		Mat neg(0, pyramids.cols, CV_32F);
		for (int i = 0; i < list.size(); i++) {
			QPair<int, QString> p = list[i];
			QString str = p.second.replace("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "/home/caglar/myfs/tasks/video_analysis/data/vocimages");
			int row = images.indexOf(str);
			assert(row >= 0);
			if (p.first > 0)
				pos.push_back(pyramids.row(row));
			else
				neg.push_back(pyramids.row(row));
		}
		neg = OpenCV::subSampleRandom(neg, pos.rows);
		Mat labelsPos = Mat::ones(pos.rows, 1, CV_32F);
		Mat labelsNeg = Mat::ones(neg.rows, 1, CV_32F) * -1;
		vconcat(labelsPos, labelsNeg, labelsPos);
		vconcat(pos, neg, pos);
		VlFeat::exportToSvm(pos, labelsPos, QString("vocsvm/%1/svm_train.txt").arg(cat));
		/* testing data */
		list = dm->voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "test", cat);
		Mat all(0, pyramids.cols, CV_32F);
		Mat labels = Mat::ones(list.size(), 1, CV_32F);
		for (int i = 0; i < list.size(); i++) {
			QPair<int, QString> p = list[i];
			QString str = p.second.replace("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "/home/caglar/myfs/tasks/video_analysis/data/vocimages");
			int row = images.indexOf(str);
			assert(row >= 0);
			all.push_back(pyramids.row(row));
			if (p.first > 0)
				labels.at<float>(i) = 1;
			else
				labels.at<float>(i) = -1;
		}
		VlFeat::exportToSvm(all, labels, QString("vocsvm/%1/svm_test.txt").arg(cat));
	}
}

void Snippets::pyr2linearsvm(const QString &trainName, const QString &testName)
{
	DatasetManager *dm = new DatasetManager;
	VlHomogeneousKernelMap *map = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, 0.5, 1, -1, VlHomogeneousKernelMapWindowRectangular);
	dm->addDataset("caltech256", "/home/caglar/myfs/tasks/video_analysis/data/256_ObjectCategories/");
	QStringList images = dm->dataSetImages("caltech256");
	QStringList cats;
	Mat labels(images.size(), 1, CV_32F);
	Mat classPos(images.size(), 1, CV_32F);
	QHash<int, int> sampleCount;
	int cp = 0;
	for (int i = 0; i < images.size(); i++) {
		QStringList dirs = images[i].split("/", QString::SkipEmptyParts);
		QString cat = dirs[dirs.size() - 2];
		if (!cats.contains(cat)) {
			cp = 0;
			cats << cat;
		}
		int l = cats.indexOf(cat) + 1;
		labels.at<float>(i) = l;
		sampleCount[l]++;
		classPos.at<float>(i) = cp++;
	}
	int trcnt = 30;
	int tscnt = 50;
	int total = trcnt + tscnt;
	vector<Mat> trainSet, testSet;
	/* split into train/test */
	for (int i = 0; i < cats.size(); i++) {
		int cnt = sampleCount[i + 1];
		Mat idx = OpenCV::createRandomized(0, cnt);
		trainSet.push_back(idx.rowRange(0, 30));
		testSet.push_back(idx.rowRange(30, idx.rows > total ? total : idx.rows));
	}
	Mat pyramids = OpenCV::importMatrix("data/pyramids_1000_L2.dat");
	assert(pyramids.rows == labels.rows);
	QFile f(trainName);
	QFile f2(testName);
	f.open(QIODevice::WriteOnly);
	f2.open(QIODevice::WriteOnly);
	for (int i = 0; i < pyramids.rows; i++) {
		int label = labels.at<float>(i);
		const Mat &mt = trainSet[label - 1];
		const Mat &me = testSet[label - 1];
		int cp = classPos.at<float>(i);
		if (OpenCV::matContains(mt, cp))
			f.write(VlFeat::toSvmLine(map, pyramids.row(i), labels.at<float>(i)).toUtf8());
		else if (OpenCV::matContains(me, cp))
			f2.write(VlFeat::toSvmLine(map, pyramids.row(i), labels.at<float>(i)).toUtf8());
		ffDebug() << i << pyramids.rows;
	}
	f.close();
	f2.close();
}

void Snippets::pyr2svm(DatasetManager *dm, const QString &trainName, const QString &testName)
{
	dm->addDataset("caltech101", "/home/amenmd/myfs/tasks/hilal_tez/dataset/101_ObjectCategories/");
	QStringList images = dm->dataSetImages("caltech101");
	QStringList cats;
	Mat labels(images.size(), 1, CV_32F);
	Mat classPos(images.size(), 1, CV_32F);
	QHash<int, int> sampleCount;
	int cp = 0;
	for (int i = 0; i < images.size(); i++) {
		QStringList dirs = images[i].split("/", QString::SkipEmptyParts);
		QString cat = dirs[dirs.size() - 2];
		if (!cats.contains(cat)) {
			cp = 0;
			cats << cat;
		}
		int l = cats.indexOf(cat) + 1;
		labels.at<float>(i) = l;
		sampleCount[l]++;
		classPos.at<float>(i) = cp++;
	}
	int trcnt = 30;
	int tscnt = 50;
	int total = trcnt + tscnt;
	vector<Mat> trainSet, testSet;
	/* split into train/test */
	for (int i = 0; i < cats.size(); i++) {
		int cnt = sampleCount[i + 1];
		Mat idx = OpenCV::createRandomized(0, cnt);
		trainSet.push_back(idx.rowRange(0, 30));
		testSet.push_back(idx.rowRange(30, idx.rows > total ? total : idx.rows));
	}
	Mat pyramids = OpenCV::importMatrix("data/pyramids_500_L2.dat");
	assert(pyramids.rows == labels.rows);
	QFile f(trainName);
	QFile f2(testName);
	f.open(QIODevice::WriteOnly);
	f2.open(QIODevice::WriteOnly);
	for (int i = 0; i < pyramids.rows; i++) {
		int label = labels.at<float>(i);
		const Mat &mt = trainSet[label - 1];
		const Mat &me = testSet[label - 1];
		int cp = classPos.at<float>(i);
		if (OpenCV::matContains(mt, cp))
			f.write(OpenCV::toSvmLine(pyramids.row(i), labels.at<float>(i)).toUtf8());
		else if (OpenCV::matContains(me, cp))
			f2.write(OpenCV::toSvmLine(pyramids.row(i), labels.at<float>(i)).toUtf8());
		ffDebug() << i << pyramids.rows;
	}
	f.close();
	f2.close();
}

static bool lessThan(const QPair<int, float> p1, const QPair<int, float> p2)
{
	if (p1.second < p2.second)
		return false;
	return true;
}

static float getAveragePrecision(int cl, const QList<int> &results,
						  const QList<QList<float> > &probs,
						  const QList<int> &truth)
{
	/* pascal style ap calculation */
	QList<QPair<int, float> > out;
	for (int i = 0; i < results.size(); i++)
		out << QPair<int, float>(i, probs[i][cl - 1]);
	qSort(out.begin(), out.end(), lessThan);
	QList<int> tp;
	QList<int> fp;
	int sum1 = 0;
	for (int i = 0; i < out.size(); i++) {
		if (truth[out[i].first] == cl) {
			tp << 1;
			fp << 0;
			sum1++;
		} else {
			tp << 0;
			fp << 1;
		}
	}
	for (int i = 1; i < tp.size(); i++) {
		tp[i] += tp[i - 1];
		fp[i] += fp[i - 1];
	}
	QList<double> prec, rec;
	for (int i = 0; i < tp.size(); i++) {
		rec <<  (double)tp[i] / sum1;
		prec <<  (double)tp[i] / (tp[i] + fp[i]);
	}

	double ap = 0;
	for (double t = 0; t <= 1.0; t += 0.1) {
		float maxPrec = 0;
		for (int i = 0; i < rec.size(); i++) {
			if (rec[i] < t)
				continue;
			if (prec[i] > maxPrec)
				maxPrec = prec[i];
		}
		assert(maxPrec > 0);
		ap += maxPrec;
	}
	return ap / 11;
}

static QList<int> readSvmResults(QString filename, QList<QList<float> > &probs)
{
	QFile f(filename);
	f.open(QIODevice::ReadOnly);
	QStringList lines = QString::fromUtf8(f.readAll()).split("\n");
	/* first line contains labels */
	lines.removeFirst();

	QList<QStringList> resultsList;
	foreach (QString line, lines) {
		QStringList vals = line.trimmed().split(" ");
		if (vals.size() < 3)
			continue;
		resultsList << vals;
	}
	f.close();

	QList<int> results;
	probs.clear();
	for (int i = 0; i < resultsList.size(); i++) {
		const QStringList vals = resultsList[i];
		results << vals[0].toInt();
		probs << QList<float>();
		for (int j = 1; j < vals.size(); j++)
			probs[i] << vals[j].toFloat();
	}
	return results;
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

QHash<QString, int> getCategories(const QString &filename)
{
	QHash<QString, int> categories;
	QFile f(filename);
	f.open(QIODevice::ReadOnly);
	QStringList lines = QString::fromUtf8(f.readAll()).split("\n");
	foreach (QString line, lines) {
		if (!line.contains(":"))
			line.append(QString(":%1").arg(categories.size() + 1));
		categories.insert(line.split(":").first(), line.split(":").last().toInt());
	}
	return categories;
}

void Snippets::getAP(const QString &resultsFile, const QString &predictInputs, const QString &categories)
{
	QHash<QString, int> cats = getCategories(categories);
	int ccnt = cats.size();
	QList<QList<float> > probs;
	QList<int> results = readSvmResults(resultsFile, probs);
	QList<int> truth = getPredictInputs(predictInputs);

	/* calculate per-class average precisions */
	float map = 0;
	QList<float> APs;
	for (int i = 0; i < ccnt; i++) {
		float ap = getAveragePrecision(i + 1, results, probs, truth);
		map += ap;
		APs << ap;
		qDebug("%s AP: %f", qPrintable(cats.key(i + 1)), ap);
	}
	qDebug("mean average precision is %f", map / ccnt);
}

void Snippets::toVOCKit(const QString &path)
{
	QStringList cats = QStringList()
		<< "aeroplane"
		<< "bicycle"
		<< "boat"
		<< "bird"
		<< "bottle"
		<< "bus"
		<< "car"
		<< "cat"
		<< "chair"
		<< "cow"
		<< "diningtable"
		<< "dog"
		<< "horse"
		<< "motorbike"
		<< "person"
		<< "pottedplant"
		<< "sheep"
		<< "sofa"
		<< "train"
		<< "tvmonitor";

	foreach (const QString &cat, cats) {
		QStringList lines;
		QList<QList<float> > probs;
		readSvmResults(QString("%1/%2/svm.predict").arg(path).arg(cat), probs);
		QList<QPair<int, QString> > list = DatasetManager::voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "test", cat);
		assert(probs.size() == list.size());
		for (int i = 0; i < probs.size(); i++) {
			QString id = list[i].second.split("/").last().remove(".jpg");
			float pr = probs[i].first();
			/*if (list[i].first < 0)
				lines << QString("%1 %2").arg(id).arg(0);
			else
				lines << QString("%1 %2").arg(id).arg(1);*/
			//lines << QString("%1 %2").arg(id).arg(pr);
			lines << QString("%1 %2").arg(id).arg((rand() % 100) / 100.0);
		}
		Common::exportText(lines.join("\n"), QString("vocsvm/comp1_cls_test_%1.txt").arg(cat));
	}

}

void Snippets::oxfordTemp()
{
#if 0
	/* pyramids */
	p->dm->addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/work/ox_complete/oxbuild_images/");
	Common::calculatePyramids(p->dm->dataSetImages("oxford"), p->pyr);
#elif 0
	/* query */
	p->dm->addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/work/ox_complete/oxbuild_images/");
	Common::calculateQuery(p->dm->dataSetImages("oxford"), p->pyr);
#elif 0
	QStringList lines = Common::importText("ranked_list.txt");
	QStringList good = Common::importText("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/all_souls_1_good.txt");
	QStringList ok = Common::importText("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/all_souls_1_ok.txt");
	QStringList junk = Common::importText("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/all_souls_1_junk.txt");
	int mcnt = 0;
	int valid = 0;
	int posTotal = good.size() + ok.size();
	for (int i = 0; i < 100; i++) {
		QString line = lines[i];
#if 0
		QString res;
		if (good.contains(line))
			res = "good";
		else if (ok.contains(line))
			res = "ok";
		else if (junk.contains(line))
			res = "junk";
		else
			res = "bad";
		qDebug() << i << res;
#else
		if (junk.contains(line))
			continue;
		if (good.contains(line) || ok.contains(line))
			mcnt++;
		float recall = mcnt / (float)posTotal;
		float precision = mcnt / (valid + 1.0);
		qDebug() << recall << precision;
		valid++;
#endif
	}

#endif
}

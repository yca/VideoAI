#include "snippets.h"
#include "datasetmanager.h"
#include "debug.h"
#include "common.h"

#include "opencv/opencv.h"

#include "vision/pyramids.h"
#include "vision/pyramidsvl.h"

#include "vlfeat/vlfeat.h"

#include "svm/liblinear.h"

#include <QDir>
#include <QFile>

#include <iostream>

typedef float precType;
#define normalizer(__m) OpenCV::getL1Norm(__m)
//#define distcalc(__m1, __m2) OpenCV::getEMDDist(__m1, __m2)
//#define distcalc(__m1, __m2) EMD(__m1, __m2, CV_DIST_L1)
#define distcalc(__m1, __m2) OpenCV::getL2Norm(__m1, __m2)
//#define distcalc(__m1, __m2) cv::compareHist(__m1, __m2, CV_COMP_CHISQR)
#define DIST_SORT_ORDER CV_SORT_ASCENDING

static Mat dcorr, dcorrD, dcorrCV;

static Mat computeQueryAP(const Mat &dists, const QStringList &images, const QStringList &queryFileNames, int maxResults);
static vector<Mat> computeQueryPR(const Mat &dists, const QStringList &images, const QStringList &queryFileNames);

enum distMetric {
	DISTM_L1,
	DISTM_L2,
	DISTM_HELLINGER,
	DISTM_HELLINGER2,
	DISTM_KL,
	DISTM_CHI2,
	DISTM_MAX,
	DISTM_MIN,
	DISTM_COS,
	DISTM_MAH,
};

static QStringList getVOCCategories()
{
	return QStringList()
			<< "aeroplane";/*
			<< "bicycle"
			<< "bird"
			<< "boat"
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
			<< "tvmonitor";*/
}

static void exportPyramidsForSvm(const QStringList &images, const Mat &pyramids, const QList<QPair<int, QString> > &list, const QString &output, double gamma)
{
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
	VlFeat::exportToSvm(all, labels, output, gamma);
}

static QPair<Mat, Mat> getDataPyramids(const QStringList &images, const Mat &pyramids, const QList<QPair<int, QString> > &list)
{
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
	return QPair<Mat, Mat>(labels, all);
}

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

void Snippets::voc2007(int step, int L, int H)
{
	DatasetManager dm;
	dm.addDataset("voc", "/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages/");

	PyramidsVl pyr;

	Mat dict = OpenCV::importMatrixTxt("/home/caglar/Downloads/codebook.txt");
	pyr.setDict(dict);
	//pyr.setDict("/home/caglar/myfs/source-codes/personal/build_x86/videoai/data/voc5000.dict");

	Mat pyramids;
	if (!H)
		pyramids = pyr.calculatePyramids(dm.dataSetImages("voc"), L, step);
	else
		pyramids = pyr.calculatePyramidsH(dm.dataSetImages("voc"), L, H, step);

	OpenCV::exportMatrix(QString("data/pyramids_min_5000_L%1H%3_s%2.dat").arg(L).arg(step).arg(H), pyramids);
}

void Snippets::voc2007Min(int step, int L, int H)
{
	DatasetManager dm;
	dm.addDataset("voc", "/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages/");

	Pyramids pyr;

	Mat dict = OpenCV::importMatrixTxt("/home/caglar/Downloads/codebook.txt");
	pyr.setDict(dict);
	//pyr.setDict("/home/caglar/myfs/source-codes/personal/build_x86/videoai/data/voc5000.dict");

	Mat pyramids;
	if (!H)
		pyramids = pyr.calculatePyramids(dm.dataSetImages("voc"), L, step);
	else
		pyramids = pyr.calculatePyramidsH(dm.dataSetImages("voc"), L, H, step);

	//OpenCV::exportMatrixTxt("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/pyramids.txt", pyramids);
	OpenCV::exportMatrix(QString("data/pyramids_min_5000_L%1H%3_s%2.dat").arg(L).arg(step).arg(H), pyramids);
}

void Snippets::vocTrain(const QString &pyramidData, const QString &subFolder, double gamma, double cost)
{
	DatasetManager *dm = new DatasetManager;
	dm->addDataset("voc", "/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages");
	QStringList images = dm->dataSetImages("voc");
	Mat pyramids = OpenCV::importMatrix(pyramidData);
	QDir out("vocsvm");
	out.mkdir(subFolder);
	out = QDir("vocsvm/" + subFolder);

	QStringList cats = getVOCCategories();

	VlHomogeneousKernelMap *map = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, gamma, 1, -1, VlHomogeneousKernelMapWindowRectangular);

	foreach (QString cat, cats) {
		LibLinear svm;
		fDebug("Processing %s", qPrintable(cat));
		out.mkdir(cat);

		/* training */
		QList<QPair<int, QString> > list = dm->voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "mintrain", cat);
		QPair<Mat, Mat> pair = getDataPyramids(images, pyramids, list);
		Mat m = VlFeat::homKerMap(map, pair.second);
		svm.setDataSize(list.size(), m.cols);
		svm.setCost(cost);
		svm.addData(m, pair.first);
		fDebug("training svm");
		int err = svm.train();
		fDebug("training done");
		if (err) {
			fDebug("error %d for training class %s", err, qPrintable(cat));
			continue;
		}
		svm.save(QString("vocsvm/%2/%1/svm.model").arg(cat).arg(subFolder));
	}

	vl_homogeneouskernelmap_delete(map);
}

void Snippets::vocPredict(const QString &pyramidData, const QString &subFolder, double gamma)
{
	DatasetManager *dm = new DatasetManager;
	dm->addDataset("voc", "/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages");
	QStringList images = dm->dataSetImages("voc");
	Mat pyramids = OpenCV::importMatrix(pyramidData);
	QDir out("vocsvm");
	out.mkdir(subFolder);
	out = QDir("vocsvm/" + subFolder);

	QStringList cats = getVOCCategories();

	VlHomogeneousKernelMap *map = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, gamma, 1, -1, VlHomogeneousKernelMapWindowRectangular);

	foreach (QString cat, cats) {
		LibLinear svm;
		fDebug("Processing %s", qPrintable(cat));
		out.mkdir(cat);

		svm.load(QString("vocsvm/%2/%1/svm.model").arg(cat).arg(subFolder));

		/* testing data */
		QList<QPair<int, QString> > list2 = dm->voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "mintest", cat);
		QPair<Mat, Mat> pair2 = getDataPyramids(images, pyramids, list2);
		Mat m2 = VlFeat::homKerMap(map, pair2.second);
		Mat plabels(m2.rows, 1, CV_32F);
		Mat probs(m2.rows, 2, CV_32F);
		int total = 0, correct = 0;
		for (int i = 0; i < m2.rows; i++) {
			//fDebug("testing %d", i);
			int l = 0;
			double prob[2];
			svm.predict(m2.row(i), &l, prob);
			plabels.at<float>(i, 0) = l;
			probs.at<float>(i, 0) = prob[0];
			probs.at<float>(i, 1) = prob[1];
			if (l == pair2.first.at<float>(i, 0))
				correct++;
			total++;
		}
		OpenCV::exportMatrix(QString("vocsvm/%2/%1/svm.predict_labels").arg(cat).arg(subFolder), plabels);
		OpenCV::exportMatrix(QString("vocsvm/%2/%1/svm.gt_labels").arg(cat).arg(subFolder), pair2.first);
		OpenCV::exportMatrix(QString("vocsvm/%2/%1/svm.predict_probs").arg(cat).arg(subFolder), probs);
		qDebug() << "acc is " << (float)correct / total * 100.0;
	}

	vl_homogeneouskernelmap_delete(map);
}

void Snippets::vocAP(const QString &subFolder)
{
	QStringList cats = getVOCCategories();

	double mAP = 0;
	foreach (QString cat, cats) {
		qDebug() << "processing" << cat;
		Mat plabels = OpenCV::importMatrix(QString("vocsvm/%2/%1/svm.predict_labels").arg(cat).arg(subFolder));
		Mat probs = OpenCV::importMatrix(QString("vocsvm/%2/%1/svm.predict_probs").arg(cat).arg(subFolder));
		Mat gt = OpenCV::importMatrix(QString("vocsvm/%2/%1/svm.gt_labels").arg(cat).arg(subFolder));
		int total = plabels.rows, correct = 0;
		for (int i = 0; i < plabels.rows; i++)
			if (plabels.at<float>(i, 0) == gt.at<float>(i, 0))
				correct++;
		//qDebug() << "acc is " << (float)correct / total * 100.0;
		/* pascal style ap calculation */
		Mat targetProb = probs.col(0);
		//for (int i = 0; i < 100; i++)
			//qDebug() << plabels.at<float>(i) << targetProb.at<float>(i);
		//qDebug() << plabels.at<float>(0) << plabels.at<float>(1) << plabels.at<float>(2) << plabels.at<float>(3);
		//qDebug() << targetProb.at<float>(0) << targetProb.at<float>(1) << targetProb.at<float>(2) << targetProb.at<float>(3);
		Mat sorted;
		sortIdx(targetProb, sorted, SORT_EVERY_COLUMN | SORT_DESCENDING);
		cv::sort(targetProb, targetProb, SORT_EVERY_COLUMN | SORT_DESCENDING);
		//qDebug() << sorted.at<int>(0) << sorted.at<int>(1) << sorted.at<int>(2) << sorted.at<int>(3);
		Mat tp(sorted.rows, 1, CV_32F);
		Mat fp(sorted.rows, 1, CV_32F);
		int gtTotal = 0;
		for (int i = 0; i < sorted.rows; i++) {
			int _gt = gt.at<float>(sorted.at<int>(i));
			if (_gt > 0) {
				tp.at<float>(i) = 1;
				fp.at<float>(i) = 0;
				gtTotal++;
			} else {
				tp.at<float>(i) = 0;
				fp.at<float>(i) = 1;
			}
		}
		for (int i = 1; i < tp.rows; i++) {
			tp.at<float>(i) +=  tp.at<float>(i - 1);
			fp.at<float>(i) +=  fp.at<float>(i - 1);
		}
		Mat rec(tp.rows, 1, CV_32F);
		Mat prec(tp.rows, 1, CV_32F);
		for (int i = 0; i < tp.rows; i++) {
			float _tp = tp.at<float>(i);
			float _fp = fp.at<float>(i);
			rec.at<float>(i) = _tp / gtTotal;
			prec.at<float>(i) = _tp / (_tp + _fp);
		}
		double ap = 0;
		for (double t = 0; t <= 1.0; t += 0.1) {
			float maxPrec = 0;
			for (int i = 0; i < rec.rows; i++) {
				if (rec.at<float>(i) < t)
					continue;
				if (prec.at<float>(i) > maxPrec)
					maxPrec = prec.at<float>(i);
			}
			assert(maxPrec > 0);
			ap += maxPrec;
		}
		qDebug() << "AP is" << ap / 11;
		mAP += ap / 11;
	}
	qDebug() << "mAP is" << mAP / cats.size();
#if 1
#else
	QList<QPair<int, float> > out;
	for (int i = 0; i < results.size(); i++)
		out << QPair<int, float>(i, probs[i][cl]);
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
#endif
}

void Snippets::vocpyr2linearsvm(const QString &pyramidData, const QString &subFolder, double gamma)
{
	srand(time(NULL));
	DatasetManager *dm = new DatasetManager;
	dm->addDataset("voc", "/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages");
	QStringList images = dm->dataSetImages("voc");
	Mat pyramids = OpenCV::importMatrix(pyramidData);
	QDir out("vocsvm");
	out.mkdir(subFolder);
	out = QDir("vocsvm/" + subFolder);

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
		exportPyramidsForSvm(images, pyramids, list, QString("vocsvm/%2/%1/svm_train.txt").arg(cat).arg(subFolder), 1);
		/* testing data */
		list = dm->voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "test", cat);
		exportPyramidsForSvm(images, pyramids, list, QString("vocsvm/%2/%1/svm_test.txt").arg(cat).arg(subFolder), gamma);
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
						  const QList<QHash<int, float> > &probs,
						  const QList<int> &truth)
{
	/* pascal style ap calculation */
	QList<QPair<int, float> > out;
	for (int i = 0; i < results.size(); i++)
		out << QPair<int, float>(i, probs[i][cl]);
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

static QList<int> readSvmResults(QString filename, QList<QHash<int, float> > &probs)
{
	QFile f(filename);
	f.open(QIODevice::ReadOnly);
	QStringList lines = QString::fromUtf8(f.readAll()).split("\n");
	/* first line contains labels */
	QStringList labels = lines.first().split(" ");
	QHash<int, int> labelPos;
	for (int i = 1; i < labels.size(); i++)
		labelPos.insert(i, labels[i].toInt());
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
		probs << QHash<int, float>();
		for (int j = 1; j < vals.size(); j++)
			probs[i].insert(labelPos[j], vals[j].toFloat());
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
	QList<QHash<int, float> > probs;
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
	QStringList cats = getVOCCategories();

	foreach (const QString &cat, cats) {
		if (QFile::exists(QString("%1/%2/svm.predict").arg(path).arg(cat))) {
			QStringList lines;
			QList<QHash<int, float> > probs;
			readSvmResults(QString("%1/%2/svm.predict").arg(path).arg(cat), probs);
			QList<QPair<int, QString> > list = DatasetManager::voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "test", cat);
			assert(probs.size() == list.size());
			for (int i = 0; i < probs.size(); i++) {
				QString id = list[i].second.split("/").last().remove(".jpg");
				float pr = probs[i][1];
				lines << QString("%1 %2").arg(id).arg(pr);
			}
			Common::exportText(lines.join("\n"), QString("vocsvm/comp1_cls_test_%1.txt").arg(cat));
		} else {
			//Mat plabels = OpenCV::importMatrix(QString("%2/%1/svm.predict_labels").arg(cat).arg(path));
			Mat probs = OpenCV::importMatrix(QString("%2/%1/svm.predict_probs").arg(cat).arg(path));
			//Mat gt = OpenCV::importMatrix(QString("%2/%1/svm.gt_labels").arg(cat).arg(path));
			QList<QPair<int, QString> > list = DatasetManager::voc2007GetImagesForCateogory("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007", "test", cat);
			assert(probs.rows == list.size());
			QStringList lines;
			for (int i = 0; i < probs.rows; i++) {
				QString id = list[i].second.split("/").last().remove(".jpg");
				float pr = probs.at<float>(i, 0);
				lines << QString("%1 %2").arg(id).arg(pr);
			}
			Common::exportText(lines.join("\n"), QString("vocsvm/comp1_cls_test_%1.txt").arg(cat));
		}
	}

}

#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <stdlib.h>

static vector<string>
load_list(const string& fname)
{
	vector<string> ret;
	ifstream fobj(fname.c_str());
	if (!fobj.good()) { cerr << "File " << fname << " not found!\n"; exit(-1); }
	string line;
	while (getline(fobj, line)) {
		ret.push_back(line);
	}
	return ret;
}

template<class T>
set<T> vector_to_set(const vector<T>& vec)
{ return set<T>(vec.begin(), vec.end()); }

float
compute_ap(const set<string>& pos, const set<string>& amb, const vector<string>& ranked_list, const Mat &dists)
{
	float old_recall = 0.0;
	float old_precision = 1.0;
	float ap = 0.0;

	size_t intersect_size = 0;
	size_t i = 0;
	size_t j = 0;
	for ( ; i<ranked_list.size(); ++i) {
		if (amb.count(ranked_list[i])) continue;
		int cnt = pos.count(ranked_list[i]);
		if (cnt)
			intersect_size++;
		if (dists.rows)
			cout << (cnt ? "good " : "bad ") << ranked_list[i] << " " << dists.row(i) << endl;

		float recall = intersect_size / (float)pos.size();
		float precision = intersect_size / (j + 1.0);

		ap += (recall - old_recall)*((old_precision + precision)/2.0);

		old_recall = recall;
		old_precision = precision;
		j++;
	}
	return ap;
}

Mat
compute_pr(const set<string>& pos, const set<string>& amb, const vector<string>& ranked_list)
{
	Mat pr(0, 2, CV_32F);
	float old_recall = 0.0;
	float old_precision = 1.0;
	float ap = 0.0;

	size_t intersect_size = 0;
	size_t i = 0;
	size_t j = 0;
	for ( ; i<ranked_list.size(); ++i) {
		if (amb.count(ranked_list[i])) {
			Mat m(1, 2, CV_32F);
			m.at<float>(0, 0) = old_recall;
			m.at<float>(0, 1) = old_precision;
			pr.push_back(m);
			continue;
		}
		int cnt = pos.count(ranked_list[i]);
		if (cnt)
			intersect_size++;

		float recall = intersect_size / (float)pos.size();
		float precision = intersect_size / (j + 1.0);

		ap += (recall - old_recall)*((old_precision + precision)/2.0);
		/*if (recall <= 1.0 && recall >= 0)
			pr.at<float>(i, 0) = recall;
		else
			pr.at<float>(i, 0) = old_recall;
		if (precision <= 1.0 && precision >= 0)
			pr.at<float>(i, 1) = precision;
		else
			pr.at<float>(i, 1) = old_precision;*/
		Mat m(1, 2, CV_32F);
		m.at<float>(0, 0) = recall;
		m.at<float>(0, 1) = precision;
		pr.push_back(m);

		old_recall = recall;
		old_precision = precision;
		j++;
	}
	return pr;
}

static float compute_ap_main(const char *path, const char *query_file)
{
	/*if (argc != 3) {
		cout << "Usage: ./compute_ap [GROUNDTRUTH QUERY] [RANKED LIST]\n";
		return -1;
	}*/

	string gtq = path;

	vector<string> ranked_list = load_list(query_file);
	set<string> good_set = vector_to_set( load_list(gtq + "_good.txt") );
	set<string> ok_set = vector_to_set( load_list(gtq + "_ok.txt") );
	set<string> junk_set = vector_to_set( load_list(gtq + "_junk.txt") );

	set<string> pos_set;
	pos_set.insert(good_set.begin(), good_set.end());
	pos_set.insert(ok_set.begin(), ok_set.end());

	return compute_ap(pos_set, junk_set, ranked_list, Mat());
}

static float compute_ap_main(const char *path, const char *query_file, const Mat &dists)
{
	/*if (argc != 3) {
		cout << "Usage: ./compute_ap [GROUNDTRUTH QUERY] [RANKED LIST]\n";
		return -1;
	}*/

	string gtq = path;

	vector<string> ranked_list = load_list(query_file);
	set<string> good_set = vector_to_set( load_list(gtq + "_good.txt") );
	set<string> ok_set = vector_to_set( load_list(gtq + "_ok.txt") );
	set<string> junk_set = vector_to_set( load_list(gtq + "_junk.txt") );

	set<string> pos_set;
	pos_set.insert(good_set.begin(), good_set.end());
	pos_set.insert(ok_set.begin(), ok_set.end());

	return compute_ap(pos_set, junk_set, ranked_list, dists);
}

static Mat applyIdf(const Mat &m, const Mat &df, int N)
{
	Mat m2(1, m.cols, CV_32F);
	static Mat idf;
	if (idf.rows == 0) {
		/* create idf index */
		idf = Mat(df.rows, df.cols, CV_32F);
		for (int i = 0; i < df.cols; i++) {
			if (df.at<int>(i) == 0)
				idf.at<float>(i) = 0;
			else
				idf.at<float>(i) = log((double)N / df.at<int>(i)) + 1;
		}
	}
	float *c = (float *)idf.data;
	float *src = (float *)m.data;
	float *dst = (float *)m2.data;
	#pragma omp parallel for
	for (int i = 0; i < m.cols; i++) {
		dst[i] = src[i] * c[i];
	}
	return m2;
}

static Mat filterBoW(const Mat &bow)
{
	Mat py2 = Mat::zeros(bow.rows, bow.cols, bow.type());
	bow.copyTo(py2);
	for (int i = 0; i < bow.cols; i++) {
		//if (py2.at<float>(i) < 0.02)
			//py2.at<float>(i) = 0;
		if (py2.at<float>(i) > 0.00)
			py2.at<float>(i) = 1;
	}
	return py2;
}

static Mat postProcBow(const Mat &py, const Mat &df, int N)
{
	Mat bow = py / OpenCV::getL1Norm(py);
	bow = applyIdf(bow, df, N);
	//bow = filterBoW(bow);
	return bow / normalizer(bow);
}

static Mat getBow(const Mat &ids, const Mat &df, int N, distMetric dm)
{
	Mat py = Mat::zeros(1, df.cols, CV_32F);
	for (int j = 0; j < ids.rows; j++)
		py.at<float>(ids.at<uint>(j)) += 1;
	Mat bow = py / OpenCV::getL1Norm(py);
	bow = applyIdf(bow, df, N);
	if (dm == DISTM_L1)
		return bow / OpenCV::getL1Norm(bow);
	if (dm == DISTM_L2)
		return bow / OpenCV::getL2Norm(bow);
	return bow;
}

void Snippets::oxfordTemp()
{
#if 0
	/* inria query preparation */
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	QList<int> queries;
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << i << images.size();
		QFileInfo fi(images[i]);
		int q = fi.baseName().split(".").first().mid(1, 3).toInt() + 1;
		if (queries.contains(q))
			continue;
		Mat im = imread(qPrintable(fi.absoluteFilePath()));
		/* query format is : basename x y w h */
		Common::exportText(QString("%1 0 0 %2 %3").arg(fi.baseName()).arg(im.cols).arg(im.rows),
						   QString("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/%1_query.txt").arg(fi.baseName()));
		queries << q;
	}
	return;
#endif
#if 0
	/* this mahollonabias work */
	Mat dict = OpenCV::importMatrix("data/myoxforddict2.bin");
	FlannBasedMatcher m;
	m.add(std::vector<Mat>(1, dict));
	m.train();
	int nCnt = 15;
	Mat dcorr(dict.rows, nCnt, CV_32S);
	Mat dcorrD(dict.rows, nCnt, CV_32F);
	qDebug() << "trained";
	for (int i = 0; i < dict.rows; i++) {
		//qDebug() << i << dict.rows;
		vector<vector<DMatch> > matches;
		m.knnMatch(dict.row(i), matches, nCnt);
		vector<DMatch> matches2 = matches[0];
		for (uint j = 0; j < matches2.size(); j++) {
			DMatch dmatch = matches2[j];
			dcorr.at<int>(i, j) = dmatch.trainIdx;
			dcorrD.at<float>(i, j) = dmatch.distance;
		}
	}
	OpenCV::exportMatrix("data/dcorr.bin", dcorr);
	OpenCV::exportMatrix("data/dcorrDist.bin", dcorrD);
	return;
#endif
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	Mat dists = OpenCV::importMatrix("data/ox_dist5_l1.bin");
	QStringList queries = Common::listDir("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407", "txt");
	QStringList queryFileNames;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		queryFileNames << q;
	}

	OpenCV::exportMatrixTxt("data/ox_dist5_l1.txt", dists);
	vector<Mat> PRs = computeQueryPR(dists, images, queryFileNames);
	Mat prec(PRs[0].rows, PRs.size(), CV_32F);
	Mat rec(PRs[0].rows, PRs.size(), CV_32F);
	for (uint i = 0; i < PRs.size(); i++) {
		const Mat &m = PRs[i];
		m.col(0).copyTo(rec.col(i));
		m.col(1).copyTo(prec.col(i));
	}
	OpenCV::exportMatrixTxt("data/ox_rec.txt", rec);
	OpenCV::exportMatrixTxt("data/ox_prec.txt", prec);
	Mat APs = computeQueryAP(dists, images, queryFileNames, -1);
	OpenCV::exportMatrixTxt("data/ox_ap.txt", APs);
	return;
	Mat allAPsL1 = OpenCV::importMatrix("data/ox_allap_l1.bin");
	Mat allAPsL2 = OpenCV::importMatrix("data/ox_allap_l2.bin");
	for (int i = 0; i < allAPsL1.rows; i++)
		cout << i << " " << allAPsL1.row(i) << " " << allAPsL2.row(i) << endl;// << allAPsL2.at<float>(i);
	return;

#if 0
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");

	vector<Mat> distsL1;
	vector<Mat> distsL2;
	//vector<Mat> sortedL2;
	for (uint i = 0; i < 7; i++) {
		Mat d1 = OpenCV::importMatrix(QString("data/ox_dist%1_l1.bin").arg(i));
		Mat d2 = OpenCV::importMatrix(QString("data/ox_dist%1_l2.bin").arg(i));
		distsL1.push_back(d1);
		distsL2.push_back(d2);
		cout << d1.row(11).col(0) << " " << d2.row(11).col(0) << " " << images[11].toStdString() << endl;

		/*Mat sorted;
		sortIdx(d1, sorted, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
		sortedL1.push_back(sorted);
		sortIdx(d2, sorted, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
		sortedL2.push_back(sorted);*/
	}
	int q = 0;
	uint sortMetric = 0;
	Mat sortedAll(distsL1[0].rows, distsL1.size(), CV_32S);
	Mat sortedDists(distsL1[0].rows, distsL1.size(), CV_32F);
	for (uint i = 0; i < distsL1.size(); i++) {
		Mat d1 = distsL1[i].col(q);
		Mat d2 = distsL2[i].col(q);
		if (i == 4 || i == 1) //KL/L2 works best with L2
			d1 = d2;
		static Mat sorted;
		if (i == sortMetric) { //sort only for one metric
			sortIdx(d1, sorted, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
			sorted.copyTo(sortedAll.col(i));
			cv::sort(d1, d1, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
			d1.copyTo(sortedDists.col(i));
		} else
			for (int j = 0; j < d1.rows; j++)
				sortedDists.at<float>(j, i) = d1.at<float>(sorted.at<int>(j));

		//qDebug() << i << sorted.at<int>(0) << d1.at<float>(0) << sortedDists.at<float>(0, i);
	}

#if 0
	for (int i = 0; i < 100; i++) {
		const Mat r = sortedAll.row(i);
		vector<int> counts(10000, 0);
		for (int j = 0; j < r.cols; j++)
			counts.at(r.at<int>(j)) += 1;
		vector<int>::iterator mel = max_element(counts.begin(), counts.end());
		int val = std::distance(counts.begin(), mel);
		//cout << sortedAll.row(i) << " " << idx << endl;
		qDebug() << r.at<int>(0) << val;
	}
	return;
#endif

	QStringList queries = Common::listDir("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/", "txt");
	QStringList queryFileNames;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		queryFileNames << q;
	}
	QString queryFileName = queryFileNames[q];
	Mat sorted = sortedAll.col(sortMetric);
	QFile f("/tmp/ranked_list.txt");
	f.open(QIODevice::WriteOnly);
	for (int i = 0; i < images.size(); i++) {
		f.write(QString(images[sorted.at<int>(i, 0)]).remove(".jpg").split("/", QString::SkipEmptyParts).last().append("\n").toUtf8());
	}
	f.close();

	qDebug() <<  compute_ap_main(qPrintable(QString(queryFileName).split("_query.").first()),
					"/tmp/ranked_list.txt", sortedDists);

#if 0
	Mat p1 = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000013_pyr.bin");
	Mat p2 = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/magdalen_000078_pyr.bin");
	Mat p3 = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000006_pyr.bin");
	Mat dp1 = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000013_dpyr.bin");
	Mat dp2 = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/magdalen_000078_dpyr.bin");
	Mat dp3 = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000006_dpyr.bin");
	Mat df = OpenCV::importMatrix("data/ox_df.bin");
	qDebug() << OpenCV::getL1Norm(postProcBow(p1, df, 5062), postProcBow(p2, df, 5062));
	qDebug() << OpenCV::getL1Norm(postProcBow(p1, df, 5062), postProcBow(p3, df, 5062));
	p1 /= OpenCV::getL1Norm(p1);
	p2 /= OpenCV::getL1Norm(p2);
	p3 /= OpenCV::getL1Norm(p3);
	qDebug() << OpenCV::getL1Norm(dp1, dp2);
	qDebug() << OpenCV::getL1Norm(dp1, dp3);
	//qDebug() << OpenCV::getL1Norm(dp1, dp2) << OpenCV::getL1Norm(dp1, dp3);
#endif

#endif
}

float Snippets::oxfordRerank(int q)
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");

	vector<Mat> distsL1;
	vector<Mat> distsL2;
	for (int i = 0; i < 7; i++) {
		Mat d1 = OpenCV::importMatrix(QString("data/ox_dist%1_l1.bin").arg(i));
		Mat d2 = OpenCV::importMatrix(QString("data/ox_dist%1_l2.bin").arg(i));
		distsL1.push_back(d1);
		distsL2.push_back(d2);
	}
	uint sortMetric = 0;
	Mat sortedAll(distsL1[0].rows, distsL1.size(), CV_32S);
	Mat sortedDists(distsL1[0].rows, distsL1.size(), CV_32F);
	Mat sortedPrimal;
	for (uint i = 0; i < distsL1.size(); i++) {
		Mat d1 = distsL1[i].col(q);
		Mat d2 = distsL2[i].col(q);
		if (i == 1) //KL/L2 works best with L2
			d1 = d2;
		if (i == sortMetric) { //sort only for one metric
			sortIdx(d1, sortedPrimal, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
			sortedPrimal.copyTo(sortedAll.col(i));
			cv::sort(d1, d1, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
			d1.copyTo(sortedDists.col(i));
		} else
			for (int j = 0; j < d1.rows; j++)
				sortedDists.at<float>(j, i) = d1.at<float>(sortedPrimal.at<int>(j));
	}
	Mat distmd(sortedDists.rows, 1, CV_32F);
	for (int i = 0; i < sortedDists.rows; i++) {
		Mat m1 = sortedDists.row(0);
		Mat m2 = sortedDists.row(i);
		//m1 /= OpenCV::getL1Norm(m1);
		//m2 /= OpenCV::getL1Norm(m2);
		distmd.at<float>(i) = OpenCV::getL2Norm(m1, m2);
	}
	Mat sorted;
	sortIdx(distmd, sorted, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);

	QStringList queries = Common::listDir("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/", "txt");
	QStringList queryFileNames;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		queryFileNames << q;
	}
	QString queryFileName = queryFileNames[q];
	//Mat sorted = sortedAll.col(sortMetric);
	QFile f("/tmp/ranked_list.txt");
	f.open(QIODevice::WriteOnly);
	for (int i = 0; i < images.size(); i++) {
		int idx = sortedPrimal.at<int>(sorted.at<int>(i));
		//int idx = sortedPrimal.at<int>(i);
		f.write(QString(images[idx]).remove(".jpg").split("/", QString::SkipEmptyParts).last().append("\n").toUtf8());
	}
	f.close();

	return compute_ap_main(qPrintable(QString(queryFileName).split("_query.").first()),
						   "/tmp/ranked_list.txt", Mat());
}

static Mat scaleDistances(const Mat &m, const Mat &sc)
{
	Mat m2(m.rows, m.cols, m.type());
	for (int i = 0; i < m.cols; i++) {
		double min = sc.at<float>(0, i);
		double max = sc.at<float>(1, i);
		double val = m.at<float>(0, i);
		m2.at<float>(0, i) = (val - min) / (max - min);
	}
	return m2;
}

static float caglarHilal(const Mat &m)
{
	double sum = 0;
	for (int i = 0; i < m.cols * m.rows; i++)
		sum += pow(m.at<float>(i), 0.6);
	return sum;
}

void Snippets::oxfordRerank(QList<int> queries)
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");

	QStringList queriesList = Common::listDir("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/", "txt");
	QStringList queryFileNames;
	foreach (const QString &q, queriesList) {
		if (!q.endsWith("_query.txt"))
			continue;
		queryFileNames << q;
	}

	vector<Mat> distsL1;
	vector<Mat> distsL2;
	Mat scaleCoeff(2, 7, CV_32F);
	Mat(Mat::ones(1, 7, CV_32F) * INT_MAX).copyTo(scaleCoeff.row(0));
	Mat(Mat::ones(1, 7, CV_32F) * INT_MIN).copyTo(scaleCoeff.row(1));
	for (int i = 0; i < 7; i++) {
		Mat d1 = OpenCV::importMatrix(QString("data/ox_dist%1_l1.bin").arg(i));
		Mat d2 = OpenCV::importMatrix(QString("data/ox_dist%1_l2.bin").arg(i));
		distsL1.push_back(d1);
		distsL2.push_back(d2);
		double min, max;
		if (i != 1)
			minMaxLoc(d1, &min, &max);
		else
			minMaxLoc(d2, &min, &max);
		if (min < scaleCoeff.at<float>(0, i))
			scaleCoeff.at<float>(0, i) = min;
		if (max > scaleCoeff.at<float>(1, i))
			scaleCoeff.at<float>(1, i) = max;
	}
	//cout << scaleCoeff.row(0) << endl;
	//cout << scaleCoeff.row(1) << endl;

	Mat distNew(images.size(), queries.size(), CV_32F);
	QStringList selectedQueries;
	for (int i = 0; i < queries.size(); i++) {
		int q = queries[i];
		Mat dl1 = Mat::zeros(images.size(), distsL1.size(), CV_32F);
		//Mat dl2 = Mat::zeros(images.size(), distsL1.size(), CV_32F);
		for (uint j = 0; j < distsL1.size(); j++) {
			if (j != 1)
				distsL1[j].col(q).copyTo(dl1.col(j));
			else
				distsL2[j].col(q).copyTo(dl1.col(j));
		}
		for (int j = 0; j < dl1.rows; j++) {
			scaleDistances(dl1.row(j), scaleCoeff).copyTo(dl1.row(j));
			if (j == 0 && i == 0)
				cout << dl1.row(j) << endl;
			double min, max;
			minMaxLoc(dl1.row(j), &min, &max);
			distNew.at<float>(j, i) = mean(dl1.row(j))[0];
			//distNew.at<float>(j, i) = caglarHilal(dl1.row(j));
			//distNew.at<float>(j, i) = dl1.row(j).at<float>(0);
		}
		selectedQueries << queryFileNames[q];
	}
	Mat APs = computeQueryAP(distNew, images, selectedQueries, -1);
	qDebug() << mean(APs)[0];
}

void Snippets::oxfordRerankAll()
{
	float sum = 0;
	for (int i = 0; i < 55; i++) {
		qDebug() << i << 55;
		sum += oxfordRerank(i);
	}
	qDebug() << sum / 55;
}

static Mat getQueryIDs(QString ftPaths, QString queryFileName, vector<KeyPoint> &keypoints)
{
	QString query = Common::importText(queryFileName).first();
	QStringList vals = query.split(" ");
	QString bname = vals[0].remove("oxc1_");
	cv::Rect r(Point2f(vals[1].toFloat(), vals[2].toFloat()), Point2f(vals[3].toFloat(), vals[4].toFloat()));
	vector<KeyPoint> kpts = OpenCV::importKeyPoints(QString(bname).prepend(ftPaths).append(".kpts"));
	Mat ids = OpenCV::importMatrix(QString(bname).prepend(ftPaths).append(".ids"));

	/* filter query keypoints and ids */
	Mat idsF(0, ids.cols, ids.type());
	for (uint i = 0; i < kpts.size(); i++) {
		KeyPoint kpt = kpts[i];
		if (r.contains(kpt.pt)) {
			idsF.push_back(ids.row(i));
			keypoints.push_back(kpt);
		}
	}
	ids = idsF;

	return ids;
}

static Mat getQueryDists(const Mat &py, const QStringList &images, const QString ftPaths, const Mat &df)
{
	Mat dists(images.size(), 1, CV_32F);
	/* due to memory constraints we run query in chunks */
	for (int i = 0; i < images.size(); i++) {
		qDebug() << i << images.size();
		QString iname = images[i].split("/", QString::SkipEmptyParts).last().split(".").first();
		Mat m = OpenCV::importMatrix(QString(iname).prepend(ftPaths).append("_pyr.bin"));
		m = postProcBow(m, df, images.size());

		//float dist = compareHist(m, py, CV_COMP_HELLINGER)
		float dist = distcalc(m, py);
		dists.at<float>(i, 0) = dist;
	}
	return dists;
}

static Mat getQueryDists(const vector<Mat> &py, const QStringList &images, const QString ftPaths, const Mat &df)
{
	Mat dists(images.size(), py.size(), CV_32F);
	/* due to memory constraints we run query in chunks */
	for (int i = 0; i < images.size(); i++) {
		qDebug() << "checking" << i << images.size();
		QString iname = images[i].split("/", QString::SkipEmptyParts).last().split(".").first();
		Mat m = OpenCV::importMatrix(QString(iname).prepend(ftPaths).append("_pyr.bin"));
		m = postProcBow(m, df, images.size());

		for (uint j = 0; j < py.size(); j++) {
			float d = distcalc(m, py[j]);
			dists.at<float>(i, j) = d;
		}
	}
	return dists;
}

static double getKL(double p, double q)
{
	if (fabs(p) < DBL_EPSILON)
		return 0;
	if (fabs(q) < DBL_EPSILON)
		q = 1e-10;
	return p * log(p / q);
}

template<typename T1>
static Mat getQueryDists(const vector<Mat> &py, const QStringList &images, const QString ftPaths, const Mat &df, const vector<vector<int> > &iidx, const vector<vector<T1> > &iidx2, distMetric dist)
{
	Q_UNUSED(ftPaths);
	Q_UNUSED(df);
	Mat dists = Mat::zeros(images.size(), py.size(), sizeof(T1) == sizeof(float) ? CV_32F : CV_64F);
	Mat dists2 = Mat::zeros(images.size(), py.size(), sizeof(T1) == sizeof(float) ? CV_32F : CV_64F);
	Mat dists3 = Mat::zeros(images.size(), py.size(), sizeof(T1) == sizeof(float) ? CV_32F : CV_64F);
	for (uint j = 0; j < py.size(); j++) {
		ffDebug() << j << py.size();
		Mat pyq = py[j];
		for (int k = 0; k < pyq.cols; k++) {
			T1 val = pyq.at<float>(k);
			if (val) {
				const Mat c1;// = dcorr.row(k);
				const Mat cvar;// = dcorrCV.row(k);
				for (int i = 0; i < dists.rows; i++) {
					if (dist == DISTM_L1 || dist == DISTM_CHI2 || dist == DISTM_MAX)
						dists.at<T1>(i, j) += val;
					else if (dist == DISTM_L2)
						dists.at<T1>(i, j) += val * val;
					else if (dist == DISTM_KL)
						dists.at<T1>(i, j) += getKL(val, 0);
					else if (dist == DISTM_COS)
						dists3.at<T1>(i, j) += val * val;
					else if (dist == DISTM_MAH) {
						float sum = 0;
						for (int ii = 0; ii < c1.cols; ii++) {
							int kc = c1.at<int>(ii);
							if (kc == k)
								sum += val * cvar.at<float>(ii);
						}
						dists.at<T1>(i, j) += sum * val;
					} else if (dist == DISTM_HELLINGER2)
						dists.at<T1>(i, j) += val;

				}
			}
			//float c = log((double)images.size() / df.at<int>(k));
			const vector<int> &v = iidx[k];
			const vector<T1> &v2 = iidx2[k];
#if 0
			static int once = 0;
			if (!once) {
				/* This piece of code checks validity index data, i.e. it loads original pyramid and checks value of given index. It may also compare L2 distances */
				Mat impy = Mat::zeros(1, pyq.cols, CV_32F);
				int rest = 3893;
				for (int ii = 0; ii < pyq.cols; ii++) {
					const vector<int> &v = iidx[ii];
					for (uint i = 0; i < v.size(); i++) {
						int ind = v[i];
						if (ind == rest)
							impy.at<float>(ii) = iidx2[ii][i];
					}
				}
				QString iname = images[rest].split("/", QString::SkipEmptyParts).last().split(".").first();
				qDebug() << OpenCV::getL2Norm(impy, pyq) << iname;
				Mat m = OpenCV::importMatrix(QString(iname).prepend(ftPaths).append("_pyr.bin"));
				m = postProcBow(m, df, images.size());
				for (int ii = 0; ii < pyq.cols; ii++)
					if (m.at<float>(ii) != impy.at<float>(ii))
						qDebug() << m.at<float>(ii) << impy.at<float>(ii) << ii;
				once = 1;
			}
#endif
			for (uint i = 0; i < v.size(); i++) {
				int ind = v[i];
				T1 tmp = 0;
				T1 val2 = v2[i];

				if (dist == DISTM_L1)
					tmp = qAbs(val2 - val) - val;
				else if (dist == DISTM_HELLINGER)
					tmp = sqrt(val2 * val);
				else if (dist == DISTM_KL) {
					tmp = getKL(val, val2) + getKL(val2, val) - getKL(val, 0);
				} else if (dist == DISTM_CHI2) {
					tmp = val - val2;
					if (val < DBL_EPSILON)
						tmp = 0;
					else
						tmp = tmp * tmp / val - val;
				} else if (dist == DISTM_MAX)
					tmp = qMax(val2, val) - val;
				else if (dist == DISTM_MIN)
					tmp = -1 * qMin(val2, val);
				else if (dist == DISTM_L2) {
					//dists.at<T1>(ind, j) -= (val * val);
					//tmp = (val2 - val) * 2.0 * (val2 - val) * 2.0;// - (val * val);
					tmp = pow(val2, 2) - 2 * val * val2;
				} else if (dist == DISTM_COS) {
					tmp = val2 * val;
					dists2.at<T1>(ind, j) += val2 * val2;
				} else if (dist == DISTM_MAH) {
					const Mat &c1 = dcorr.row(k);
					const Mat &cvar = dcorrCV.row(k);
					T1 tmp2 = 0;
					for (int ii = 0; ii < c1.cols; ii++) {
						int kc = c1.at<int>(ii);
						float ck = cvar.at<float>(ii);
						if (kc == k) {
							tmp += val2 * ck - val * ck;
							tmp2 += val2 * ck;
						}
					}
					tmp = tmp * val2 - tmp2 * val;
				} else if (dist == DISTM_HELLINGER2)
					tmp = val2 - 2 * sqrt(val) * sqrt(val2);
				dists.at<T1>(ind, j) += tmp;
			}
		}
		if (dist == DISTM_HELLINGER) {
			for (int i = 0; i < dists.rows; i++) {
				T1 r = dists.at<T1>(i, j);
				dists.at<T1>(i, j) = sqrt(qMax(1. - r, 0.));
			}
		} else if (dist == DISTM_COS) {
			for (int i = 0; i < dists.rows; i++) {
				T1 p = dists.at<T1>(i, j);
				T1 q = dists2.at<T1>(i, j);
				T1 r = dists3.at<T1>(i, j);
				dists.at<T1>(i, j) = -1 * p / (sqrt(q) * sqrt(r)); /* remember cosine is an inverse smilarity */
			}
		} else if (dist == DISTM_HELLINGER2) {
			for (int i = 0; i < dists.rows; i++) {
				T1 r = dists.at<T1>(i, j);
				dists.at<T1>(i, j) = sqrt(r) / sqrt(2);
			}
		}
	}

	return dists;
}

static vector<Mat> computeQueryPR(const Mat &dists, const QStringList &images, const QStringList &queryFileNames)
{
	Mat sorted;
	sortIdx(dists, sorted, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);

	vector<Mat> pr;
	for (int j = 0; j < sorted.cols; j++) {
		QFile f("/tmp/ranked_list.txt");
		f.open(QIODevice::WriteOnly);
		for (int i = 0; i < images.size(); i++)
			f.write(QString(images[sorted.at<int>(i, j)]).remove(".jpg").split("/", QString::SkipEmptyParts).last().append("\n").toUtf8());
		f.close();

		string gtq = qPrintable(QString(queryFileNames[j]).split("_query.").first());
		const char *query_file= "/tmp/ranked_list.txt";
		vector<string> ranked_list = load_list(query_file);
		set<string> good_set = vector_to_set( load_list(gtq + "_good.txt") );
		set<string> ok_set = vector_to_set( load_list(gtq + "_ok.txt") );
		set<string> junk_set = vector_to_set( load_list(gtq + "_junk.txt") );

		set<string> pos_set;
		pos_set.insert(good_set.begin(), good_set.end());
		pos_set.insert(ok_set.begin(), ok_set.end());
		pr.push_back(compute_pr(pos_set, junk_set, ranked_list));
	}
	return pr;
}

static Mat computeQueryAP(const Mat &dists, const QStringList &images, const QStringList &queryFileNames, int maxResults = -1)
{
	Mat sorted;
	sortIdx(dists, sorted, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);

#if 0
	Mat dists2(dists.rows, dists.cols, dists.type());
	dists.copyTo(dists2);
	/* query expansion and re-ranking */
	QStringList imagesPos, imagesNeg;
	int pCount = 25, nCount = 150;
	for (int i = 0; i < pCount; i++)
		imagesPos << images[sorted.at<int>(i, 0)];
	for (int i = images.size() - nCount; i < images.size(); i++)
		imagesNeg << images[sorted.at<int>(i, 0)];
	QString ftPaths = "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/";
	LibLinear *svm = new LibLinear;
	//svm->setCost(20);
	double gamma = 0.5;
	static VlHomogeneousKernelMap *map = NULL;
	if (!map)
		map = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, gamma, 1, -1, VlHomogeneousKernelMapWindowRectangular);
	QStringList lines;
	for (int i = 0; i < imagesPos.size(); i++) {
		QFileInfo fi(imagesPos[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		Mat p = OpenCV::importMatrix(QString(prefix).append("_dpyr.bin"));
		p /= OpenCV::getL1Norm(p);
		lines << VlFeat::toSvmLine(map, p, 1);
		//Mat m = VlFeat::homKerMap(map, p);
		//if (i == 0)
			//svm->setDataSize(imagesPos.size() + imagesNeg.size(), p.cols);
		//svm->addData(m, Mat::ones(1, 1, CV_32F));
		//ffDebug() << i << imagesPos.size() << p.cols << prefix;
	}
	for (int i = 0; i < imagesNeg.size(); i++) {
		QFileInfo fi(imagesNeg[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		Mat p = OpenCV::importMatrix(QString(prefix).append("_dpyr.bin"));
		p /= OpenCV::getL1Norm(p);
		lines << VlFeat::toSvmLine(map, p, -1);
		//Mat m = VlFeat::homKerMap(map, p);
		//svm->addData(m, Mat::ones(1, 1, CV_32F) * -1);
		//ffDebug() << i << imagesNeg.size() << p.cols << prefix;
	}
	Common::exportText(lines.join(""), "/tmp/svm.train");
	/* training */
	fDebug("training svm");
	//int err = svm->train();
	QProcess::execute("/home/amenmd/myfs/tasks/hilal_tez/liblinear-1.96/train -c 200 /tmp/svm.train /tmp/svm.model");
	fDebug("training done");
	svm->load("/tmp/svm.model");
	//svm.save(QString("vocsvm/%2/%1/svm.model").arg(cat).arg(subFolder));
	for (int i = 0; i < images.size(); i++) {
		int sidx = sorted.at<int>(i, 0);
		QFileInfo fi(images[sidx]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		Mat p = OpenCV::importMatrix(QString(prefix).append("_dpyr.bin"));
		p /= OpenCV::getL1Norm(p);
		Mat m = VlFeat::homKerMap(map, p);
		int label = 0;
		double probs[2];
		svm->predict(m, &label, probs);
		if (i > 150)
			break;
		if (label < 0)
			dists2.at<float>(sidx) = dists2.at<float>(sorted.at<int>(i + 150, 0));
			//dists2.at<float>(sidx) += 0.1;
		//qDebug() << fi.fileName() << label << probs[0] << dists2.at<float>(sidx);
		//qDebug() << label << probs[0] << probs[1];
	}
	delete svm;
	sortIdx(dists2, sorted, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
#endif

	if (maxResults == -1)
		maxResults = images.size();
	Mat AP(sorted.cols, 1, CV_32F);
	for (int j = 0; j < sorted.cols; j++) {
		QFile f("/tmp/ranked_list.txt");
		f.open(QIODevice::WriteOnly);
		for (int i = 0; i < maxResults; i++)
			f.write(QString(images[sorted.at<int>(i, j)]).remove(".jpg").split("/", QString::SkipEmptyParts).last().append("\n").toUtf8());
		f.close();

		float ap = compute_ap_main(qPrintable(QString(queryFileNames[j]).split("_query.").first()),
						"/tmp/ranked_list.txt");
		AP.at<float>(j) = ap;
	}
	return AP;
}

void Snippets::oxfordRunQueriesPar()
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	/* import idf vector */
	Mat df = OpenCV::importMatrix("data/ox_df.bin");
	vector<vector<int> > iidx;
	vector<vector<precType> > iidx2;
	distMetric normMet = DISTM_L1;
	if (normMet == DISTM_L1) {
		iidx = OpenCV::importVector2("data/ox_iidx_l1.bin");
		iidx2 = OpenCV::importVector2f("data/ox_iidx2_l1.bin");
	} else if (normMet == DISTM_L2) {
		iidx = OpenCV::importVector2("data/ox_iidx_l2.bin");
		iidx2 = OpenCV::importVector2f("data/ox_iidx2_l2.bin");
	}
	/* parse query info */
	QString ftPaths = "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/";
	vector<Mat> qIds, qbes;
	QStringList queries = Common::listDir("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/", "txt");
	QStringList queryFileNames;
	vector<vector<KeyPoint> > keypoints;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		vector<KeyPoint> kpts;
		Mat ids = getQueryIDs(ftPaths, q, kpts);
		qIds.push_back(ids);
		/* calculate query pyramid, i.e. QBE */
		Mat py = getBow(ids, df, images.size(), normMet);
		qbes.push_back(py);
		queryFileNames << q;
		keypoints.push_back(kpts);
	}

	vector<Mat> allDists;
	QList<int> distMetrics;
	distMetrics << DISTM_L1;
	distMetrics << DISTM_L2;
	distMetrics << DISTM_MAX;
	distMetrics << DISTM_MIN;
	distMetrics << DISTM_COS;
	distMetrics << DISTM_HELLINGER;
	distMetrics << DISTM_CHI2;
	Mat allAPs(distMetrics.size(), 1, CV_32F);
	int curr = 0;
	foreach (int dm, distMetrics) {
		Mat dists;
		if (iidx.size())
			dists = getQueryDists<precType>(qbes, images, ftPaths, df, iidx, iidx2, (distMetric)dm);
		else
			dists = getQueryDists(qbes, images, ftPaths, df);
		allDists.push_back(dists);

		/* AP calculation */
		Mat APs = computeQueryAP(dists, images, queryFileNames);
		allAPs.at<float>(curr++) = mean(APs)[0];
	}
	if (normMet == DISTM_L2)
		OpenCV::exportMatrix("data/ox_allap_l2.bin", allAPs);
	else
		OpenCV::exportMatrix("data/ox_allap_l1.bin", allAPs);
	for (uint i = 0; i < allDists.size(); i++)
		OpenCV::exportMatrix(QString("data/ox_dist%1_l%2.bin").arg(i).arg(normMet == DISTM_L1 ? "1" : "2"), allDists[i]);
}

void Snippets::oxfordRunQueries()
{
#if 0
	/* this is mahollanobias trial code */
	dcorr = OpenCV::importMatrix("data/dcorr.bin");
	dcorrD = OpenCV::importMatrix("data/dcorrDist.bin");
	//dcorr = dcorr.col(0);
	Mat dcorr2 = Mat(dcorr.rows, 10, dcorr.type());
	for (int i = 0; i < dcorr2.cols; i++)
		dcorr.col(i).copyTo(dcorr2.col(i));
	//dcorr.col(1).copyTo(dcorr2.col(2));
	dcorr = dcorr2;
	dcorrCV = Mat::ones(dcorr.rows, dcorr.cols, CV_32F);
	for (int i = 0; i < dcorr.rows; i++) {
		double sum = 0;
		double beta = 0.075;
		for (int j = 1; j < dcorr.cols; j++) {
			float dist = dcorrD.at<float>(i, j);
			sum += pow(2.7183, -1 * beta * dist);
		}
		for (int j = 1; j < dcorr.cols; j++) {
			float dist = dcorrD.at<float>(i, j);
			dcorrCV.at<float>(i, j) = pow(2.7183, -1 * beta * dist) / sum;
		}
	}
	ffDebug() << "CV done";
	assert(dcorrCV.type() == CV_32F);
	assert(dcorr.type() == CV_32S);
#endif
	/*Mat cdists1 = OpenCV::importMatrix("data/ox_dists1.bin");
	Mat cdists2 = OpenCV::importMatrix("data/ox_dists2.bin");
	for (int i = 0; i < cdists1.rows; i++)
		qDebug() << cdists1.col(0).at<float>(i) << cdists2.col(0).at<float>(i);distMetric
	return;*/
	distMetric normMet = DISTM_L2;
	distMetric dmet = DISTM_HELLINGER2;
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");

	/* import idf vector */
	Mat df = OpenCV::importMatrix("data/ox_df.bin");
	vector<vector<int> > iidx;
	vector<vector<precType> > iidx2;
	if (normMet == DISTM_L1) {
		iidx = OpenCV::importVector2("data/ox_iidx_l1.bin");
		iidx2 = OpenCV::importVector2f("data/ox_iidx2_l1.bin");
	} else if (normMet == DISTM_L2) {
		iidx = OpenCV::importVector2("data/ox_iidx_l2.bin");
		iidx2 = OpenCV::importVector2f("data/ox_iidx2_l2.bin");
	}

	/* parse query info */
	QString ftPaths = "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/";

	vector<Mat> qIds, qbes;
	QStringList queries = Common::listDir("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/", "txt");
	QStringList queryFileNames;
	vector<vector<KeyPoint> > keypoints;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		vector<KeyPoint> kpts;
		Mat ids = getQueryIDs(ftPaths, q, kpts);
		qIds.push_back(ids);
		/* calculate query pyramid, i.e. QBE */
		Mat py = getBow(ids, df, images.size(), normMet);
		qbes.push_back(py);
		queryFileNames << q;
		keypoints.push_back(kpts);
		qDebug() << q << ids.rows << py.cols << kpts.size();
	}

#if 1
	ffDebug() << "querying";
	/* real operation */
	Mat dists;
	if (iidx.size())
		dists = getQueryDists<precType>(qbes, images, ftPaths, df, iidx, iidx2, dmet);
	else
		dists = getQueryDists(qbes, images, ftPaths, df);
#else
	Mat dists1 = OpenCV::importMatrixTxt("/home/amenmd/myfs/temp/dists1.txt");
	Mat dists2 = OpenCV::importMatrixTxt("/home/amenmd/myfs/temp/dists2.txt");
	qDebug() << dists1.at<float>(3772, 0) << dists1.at<float>(4007, 0);
	Mat dists = dists1;//Mat::zeros(dists1.rows, dists1.cols, dists1.type());
	/*for (int i = 0; i < dists.rows; i++)
		qDebug() << dists1.at<float>(i, 0) << dists2.at<float>(i, 0);
	for (int i = 0; i < dists.cols; i++) {
		if (i < 0)
			dists1.col(i).copyTo(dists.col(i));
		else
			dists2.col(i).copyTo(dists.col(i));*/
	/*Mat sorted1, sorted2;
	sortIdx(dists1, sorted1, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
	sortIdx(dists2, sorted2, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
	for (int i = 0; i < 10; i++) {
		int s1 = sorted1.at<int>(i);
		int s2 = sorted2.at<int>(i);
		qDebug() << s1 << s2 << dists1.at<float>(s1, 0) << dists2.at<float>(s2, 0);
	}
	dists = dists1;*/
	/* AP calculation */
	/*Mat APs1 = computeQueryAP(dists1, images, queryFileNames);
	Mat APs2 = computeQueryAP(dists2, images, queryFileNames);
	qDebug() << mean(APs1)[0] << mean(APs2)[0];*/
#endif
	Mat APs = computeQueryAP(dists, images, queryFileNames);
	qDebug() << mean(APs)[0];
}

void Snippets::oxfordRunQuery()
{
	distMetric normMet = DISTM_L2;
	distMetric dmet = DISTM_L2;
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");

	QString ftPaths = "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/";
	/* import idf vector */
	Mat df = OpenCV::importMatrix("data/ox_df.bin");
	vector<vector<int> > iidx;
	vector<vector<precType> > iidx2;
	if (normMet == DISTM_L1) {
		iidx = OpenCV::importVector2("data/ox_iidx_l1.bin");
		iidx2 = OpenCV::importVector2f("data/ox_iidx2_l1.bin");
	} else if (normMet == DISTM_L2) {
		iidx = OpenCV::importVector2("data/ox_iidx_l2.bin");
		iidx2 = OpenCV::importVector2f("data/ox_iidx2_l2.bin");
	}

	vector<Mat> qIds, qbes;
	QStringList queries = Common::listDir("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/", "txt");
	QStringList queryFileNames;
	vector<vector<KeyPoint> > keypoints;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		vector<KeyPoint> kpts;
		Mat ids = getQueryIDs(ftPaths, q, kpts);
		qIds.push_back(ids);
		/* calculate query pyramid, i.e. QBE */
		Mat py;// = getBow(ids, df, images.size(), normMet);
		qbes.push_back(py);
		queryFileNames << q;
		keypoints.push_back(kpts);
		qDebug() << q << ids.rows << py.cols << kpts.size();
	}

#if 1
	Mat dists = getQueryDists<precType>(qbes, images, ftPaths, df, iidx, iidx2, dmet);
	OpenCV::exportMatrix("data/inria_dists.bin", dists);
#else
	Mat dists = OpenCV::importMatrix("data/inria_dists.bin");
	qDebug() << dists.rows << dists.cols;
#endif

	QStringList lines;
	/* INRIA */
	Mat sorted;
	sortIdx(dists, sorted, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);
	for (int i = 0; i < queries.size(); i++) {
		Mat ranks = sorted.col(i);
		QStringList results;
		for (int j = 0; j < ranks.rows; j++)
			results << images[ranks.at<int>(j)].split("/").last();

		QString q = queries[i];
		QString line = QString("%1").arg(q.split("/").last().split("_").first().append(".jpg"));
		for (int j = 0; j < results.size(); j++)
			line.append(QString(" %1 %2").arg(j).arg(results[j]));
		lines << line;
	}
	Common::exportText(lines.join("\n"), "data/inria_results.txt");
}

void Snippets::oxfordPrepare()
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	dm.convertOxfordFeatures("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/");
}

void Snippets::oxfordMakePyramids()
{
	Mat dict = OpenCV::importMatrix("data/myoxforddict2.bin");
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	int K = dict.rows;
	QStringList images = dm.dataSetImages("oxford");
	Mat df = Mat::zeros(1, K, CV_32S);
	for (int i = 0; i < images.size(); i++) {
		qDebug() << "making pyramid" << i << images.size();
		QString prefix = images[i].split(".").first().replace("oxbuild_images/", "oxford_features/");
		Mat py = Mat::zeros(1, K, CV_32F);
		Mat ids = OpenCV::importMatrix(prefix + ".ids");
		for (int j = 0; j < ids.rows; j++)
			py.at<float>(ids.at<uint>(j)) += 1;
		//py.copyTo(pyramids.row(i));
		OpenCV::exportMatrix(prefix + "_pyr.bin", py);
		//OpenCV::exportMatrixTxt(prefix + "_pyr.txt", py);

		/* df computation */
		Mat tcont = Mat::zeros(1, py.cols, CV_32S);
		for (int j = 0; j < py.cols; j++)
			if (py.at<float>(0, j) > 0)
				tcont.at<int>(0, j) = 1;
		for (int j = 0; j < py.cols; j++)
			if (tcont.at<int>(0, j))
				df.at<int>(0, j) += 1;
	}

	OpenCV::exportMatrix("data/ox_df.bin", df);
	//dm.calculateOxfordIdfs(images, ftPaths, K);
}

void Snippets::oxfordMakeDensePyramids()
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
#if 0
	Mat dict = OpenCV::importMatrix("data/myoxforddict2.bin");
	Mat dict2 = OpenCV::subSampleRandom(dict, 512);
	OpenCV::exportMatrix("data/ox_dense_dict.bin", dict2);
#else
	Mat dict2 = OpenCV::importMatrix("data/ox_dense_dict.bin");
#endif
	Pyramids pyr;
	pyr.setDict(dict2);
	Mat pyramids = pyr.calculatePyramids(images, 1, 8);

	for (int i = 0; i < images.size(); i++) {
		ffDebug() << i << images.size();
		QString prefix = images[i].split(".").first().replace("oxbuild_images/", "oxford_features/");
		OpenCV::exportMatrix(prefix + "_dpyr.bin", pyramids.row(i));
	}
}

void Snippets::oxfordMakeInvertedIndex()
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	vector<vector<int> > iidx;
	vector<vector<precType> > iidx2;
	Mat df = OpenCV::importMatrix("data/ox_df.bin");
	int n = 0;
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << i << images.size();
		QString prefix = images[i].split(".").first().replace("oxbuild_images/", "oxford_features/");
		Mat py = OpenCV::importMatrix(prefix + "pyr.bin");
		Mat pyp = postProcBow(py, df, images.size());
		if (OpenCV::getL1Norm(pyp) == 1)
			n = 1;
		if (OpenCV::getL2Norm(pyp) == 1)
			n = 2;
		if (!iidx.size()) {
			for (int j = 0; j < py.cols; j++) {
				iidx.push_back(vector<int>());
				iidx2.push_back(vector<precType>());
			}
		}
		for (int j = 0; j < py.cols; j++) {
			if (py.at<float>(j) > 0) {
				iidx[j].push_back(i);
				iidx2[j].push_back(pyp.at<float>(j));
			}
		}
	}
	assert(n != 0);
	if (n == 1) {
		OpenCV::exportVector2("data/ox_iidx_l1.bin", iidx);
		OpenCV::exportVector2f("data/ox_iidx2_l1.bin", iidx2);
	} else if (n == 2) {
		OpenCV::exportVector2("data/ox_iidx_l2.bin", iidx);
		OpenCV::exportVector2f("data/ox_iidx2_l2.bin", iidx2);
	}
}

void Snippets::oxfordSpatialRerank()
{
	vector<KeyPoint> qkpts = OpenCV::importKeyPoints("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000013.kpts");
	Mat qfts = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000013.bin");
	vector<KeyPoint> kpts = OpenCV::importKeyPoints("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000019.kpts");
	Mat fts = OpenCV::importMatrix("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/all_souls_000019.bin");

	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(qfts, fts, matches);
	vector<Point2f> qpts, tpts;
	for (uint i = 0; i < matches.size(); i++) {
		tpts.push_back(kpts[matches[i].trainIdx].pt);
		qpts.push_back(qkpts[matches[i].queryIdx].pt);
	}

	Mat mask;
	Mat mrans = findHomography(tpts, qpts, cv::RANSAC, 4, mask);
	qDebug() << mask.rows << mask.cols << countNonZero(mask) << qkpts.size();
}

void Snippets::oxfordCreate()
{
	int K = 1000000;
	QString ftPaths = "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/";
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	Mat clusterData(0, 128, CV_32F);
	//#pragma omp parallel for
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << i << images.size();
		QFileInfo fi(images[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		if (!QFile::exists(prefix + ".bin")) {
#if 0
			Mat im = OpenCV::loadImage(images[i]);
			vector<KeyPoint> kpts = Pyramids::extractKeypoints(im);
			Mat fts = Pyramids::computeFeatures(im, kpts);
#else
			vector<KeyPoint> kpts;
			Mat fts;
			QString ffile = QString(images[i]).append(".hesaff.sift");
			QStringList lines = Common::importText(ffile);
			for (int j = 2; j < lines.size(); j++) {
				if (lines[j].trimmed().isEmpty())
					continue;
				QStringList vals = lines[j].split(" ");
				KeyPoint kpt;
				kpt.pt.x = vals[0].toFloat();
				kpt.pt.y = vals[1].toFloat();
				kpts.push_back(kpt);
				Mat f(1, 128, CV_32F);
				for (int k = 0; k < 128; k++)
					f.at<float>(k) = vals[5 + k].toFloat();
				fts.push_back(f);
			}
#endif
			OpenCV::exportKeyPoints(prefix + ".kpts", kpts);
			OpenCV::exportMatrix(prefix + ".bin", fts);
			/* for dictionary */
			clusterData.push_back(OpenCV::subSampleRandom(fts, 1000));
		}
	}
#if 0
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << "importing" << i << images.size();
		QFileInfo fi(images[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		Mat fts = OpenCV::importMatrix(prefix + ".bin");
		clusterData.push_back(OpenCV::subSampleRandom(fts, 1000));
	}
#endif
#if 1
	ffDebug() << "clustering dictionary";
	Mat dict = clusterData;
	if (clusterData.rows > K)
		dict = Pyramids::clusterFeatures(clusterData, K);
	OpenCV::exportMatrix("data/myoxforddict2.bin", dict);
#else
	Mat dict = OpenCV::importMatrix("data/myoxforddict.bin");
#endif
	clusterData = Mat();
	/* now calculate id's */
	Pyramids pyr;
	pyr.setDict(dict);
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << "id calc" << i << images.size();
		QFileInfo fi(images[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		//vector<KeyPoint> kpts = OpenCV::importKeyPoints(prefix + ".kpts");
		Mat fts = OpenCV::importMatrix(prefix + ".bin");
		std::vector<DMatch> matches = pyr.matchFeatures(fts);
		Mat ids(fts.rows, 1, CV_32S);
		for (uint i = 0; i < matches.size(); i++) {
			int idx = matches[i].trainIdx;
			ids.at<int>(i, 0) = idx;
		}
		OpenCV::exportMatrix(prefix + ".ids", ids);
	}
}

void Snippets::oxfordCreateSoft()
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	Mat dict = OpenCV::importMatrix("data/myoxforddict2.bin");
	Pyramids pyr;
	pyr.setDict(dict);
	QString ftPaths = "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxford_features/";
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << "id calc" << i << images.size();
		QFileInfo fi(images[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		Mat fts = OpenCV::importMatrix(prefix + ".bin");
		vector<vector<DMatch> > matches = pyr.matchFeatures(fts, 3);
		Mat py = Mat::zeros(1, dict.rows, CV_32S);
		for (uint j = 0; j < matches.size(); j++) {
			vector<DMatch> ma2 = matches[j];
			double sum = 0;
			Mat coeff = Mat::zeros(ma2.size(), 1, CV_32F);
			for (uint k = 0; k < ma2.size(); k++) {
				const DMatch &m = ma2[k];
				float dist = m.distance;
				double beta = 0.01;
				coeff.at<float>(k) = pow(2.7183, -1 * beta * dist);
				//qDebug() << dist << coeff.at<float>(k) << OpenCV::getL2Norm(fts.row(j), dict.row(m.trainIdx));
				sum += coeff.at<float>(k);
			}
			for (uint k = 0; k < ma2.size(); k++) {
				const DMatch &m = ma2[k];
				py.at<float>(0, m.trainIdx) += coeff.at<float>(k) / sum;
			}
		}
		//qDebug() << OpenCV::getL1Norm(py) << countNonZero(py);
		OpenCV::exportMatrix(prefix + "_spyr.bin", py);
	}
}

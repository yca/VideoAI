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

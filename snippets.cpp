#include "snippets.h"
#include "datasetmanager.h"
#include "debug.h"

#include "opencv/opencv.h"

#include "vision/pyramids.h"

#include "vlfeat/vlfeat.h"

#include <QFile>

Snippets::Snippets(QObject *parent) :
	QObject(parent)
{
}

void Snippets::caltech1(Pyramids *pyr, DatasetManager *dm)
{
	pyr->setDict("/home/amenmd/myfs/source-codes/personal/build_x86/videoai/data/dict_500.bin");
	dm->addDataset("caltech101", "/home/amenmd/myfs/tasks/hilal_tez/dataset/101_ObjectCategories/");
	Mat pyramids = pyr->calculatePyramids(dm->dataSetImages("caltech101"), 2, 8);
	OpenCV::exportMatrix("data/pyramids_500_L2.dat", pyramids);
}

void Snippets::pyr2linearsvm(DatasetManager *dm, const QString &trainName, const QString &testName)
{
	VlHomogeneousKernelMap *map = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, 0.5, 1, -1, VlHomogeneousKernelMapWindowRectangular);
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

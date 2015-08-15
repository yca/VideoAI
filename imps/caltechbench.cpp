#include "caltechbench.h"
#include "datasetmanager.h"
#include "debug.h"
#include "vision/pyramids.h"
#include "vlfeat/vlfeat.h"

//#define DPATH "/home/caglar/myfs/tasks/video_analysis/data/cal_sub/"
//#define DATASET "cal_sub"

#define DPATH "/home/caglar/myfs/tasks/video_analysis/data/101_ObjectCategories/"
#define DATASET "caltech"

static Mat getBow(const Mat &ids, int cols)
{
	Mat py = Mat::zeros(1, cols, CV_32F);
	for (int j = 0; j < ids.rows; j++)
		py.at<float>(ids.at<uint>(j)) += 1;
	return py / OpenCV::getL1Norm(py);
}

static QList<int> getNeighbours1(int col, int cols)
{
	QList<int> ns;
	ns << col;
	if (col != cols - 1)
		ns << col + 1;
	if (col != 0)
		ns << col - 1;
	return ns;
}

static Mat getNeighbours(int idx, int rows, int cols)
{
	int col = idx % cols;
	int row = idx / cols;
	Mat ns(0, 1, CV_32S);
	QList<int> cns = getNeighbours1(col, cols);
	QList<int> rns = getNeighbours1(row, rows);
	for (int i = 0; i < cns.size(); i++) {
		int n1 = cns[i];
		for (int j = 0; j < rns.size(); j++) {
			int n2 = rns[j];
			if (n1 == col && n2 == row)
				continue;
			Mat m(1, 1, CV_32S);
			m.at<int>(0) = n2 * cols + n1;
			ns.push_back(m);
		}
	}
	return ns;
}

static Mat getRBow(const Mat &imBows, int dimX, int dimY)
{
	Mat desc(1, imBows.rows, CV_32F);
	for (int j = 0; j < imBows.rows; j++) {
		const Mat &bow = imBows.row(j);
		Mat ns = getNeighbours(j, dimY, dimX);
		float sum = 0;
		for (int k = 0; k < ns.rows; k++) {
			sum += OpenCV::getL1Norm(bow, imBows.row(ns.at<int>(k)));
		}
		desc.at<float>(0, j) = sum / ns.rows;
	}
	return desc / OpenCV::getL1Norm(desc);
}

static Mat bow2bow(const Mat &b1, const Mat &b2)
{
	Mat bow(1, b1.cols, b1.type());
	for (int i = 0; i < b1.cols; i++)
		//bow.at<float>(i) = max(b1.at<float>(i), b2.at<float>(i));
		bow.at<float>(i) = b1.at<float>(i) + b2.at<float>(i);
	return bow;
}

static Mat getRBow2(const Mat &imBows, int dimX, int dimY)
{
	Mat desc = Mat::ones(1, imBows.cols * imBows.rows, CV_32F) * INT_MAX;
	for (int j = 0; j < imBows.rows; j++) {
		Mat bow = imBows.row(j);
		Mat ns = getNeighbours(j, dimY, dimX);
		for (int k = 0; k < ns.rows; k++)
			bow = bow2bow(bow, imBows.row(ns.at<int>(k)));
		bow.copyTo(desc.colRange(j * imBows.cols, j * imBows.cols + imBows.cols));
	}
	return desc / OpenCV::getL1Norm(desc);
}

static Mat getRBow3(const Mat &imBows, int dimX, int dimY)
{
	Mat desc(1, 0, CV_32F);
	for (int j = 0; j < imBows.rows; j++) {
		const Mat &bow = imBows.row(j);
		Mat ns = getNeighbours(j, dimY, dimX);
		Mat m(1, ns.rows, CV_32F);
		for (int k = 0; k < ns.rows; k++)
			m.at<float>(k) = OpenCV::getL1Norm(bow, imBows.row(ns.at<int>(k)));
		if (desc.cols == 0)
			desc = m;
		else
			hconcat(m, desc, desc);
	}
	return desc / OpenCV::getL1Norm(desc);
}

static void exportSvmData(const Mat &pyramids, const Mat &labels, const QString &dataPrefix, const vector<Mat> &trainSet, const vector<Mat> &testSet, const Mat &classPos)
{
	Mat trainPyramids(0, pyramids.cols, pyramids.type());
	Mat testPyramids(0, pyramids.cols, pyramids.type());
	Mat trainLabels(0, labels.cols, labels.type());
	Mat testLabels(0, labels.cols, labels.type());

	for (int i = 0; i < pyramids.rows; i++) {
		int label = labels.at<float>(i);
		const Mat &mt = trainSet[label - 1];
		const Mat &me = testSet[label - 1];
		int cp = classPos.at<float>(i);
		if (OpenCV::matContains(mt, cp)) {
			trainPyramids.push_back(pyramids.row(i));
			trainLabels.push_back(labels.row(i));
		} else if (OpenCV::matContains(me, cp)) {
			testPyramids.push_back(pyramids.row(i));
			testLabels.push_back(labels.row(i));
		}
		ffDebug() << i << pyramids.rows;
	}


	double gamma = 0.5;
	VlFeat::exportToSvm(trainPyramids, trainLabels, dataPrefix + "_train.txt", gamma);
	VlFeat::exportToSvm(testPyramids, testLabels, dataPrefix + "_test.txt", gamma);
}

CaltechBench::CaltechBench(QObject *parent) :
	QObject(parent)
{
}

void CaltechBench::createImageFeatures()
{
	DatasetManager dm;
	dm.addDataset(DATASET, DPATH);
	int size = dm.dataSetImages(DATASET).size();
	const QStringList images = dm.dataSetImages(DATASET);
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		const Mat &m = OpenCV::loadImage(iname);
		vector<KeyPoint> kpts = Pyramids::extractDenseKeypoints(m, 4);
		const Mat features = Pyramids::computeFeatures(m, kpts);
		OpenCV::exportKeyPoints(QString(iname).replace(".jpg", ".kpts"), kpts);
		OpenCV::exportMatrix(QString(iname).replace(".jpg", ".bin"), features);
	}
}

void CaltechBench::createImageDescriptors(const QString &dictFileName)
{
	const Mat dict = OpenCV::importMatrix(dictFileName);
	DatasetManager dm;
	dm.addDataset(DATASET, DPATH);
	int size = dm.dataSetImages(DATASET).size();
	const QStringList images = dm.dataSetImages(DATASET);
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		QString prefix = iname.remove(".jpg");
		const Mat ids = OpenCV::importMatrix(prefix + ".ids");
		OpenCV::exportMatrix(prefix + "_pyr.bin", getBow(ids, dict.rows));
	}
}

static Mat getSubBoWs(int dimX, int dimY, int imCols, int imRows, const vector<KeyPoint> &kpts, const Mat &ids, int K)
{
	QHash<int, QHash<int, Mat> > regions;
	int rowMax = 0, colMax = 0;
	float pcntX = (float)imCols / dimX;
	float pcntY = (float)imRows / dimY;
	for (uint j = 0; j < kpts.size(); j++) {
		const KeyPoint &kp = kpts[j];
		int col = floor(kp.pt.x / pcntX);
		int row = floor(kp.pt.y / pcntY);
		if (!regions.contains(row))
			regions.insert(row, QHash<int, Mat>());
		if (!regions[row].contains(col))
			regions[row].insert(col, Mat(0, 1, CV_32S));
		regions[row][col].push_back(ids.row(j));
		if (row > rowMax)
			rowMax = row;
		if (col > colMax)
			colMax = col;
	}
	assert(rowMax == dimY - 1);
	assert(colMax == dimX - 1);
	Mat imBows(0, K, CV_32F);
	for (int j = 0; j <= rowMax; j++) {
		for (int k = 0; k <= colMax; k++) {
			const Mat &m = regions[j][k];
			Mat bow = getBow(m, K);
			imBows.push_back(bow);
		}
	}
	return imBows;
}

void CaltechBench::createImageDescriptors2(const QString &dictFileName, int L, int flags)
{
	const Mat &dict = OpenCV::importMatrix(dictFileName);
	DatasetManager dm;
	dm.addDataset(DATASET, DPATH);
	int size = dm.dataSetImages(DATASET).size();
	const QStringList images = dm.dataSetImages(DATASET);
	//#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		const Mat &img = OpenCV::loadImage(iname);
		QString prefix = iname.remove(".jpg");
		const Mat ids = OpenCV::importMatrix(prefix + ".ids");
		const vector<KeyPoint> kpts = OpenCV::importKeyPoints(prefix + ".kpts");

		Mat py, desc, merged;

		if (flags & 0x2) {
			/* calculate relational-bow descriptor */
			int dimX = 8;
			int dimY = 8;
			Mat imBows = getSubBoWs(dimX, dimY, img.cols, img.rows, kpts, ids, dict.rows);
			desc = getRBow(imBows, dimX, dimY);
		}

		if (flags & 0x1) {
			/* calculate pyramids */
			/*py = getBow(ids, dict.rows);
			for (int j = 1; j <= L; j++) {
				int dim = pow(2, j);
				Mat pyrBows = getSubBoWs(dim, dim, img.cols, img.rows, kpts, ids, dict.rows);
				for (int k = 0; k < pyrBows.rows; k++)
					hconcat(py, pyrBows.row(k), py);
				py /= OpenCV::getL1Norm(py);
			}*/

			py = Pyramids::makeSpmFromIds(ids, L, img.cols, img.rows, kpts, dict.rows);
		}

		if ((flags & 0x3) == 0x3) {
			/* merge 2 descriptors */
			hconcat(desc, py, merged);
		}

		/* export all  descriptors */
		if (flags & 0x1)
			OpenCV::exportMatrix(prefix + "_pyr1.bin", py);
		if (flags & 0x2)
			OpenCV::exportMatrix(prefix + "_pyr2.bin", desc);
		if ((flags & 0x3) == 0x3)
			OpenCV::exportMatrix(prefix + "_pyr3.bin", merged);
	}
}

void CaltechBench::createImageIds(const QString &dictFileName)
{
	const Mat dict = OpenCV::importMatrix(dictFileName);
	Pyramids py;
	py.setDict(dict);
	DatasetManager dm;
	dm.addDataset(DATASET, DPATH);
	int size = dm.dataSetImages(DATASET).size();
	const QStringList images = dm.dataSetImages(DATASET);
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		QString prefix = iname.remove(".jpg");
		const Mat fts = OpenCV::importMatrix(prefix + ".bin");
		vector<DMatch> matches = py.matchFeatures(fts);
		Mat ids(fts.rows, 1, CV_32S);
		for (uint i = 0; i < matches.size(); i++) {
			int idx = matches[i].trainIdx;
			ids.at<int>(i, 0) = idx;
		}
		OpenCV::exportMatrix(prefix + ".ids", ids);
	}
}

void CaltechBench::createDictionary(int K, int subSample)
{
	DatasetManager dm;
	dm.addDataset(DATASET, DPATH);
	int size = dm.dataSetImages(DATASET).size();
	const QStringList images = dm.dataSetImages(DATASET);
	Mat features(0, 128, CV_32F);
	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		Mat m = OpenCV::importMatrix(QString(iname).replace(".jpg", ".bin"));
		if (subSample > 0)
			features.push_back(OpenCV::subSampleRandom(m, subSample));
		else
			features.push_back(m);
	}
	Mat dict = Pyramids::clusterFeatures(features, K);
	OpenCV::exportMatrix(QString("data/dict_%1.bin").arg(K), dict);
}

void CaltechBench::exportForLibSvm()
{
	DatasetManager dm;
	dm.addDataset(DATASET, DPATH);
	int size = dm.dataSetImages(DATASET).size();
	const QStringList images = dm.dataSetImages(DATASET);
	Mat pyramids(0, 0, CV_32F);
	QStringList cats;
	Mat labels(images.size(), 1, CV_32F);
	Mat classPos(images.size(), 1, CV_32F);
	int cp = 0;
	QHash<int, int> sampleCount;

	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		QString prefix = iname.remove(".jpg");
		const Mat pyr = OpenCV::importMatrix(prefix + "_pyr.bin");
		if (pyramids.cols == 0)
			pyramids = Mat(0, pyr.cols, CV_32F);
		pyramids.push_back(pyr);

		/* find label */
		QStringList dirs = iname.split("/", QString::SkipEmptyParts);
		assert(dirs.size() > 2);
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

	/* split into train/test */
	int trcnt = 30;
	int tscnt = 50;
	int total = trcnt + tscnt;
	vector<Mat> trainSet, testSet;
	for (int i = 0; i < cats.size(); i++) {
		int cnt = sampleCount[i + 1];
		Mat idx = OpenCV::createRandomized(0, cnt);
		trainSet.push_back(idx.rowRange(0, 30));
		testSet.push_back(idx.rowRange(30, idx.rows > total ? total : idx.rows));
	}

	exportSvmData(pyramids, labels, "data/svm", trainSet, testSet, classPos);
}

void CaltechBench::exportForLibSvmMulti(int flags)
{
	DatasetManager dm;
	dm.addDataset(DATASET, DPATH);
	int size = dm.dataSetImages(DATASET).size();
	const QStringList images = dm.dataSetImages(DATASET);
	Mat pyramids1(0, 0, CV_32F);
	Mat pyramids2(0, 0, CV_32F);
	Mat pyramids3(0, 0, CV_32F);
	QStringList cats;
	Mat labels(images.size(), 1, CV_32F);
	Mat classPos(images.size(), 1, CV_32F);
	int cp = 0;
	QHash<int, int> sampleCount;

	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		QString prefix = iname.remove(".jpg");
		const Mat pyr1 = OpenCV::importMatrix(prefix + "_pyr1.bin");
		const Mat pyr2 = OpenCV::importMatrix(prefix + "_pyr2.bin");
		const Mat pyr3 = OpenCV::importMatrix(prefix + "_pyr3.bin");
		if (pyramids1.cols == 0)
			pyramids1 = Mat(0, pyr1.cols, CV_32F);
		pyramids1.push_back(pyr1);
		if (pyramids2.cols == 0)
			pyramids2 = Mat(0, pyr2.cols, CV_32F);
		pyramids2.push_back(pyr2);
		if (pyramids3.cols == 0)
			pyramids3 = Mat(0, pyr3.cols, CV_32F);
		pyramids3.push_back(pyr3);

		/* find label */
		QStringList dirs = iname.split("/", QString::SkipEmptyParts);
		assert(dirs.size() > 2);
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

	/* split into train/test */
	int trcnt = 30;
	int tscnt = 50;
	int total = trcnt + tscnt;
	vector<Mat> trainSet, testSet;
	for (int i = 0; i < cats.size(); i++) {
		int cnt = sampleCount[i + 1];
		Mat idx = OpenCV::createRandomized(0, cnt);
		trainSet.push_back(idx.rowRange(0, 30));
		testSet.push_back(idx.rowRange(30, idx.rows > total ? total : idx.rows));
	}

	if (flags & 0x1)
		exportSvmData(pyramids1, labels, "data/svm1", trainSet, testSet, classPos);
	if (flags & 0x2)
		exportSvmData(pyramids2, labels, "data/svm2", trainSet, testSet, classPos);
	if ((flags & 0x3) == 0x3)
		exportSvmData(pyramids3, labels, "data/svm3", trainSet, testSet, classPos);
}

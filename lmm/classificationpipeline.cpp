#include "classificationpipeline.h"
#include "datasetmanager.h"
#include "opencv/cvbuffer.h"
#include "vision/pyramids.h"
#include "vlfeat/vlfeat.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>

#include <QDir>
#include <QFileInfo>

#include <errno.h>

#if CV_MAJOR_VERSION > 2
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/nonfree/features2d.hpp>
#endif
#if CV_MAJOR_VERSION > 2
using namespace xfeatures2d;
#endif

#define DPATH "/home/caglar/myfs/tasks/video_analysis/data/101_ObjectCategories/"
#define DATASET "caltech"

#define createEl(_func, _priv) new OpElement<ClassificationPipeline>(this, &ClassificationPipeline::_func, _priv)
#define createEl2(_func) new OpSrcElement<ClassificationPipeline>(this, &ClassificationPipeline::_func)

class ThreadData
{
public:
	ThreadData()
	{
		py = NULL;
	}

	Pyramids *py;
	VlHomogeneousKernelMap *map;
};

struct TrainInfo
{
	bool useForTrain;
	bool useForTest;
	int label;
};

static Mat getBow(const Mat &ids, int cols)
{
	Mat py = Mat::zeros(1, cols, CV_32F);
	for (int j = 0; j < ids.rows; j++)
		py.at<float>(ids.at<uint>(j)) += 1;
	return py / OpenCV::getL1Norm(py);
}

static RawBuffer createNewBuffer(const vector<KeyPoint> &kpts, const Mat &m, const RawBuffer &buf)
{
	CVBuffer c2(kpts);
	c2.setReferenceMat(m);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	return c2;
}

static RawBuffer createNewBuffer(const Mat &m, const RawBuffer &buf)
{
	CVBuffer c2(m);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	return c2;
}

ClassificationPipeline::ClassificationPipeline(QObject *parent) :
	PipelineManager(parent)
{
	dm = new DatasetManager;
	dm->addDataset(DATASET, DPATH);
	images = dm->dataSetImages(DATASET);

	pars.ft = FEAT_SURF;
	pars.xStep = pars.yStep = 16;
	pars.exportData = true;
	pars.threads = 4;
	pars.K = 2048;
	pars.dictSubSample = 1000;
	pars.useExisting = true;
	pars.createDict = false;
	pars.dataPath = "dataset1";
	pars.gamma = 0.5;
	pars.trainCount = 30;
	pars.testCount = 50;
	pars.L = 0;

	QDir d = QDir::current();
	d.mkpath(pars.dataPath);

	/* create processing pipeline */
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	BaseLmmElement *node1 = p1->getPipe(p1->getPipeCount() - 1);
	QList<BaseLmmElement *> join1;
	for (int i = 0; i < pars.threads; i++) {
		p1->insert(node1, createEl(detectKeypoints, i));
		p1->append(createEl(extractFeatures, i));
		p1->append(createEl(addToDictPool, i));
		p1->append(createEl(createIDs, i));
		p1->append(createEl(createImageDescriptor, i));
		p1->append(createEl(mapDescriptor, i));
		join1 << p1->getPipe(p1->getPipeCount() - 1);
	}
	p1->appendJoin(createEl(exportForSvm, 0), join1);

	p1->end();

	if (!pars.createDict) {
		QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(pars.K).arg(pars.ft);
		dict = OpenCV::importMatrix(fname);
		for (int i = 0; i < pars.threads; i++) {
			ThreadData *data = new ThreadData;
			data->py = new Pyramids;
			data->py->setDict(dict);
			data->map =  vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, pars.gamma, 1, -1, VlHomogeneousKernelMapWindowRectangular);
			threadsData << data;
		}
	}

	/* split into train/test */
	QStringList cats;
	Mat labels(images.size(), 1, CV_32F);
	Mat classPos(images.size(), 1, CV_32F);
	int cp = 0;
	QHash<int, int> sampleCount;
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << i << images.size();
		QString iname = images[i];

		/* find label */
		QFileInfo fi(iname);
		QString cat = fi.dir().dirName();
		if (!cats.contains(cat)) {
			cp = 0;
			cats << cat;
		}
		int l = cats.indexOf(cat) + 1;
		labels.at<float>(i) = l;
		sampleCount[l]++;
		classPos.at<float>(i) = cp++;
	}

	int trcnt = pars.trainCount;
	int tscnt = pars.testCount;
	int total = trcnt + tscnt;
	vector<Mat> trainSet, testSet;
	for (int i = 0; i < cats.size(); i++) {
		int cnt = sampleCount[i + 1];
		Mat idx = OpenCV::createRandomized(0, cnt);
		trainSet.push_back(idx.rowRange(0, 30));
		testSet.push_back(idx.rowRange(30, idx.rows > total ? total : idx.rows));
	}


	for (int i = 0; i < images.size(); i++) {
		TrainInfo *info = new TrainInfo;
		info->useForTrain = info->useForTest = false;

		int label = labels.at<float>(i);
		info->label = label;
		const Mat &mt = trainSet[label - 1];
		const Mat &me = testSet[label - 1];
		int cp = classPos.at<float>(i);
		if (OpenCV::matContains(mt, cp))
			info->useForTrain = true;
		else if (OpenCV::matContains(me, cp))
			info->useForTest = true;
		trainInfo << info;
	}

	trainFile = new QFile(QString("%1/svm_train_ftype%2_K%3_step%4.txt")
						  .arg(pars.dataPath)
						  .arg(pars.ft)
						  .arg(pars.xStep)
						  .arg(pars.L)
						  );
	trainFile->open(QIODevice::WriteOnly);
	testFile = new QFile(QString("%1/svm_test_ftype%2_K%3_step%4.txt")
						  .arg(pars.dataPath)
						  .arg(pars.ft)
						  .arg(pars.xStep)
						  .arg(pars.L)
						  );
	testFile->open(QIODevice::WriteOnly);
}

const RawBuffer ClassificationPipeline::readNextImage()
{
	while (1) {
		if (images.size() <= 0)
			return RawBuffer::eof(this);
		int index = trainInfo.size() - images.size();
		const QString iname = images.takeFirst();
		TrainInfo *info = trainInfo[index];
		if (info->useForTest == false && info->useForTrain == false)
			continue;
		CVBuffer buf(OpenCV::loadImage(iname));
		buf.pars()->metaData = iname.toUtf8();
		buf.pars()->streamBufferNo = index;
		return buf;
	}
}

RawBuffer ClassificationPipeline::detectKeypoints(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer::eof();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	vector<KeyPoint> kpts;
	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString fname = getExportFilename(imname, "kpts");

	/* use existing ones if exists */
	if (pars.useExisting && QFile::exists(fname))
		kpts = OpenCV::importKeyPoints(fname);
	else {
		kpts = extractDenseKeypoints(cbuf->getReferenceMat(), pars.xStep);
		if (pars.exportData)
			OpenCV::exportKeyPoints(fname, kpts);
	}

	return createNewBuffer(kpts, cbuf->getReferenceMat(), buf);
}

RawBuffer ClassificationPipeline::extractFeatures(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "applicaiton/cv-kpts")
		return RawBuffer::eof();
	CVBuffer *cbuf = (CVBuffer *)&buf;
	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString fname = getExportFilename(imname, "bin");
	Mat features;

	if (pars.useExisting && QFile::exists(fname))
		features = OpenCV::importMatrix(fname);
	else {
		features = computeFeatures(cbuf->getReferenceMat(), cbuf->getKeypoints());
		if (pars.exportData)
			OpenCV::exportMatrix(fname, features);
	}

	return createNewBuffer(features, buf);
}

RawBuffer ClassificationPipeline::addToDictPool(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (!pars.createDict)
		return buf;
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer::eof();
	CVBuffer *cbuf = (CVBuffer *)&buf;
	Mat m = OpenCV::subSampleRandom(cbuf->getReferenceMat(), pars.dictSubSample);
	dplock.lock();
	dictPool.push_back(m);
	dplock.unlock();
	return buf;
}

RawBuffer ClassificationPipeline::createIDs(const RawBuffer &buf, int priv)
{
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer::eof();

	tdlock.lock();
	Pyramids *py = NULL;
	if (priv < threadsData.size())
		py = threadsData[priv]->py;
	tdlock.unlock();
	if (!py)
		return buf;

	CVBuffer *cbuf = (CVBuffer *)&buf;
	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString fname = getExportFilename(imname, "ids");
	const Mat &fts = cbuf->getReferenceMat();

	Mat ids;
	if (pars.useExisting && QFile::exists(fname))
		ids = OpenCV::importMatrix(fname);
	else {
		vector<DMatch> matches = py->matchFeatures(fts);
		ids = Mat(fts.rows, 1, CV_32S);
		for (uint i = 0; i < matches.size(); i++) {
			int idx = matches[i].trainIdx;
			ids.at<int>(i, 0) = idx;
		}
		if (pars.exportData)
			OpenCV::exportMatrix(fname, ids);
	}

	return createNewBuffer(ids, buf);
}

RawBuffer ClassificationPipeline::createImageDescriptor(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer::eof();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &ids = cbuf->getReferenceMat();
	return createNewBuffer(getBow(ids, dict.rows), buf);
}

RawBuffer ClassificationPipeline::mapDescriptor(const RawBuffer &buf, int priv)
{
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer::eof();

	tdlock.lock();
	VlHomogeneousKernelMap *map = NULL;
	if (priv < threadsData.size())
		map = threadsData[priv]->map;
	tdlock.unlock();
	if (!map)
		return buf;

	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &ft = cbuf->getReferenceMat();

	Mat m = Mat(1, ft.cols * 3, ft.type());
	for (int j = 0; j < ft.cols; j++) {
		float d[3];
		float val = ft.at<float>(0, j);
		vl_homogeneouskernelmap_evaluate_f(map, d, 1, val);
		m.at<float>(0, j * 3) = d[0];
		m.at<float>(0, j * 3 + 1) = d[1];
		m.at<float>(0, j * 3 + 2) = d[2];
	}

	return createNewBuffer(m, buf);
}

RawBuffer ClassificationPipeline::exportForSvm(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer::eof();

	int index = buf.constPars()->streamBufferNo;
	TrainInfo *info = trainInfo[index];
	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &desc = cbuf->getReferenceMat();
	int label = info->label;
	QString line = QString("%1 ").arg(label);
	float *data = (float *)desc.row(0).data;
	for (int j = 0; j < desc.cols; j++) {
		if (data[j] != 0)
			line.append(QString("%1:%2 ").arg(j + 1).arg(data[j]));
	}
	if (info->useForTrain) {
		trainFile->write(line.toUtf8());
		trainFile->write("\n");
	} else if (info->useForTest) {
		testFile->write(line.toUtf8());
		testFile->write("\n");
	}

	return buf;
}

void ClassificationPipeline::pipelineFinished()
{
	ffDebug() << "finished";
	if (pars.createDict) {
		Mat dict = Pyramids::clusterFeatures(dictPool, pars.K);
		QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(pars.K).arg(pars.ft);
		ffDebug() << fname << dict.rows << dict.cols;
		OpenCV::exportMatrix(fname, dict);
	}
	if (trainFile)
		trainFile->close();
	if (testFile)
		testFile->close();
	stop();
}

QString ClassificationPipeline::getExportFilename(const QString &imname, const QString &suffix)
{
	QFileInfo fi(imname);
	return QString("%1/%2_ftype%3.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix);
}

std::vector<KeyPoint> ClassificationPipeline::extractDenseKeypoints(const Mat &m, int step)
{
	vector<KeyPoint> keypoints;
	DenseFeatureDetector dec(11.f, 1, 0.1f, step, 0);
	dec.detect(m, keypoints);
	return keypoints;
}

std::vector<KeyPoint> ClassificationPipeline::extractKeypoints(const Mat &m)
{
	vector<KeyPoint> keypoints;
	if (pars.ft == FEAT_SIFT) {
		SiftFeatureDetector dec;
		dec.detect(m, keypoints);
	} else if (pars.ft == FEAT_SURF) {
		SurfFeatureDetector dec;
		dec.detect(m, keypoints);
	}
	return keypoints;
}

Mat ClassificationPipeline::computeFeatures(const Mat &m, std::vector<KeyPoint> &keypoints)
{
	Mat features;
	if (pars.ft == FEAT_SIFT) {
		SiftDescriptorExtractor ex;
		ex.compute(m, keypoints, features);
	} else if (pars.ft == FEAT_SURF) {
		SurfDescriptorExtractor ex;
		ex.compute(m, keypoints, features);
	}
	return features;
}

int SourceLmmElement::processBlocking(int ch)
{
	Q_UNUSED(ch);
	return newOutputBuffer(ch, RawBuffer("video/x-raw-yuv", 1024));
}

#include "classificationpipeline.h"
#include "datasetmanager.h"
#include "opencv/cvbuffer.h"
#include "vision/pyramids.h"
#include "vlfeat/vlfeat.h"
#include "common.h"
#include "caffe/caffecnn.h"
#include "buffercloner.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>

#include <QDir>
#include <QFileInfo>
#include <QApplication>

#include <errno.h>

#if CV_MAJOR_VERSION > 2
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/nonfree/features2d.hpp>
#endif
#if CV_MAJOR_VERSION > 2
using namespace xfeatures2d;
#endif

#define createEl(_func, _priv) new OpElement<ClassificationPipeline>(this, &ClassificationPipeline::_func, _priv,  #_func)
#define createEl2(_func) new OpSrcElement<ClassificationPipeline>(this, &ClassificationPipeline::_func, #_func)

class ThreadData
{
public:
	ThreadData()
	{
		py = NULL;
	}

	Pyramids *py;
	VlHomogeneousKernelMap *map;
	QList<CaffeCnn *> cnns;
	QStringList inetCats;
};

static __thread ThreadData *threadData = NULL;

class TrainInfo
{
public:
	TrainInfo()
	{
		preprocess = 0;
	}

	TrainInfo(const TrainInfo *other, int pp)
	{
		useForTest = other->useForTest;
		useForTrain = other->useForTrain;
		label = other->label;
		preprocess = pp;
		imageFileName = other->imageFileName;
	}

	bool useForTrain;
	bool useForTest;
	int label;
	uint preprocess;
	QString imageFileName;
};

static Mat getBow(const Mat &ids, int cols)
{
	Mat py = Mat::zeros(1, cols, CV_32F);
	for (int j = 0; j < ids.rows; j++)
		py.at<float>(ids.at<uint>(j)) += 1;
	return py / OpenCV::getL1Norm(py);
}

static int histCount(int L)
{
	int binCount = 0;
	for (int i = 0; i <= L; i++)
		binCount += pow(4, i);
	return binCount;
}

static RawBuffer createNewBuffer(const vector<KeyPoint> &kpts, const Mat &m, const RawBuffer &buf)
{
	CVBuffer c2(kpts);
	c2.setReferenceMat(m);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	c2.pars()->videoWidth = buf.constPars()->videoWidth;
	c2.pars()->videoHeight = buf.constPars()->videoHeight;
	return c2;
}

static RawBuffer createNewBuffer(const vector<Mat> &fts, const RawBuffer &buf)
{
	CVBuffer c2(fts);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	c2.pars()->videoWidth = buf.constPars()->videoWidth;
	c2.pars()->videoHeight = buf.constPars()->videoHeight;
	return c2;
}

static RawBuffer createNewBuffer(const Mat &m, const RawBuffer &buf)
{
	CVBuffer c2(m);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	c2.pars()->videoWidth = buf.constPars()->videoWidth;
	c2.pars()->videoHeight = buf.constPars()->videoHeight;
	return c2;
}

static Mat mergeBuffersOrdered(const Mat &m1, const Mat &m2, int size1)
{
	Mat m(1, m1.cols + m2.cols, CV_32F);
	if (m1.cols == size1) {
		for (int i = 0; i < m1.cols; i++)
			m.at<float>(0, i) = m1.at<float>(0, i);
		for (int i = 0; i < m2.cols; i++)
			m.at<float>(0, m1.cols + i) = m2.at<float>(0, i);
	} else if (m2.cols == size1) {
		for (int i = 0; i < m1.cols; i++)
			m.at<float>(0, m2.cols + i) = m1.at<float>(0, i);
		for (int i = 0; i < m2.cols; i++)
			m.at<float>(0, i) = m2.at<float>(0, i);
	} else
		assert(0);
	return m;
}

static Mat mergeBuffers(const Mat &m1, const Mat &m2)
{
	Mat m(1, m1.cols + m2.cols, CV_32F);
	for (int i = 0; i < m1.cols; i++)
		m.at<float>(0, i) = m1.at<float>(0, i);
	for (int i = 0; i < m2.cols; i++)
		m.at<float>(0, m1.cols + i) = m2.at<float>(0, i);
	return m;
}

static vector<Mat> subSampleImage(const Mat &img, int L)
{
	vector<Mat> images;
	for (int i = 0; i <= L; i++) {
		int cnt = pow(2, i);
		int w = img.cols / cnt;
		int h = img.rows / cnt;
		for (int j = 0; j < cnt * cnt; j++) {
			int x = (j % cnt) * w;
			int y = (j / cnt) * h;
			images.push_back(img(Rect(x, y, w, h)));
		}
	}
	return images;
}

#define TRANS_CROP1			0x01
#define TRANS_CROP2			0x02
#define TRANS_CROP3			0x04
#define TRANS_CROP4			0x08
#define TRANS_CROP5			0x10
#define TRANS_ROTATE_CW		0x20
#define TRANS_ROTATE_CCW	0x40
#define TRANS_PT			0x80
#define TRANS_FLIP_HOR		0x100

static void augmentTrainData(QList<TrainInfo *> &trainInfo, TrainInfo *info, int dataAug)
{
	if (dataAug & 0x02) {
		/* Razavian style */
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR);
		trainInfo << new TrainInfo(info, TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_CROP5);
		trainInfo << new TrainInfo(info, TRANS_ROTATE_CW);
		trainInfo << new TrainInfo(info, TRANS_ROTATE_CCW);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP5);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_ROTATE_CW);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_ROTATE_CCW);
	}
	if (dataAug & 0x01) {
		/* Kriz style */
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR);
		trainInfo << new TrainInfo(info, TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_CROP5);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP5);
	}
	if (dataAug & 0x04) {
		trainInfo << new TrainInfo(info, TRANS_PT);
	}
}

ClassificationPipeline::ClassificationPipeline(QObject *parent) :
	PipelineManager(parent)
{
	pars.datasetName = "odtu";
	pars.datasetPath = "/home/caglar/myfs/tasks/video_analysis/data/odtu/annotation/dataset/";
	pars.ft = FEAT_SURF;
	pars.xStep = pars.yStep = 3;
	pars.exportData = true;
	pars.threads = 4;
	pars.K = 2048;
	pars.dictSubSample = 0;
	pars.useExisting = true;
	pars.createDict = false;
	pars.dataPath = "dataset2";
	pars.gamma = 1;
	pars.trainCount = 15;
	pars.testCount = 15;
	pars.L = 2;
	pars.maxFeaturesPerImage = 0;
	pars.maxMemBytes = (quint64)1024 * 1024 * 1024 * 2;
	pars.useExistingTrainSet = true;
	pars.cl = CLASSIFY_BOW;
	pars.imFlags = IMREAD_GRAYSCALE;

	init();
}

ClassificationPipeline::ClassificationPipeline(const ClassificationPipeline::parameters &params, QObject *parent) :
	PipelineManager(parent)
{
	pars = params;
	init();
}

const RawBuffer ClassificationPipeline::readNextImage()
{
	if (datasetIndex >= trainInfo.size())
		return RawBuffer(this);
	TrainInfo *info = trainInfo[datasetIndex++];
	if (info->useForTest == false && info->useForTrain == false)
		return RawBuffer("application/empty", 1);
	Mat img = OpenCV::loadImage(info->imageFileName, pars.imFlags);
	int w = img.cols;
	int h = img.rows;
	int modelW = 227 * 2;
	int modelH = 227 * 2;
	int tw = modelW < w ? modelW : w / 2;
	int th = modelH < h ? modelH : h / 2;
	int x = w - tw;
	int y = h - th;
	if (info->preprocess & TRANS_CROP1)
		img = img(Rect(0, 0, tw, th));
	if (info->preprocess & TRANS_CROP2)
		img = img(Rect(x, 0, tw, th));
	if (info->preprocess & TRANS_CROP3)
		img = img(Rect(0, y, tw, th));
	if (info->preprocess & TRANS_CROP4)
		img = img(Rect(x, y, tw, th));
	if (info->preprocess & TRANS_CROP5)
		img = img(Rect(x / 2, y / 2, tw, th));
	if (info->preprocess & TRANS_ROTATE_CW)
		img = OpenCV::rotate(img, pars.rotationDegree);
	if (info->preprocess & TRANS_ROTATE_CCW)
		img = OpenCV::rotate(img, -1 * pars.rotationDegree);
	if (info->preprocess & TRANS_PT)
		img = OpenCV::gammaCorrection(img, 0.5);
	if (info->preprocess & TRANS_FLIP_HOR)
		cv::flip(img, img, 1);

	CVBuffer buf(img);
	buf.pars()->metaData = info->imageFileName.toUtf8();
	buf.pars()->streamBufferNo = datasetIndex - 1;
	buf.pars()->videoWidth = buf.getReferenceMat().cols;
	buf.pars()->videoHeight = buf.getReferenceMat().rows;
	return buf;
}

const RawBuffer ClassificationPipeline::readNextLMDBImageFeature()
{
	tdlock.lock();
	CaffeCnn *c = NULL;
	if (!threadsData[0]->cnns.size()) {
		QString filename = pars.lmdbFeaturePath;
		c = new CaffeCnn;
		c->load(filename);
		threadsData[0]->cnns << c;
	}
	c = threadsData[0]->cnns.first();
	tdlock.unlock();
	if (!c)
		return RawBuffer(this);

	while (1) {
		QString key;
		Mat m = c->readNextFeature(key);
		int ino = key.toInt();
		if (!m.rows)
			return RawBuffer(this);
		if (ino >= trainInfo.size())
			continue;
		if (trainInfo.size() && trainInfo[ino]->useForTrain == false && trainInfo[ino]->useForTest == false)
			continue;
		CVBuffer buf(m);
		buf.pars()->metaData = QString("%1").arg(key).toUtf8();
		buf.pars()->streamBufferNo = ino;
		buf.pars()->videoWidth = m.cols;
		buf.pars()->videoHeight = m.rows;
		return buf;
	}
}

RawBuffer ClassificationPipeline::detectKeypoints(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

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
		return RawBuffer();
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
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();
	CVBuffer *cbuf = (CVBuffer *)&buf;
	Mat m;
	if (pars.dictSubSample)
		m = OpenCV::subSampleRandom(cbuf->getReferenceMat(), pars.dictSubSample);
	else if (pars.maxFeaturesPerImage)
		m = OpenCV::subSampleRandom(cbuf->getReferenceMat(), pars.maxFeaturesPerImage);
	else
		m = cbuf->getReferenceMat();
	dplock.lock();
	dictPool.push_back(m);
	dplock.unlock();
	return buf;
}

RawBuffer ClassificationPipeline::createIDs(const RawBuffer &buf, int priv)
{
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

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
		return RawBuffer();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &ids = cbuf->getReferenceMat();
	if (pars.L == 0)
		return createNewBuffer(getBow(ids, dict.rows), buf);

	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString fname = getExportFilename(imname, "kpts");
	vector<KeyPoint> kpts = OpenCV::importKeyPoints(fname);

	int imW = buf.constPars()->videoWidth;
	int imH = buf.constPars()->videoHeight;
	const Mat &pyr = Pyramids::makeSpmFromIds(ids, pars.L, imW, imH, kpts, pars.K);
	return createNewBuffer(pyr, buf);
}

RawBuffer ClassificationPipeline::mapDescriptor(const RawBuffer &buf, int priv)
{
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

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
		return RawBuffer();

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

RawBuffer ClassificationPipeline::exportForSvmMulti(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);

	if (buf.getMimeType() == "application/cv-mat")
		return exportForSvm(buf, priv);

	if (buf.getMimeType() != "application/cv-matv")
		return RawBuffer();

	int index = buf.constPars()->streamBufferNo;
	TrainInfo *info = trainInfo[index];
	CVBuffer *cbuf = (CVBuffer *)&buf;
	int label = info->label;
	const vector<Mat> &fts = cbuf->getVector();
	for (uint k = 0; k < fts.size(); k++) {
		const Mat &desc = fts[k];

		QString line = QString("%1 ").arg(label);
		float *data = (float *)desc.row(0).data;
		for (int j = 0; j < desc.cols; j++) {
			if (data[j] != 0)
				line.append(QString("%1:%2 ").arg(j + 1).arg(data[j]));
		}
		int multiIndex = k;

		if (multiIndex == trainFilesMulti.size()) {
			QFile *file = new QFile(QString("%1/svm_train_ftype%2_K%3_step%4_L%5_gamma%6.txt")
									.arg(pars.dataPath)
									.arg(pars.ft)
									.arg(pars.K + multiIndex)
									.arg(pars.xStep)
									.arg(pars.L)
									.arg(pars.gamma)
									);
			file->open(QIODevice::WriteOnly);
			trainFilesMulti << file;
			file = new QFile(QString("%1/svm_test_ftype%2_K%3_step%4_L%5_gamma%6.txt")
							 .arg(pars.dataPath)
							 .arg(pars.ft)
							 .arg(pars.K + multiIndex)
							 .arg(pars.xStep)
							 .arg(pars.L)
							 .arg(pars.gamma)
							 );
			file->open(QIODevice::WriteOnly);
			testFilesMulti << file;
		}

		QFile *trainFile = trainFilesMulti[multiIndex];
		QFile *testFile = testFilesMulti[multiIndex];
		if (info->useForTrain) {
			trainFile->write(line.toUtf8());
			trainFile->write("\n");
		} else if (info->useForTest) {
			testFile->write(line.toUtf8());
			testFile->write("\n");
		}
	}

	return buf;
}

RawBuffer ClassificationPipeline::cnnExtract(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	const QList<CaffeCnn *> list = getCurrentThreadCaffe(0);
	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &img = cbuf->getReferenceMat();

	QString featureLayer = pars.cnnFeatureLayer;

	if (pars.targetCaffeModel < 0) {
		/* use all models */
		Mat merged;

		for (int i = 0; i < list.size(); i++) {
			CaffeCnn *c = list[i];
			Mat m = c->extractLinear(img, featureLayer);
			if (pars.featureMergingMethod == 0) {
				if (i == 0)
					merged = Mat::zeros(1, list.size() * m.cols, CV_32F);
				for (int j = 0; j < m.cols; j++)
					merged.at<float>(0, i * m.cols + j) = m.at<float>(0, j);
			} else if (pars.featureMergingMethod == 1) {
				if (i == 0)
					merged = m;
				else
					merged += m;
			} else if (pars.featureMergingMethod == 2) {
				if (i == 0)
					merged = Mat::zeros(1, list.size() * m.cols, CV_32F);
				for (int j = 0; j < m.cols; j++)
					merged.at<float>(0, j) = qMax<float>(m.at<float>(0, j), merged.at<float>(0, j));
			}

		}

		if (pars.featureMergingMethod == 1)
			merged /= list.size();

		if (pars.featureMergingMethod == 2)
			merged /= OpenCV::getL2Norm(merged);

		return createNewBuffer(merged, buf);
	}

	/* use single model */
	CaffeCnn *c = list[pars.targetCaffeModel];
	if (!c)
		return buf;
	if (featureLayer.contains("&"))
		return createNewBuffer(c->extractLinear(img, featureLayer.split("&")), buf);
	return createNewBuffer(c->extractLinear(img, featureLayer), buf);
}

static const vector<Mat> extractCnnFeatures(const Mat &img, const QString layerDesc, const QString &descTypes, CaffeCnn *c, int aug)
{
	QStringList layers;
	if (layerDesc == "__all__")
		layers = c->getBlobbedLayerNames();
	else
		layers = layerDesc.split("&");
	const vector<Mat> fts = c->extractMulti(img, layers, descTypes.split("&"), aug);
	return fts;
}

RawBuffer ClassificationPipeline::cnnExtractMultiFts(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	const QList<CaffeCnn *> list = getCurrentThreadCaffe(0);
	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &img = cbuf->getReferenceMat();
	int index = buf.constPars()->streamBufferNo;
	TrainInfo *info = trainInfo[index];

	if (pars.targetCaffeModel < 0) {
		QStringList cnnLayers = pars.cnnFeatureLayer.split(",");
		QStringList cnnLayerTypes = pars.cnnFeatureLayerType.split(",");
		assert(cnnLayers.size() == list.size());
		vector<vector<Mat> > all;
		for (int i = 0; i < list.size(); ++i) {
			const vector<Mat> fts = extractCnnFeatures(img, cnnLayers[i], cnnLayerTypes[i], list[i], info->useForTest ? pars.dataAug : 0);
			all.push_back(fts);
		}
		if (pars.featureMergingMethod == 0) { //concat all
			vector<Mat> channels;
			for (uint i = 0; i < all.size(); ++i)
				channels.push_back(OpenCV::merge(all[i]));
			return createNewBuffer(OpenCV::merge(channels), buf);
		} else {
			vector<vector<Mat> > all2;
			for (uint i = 0; i < all[0].size(); i++)
				all2.push_back(vector<Mat>());
			for (uint i = 0; i < all.size(); ++i) {
				const vector<Mat> &model = all[i];
				for (uint j = 0; j < model.size(); j++)
					all2[j].push_back(model[j]);
			}
			vector<Mat> channels;
			OpenCV::MergeMethod method = OpenCV::MM_SUM;
			if (pars.featureMergingMethod == 2)
				method = OpenCV::MM_MAX;
			for (uint i = 0; i < all2.size(); ++i)
				channels.push_back(OpenCV::merge(all2[i], method));
			return createNewBuffer(OpenCV::merge(channels), buf);
		}
		assert(0);
	}

	/* use single model */
	CaffeCnn *c = list[pars.targetCaffeModel];
	vector<Mat> fts = extractCnnFeatures(img, pars.cnnFeatureLayer, pars.cnnFeatureLayerType, c, info->useForTest ? pars.dataAug : 0);
	if (pars.featureMergingMethod == 0) //concat
		return createNewBuffer(OpenCV::merge(fts), buf);
	else if (pars.featureMergingMethod == 1) //sum pooling
		return createNewBuffer(OpenCV::merge(fts, OpenCV::MM_SUM), buf);
	else if (pars.featureMergingMethod == 2) //max pooling
		return createNewBuffer(OpenCV::merge(fts, OpenCV::MM_MAX), buf);
	return createNewBuffer(fts, buf);
}

RawBuffer ClassificationPipeline::mergeFeatures(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	static QHash<int, RawBuffer> buffers;
	int sno = buf.constPars()->streamBufferNo;
	if (buffers.contains(sno)) {
		const RawBuffer &buf2 = buffers[sno];
		CVBuffer *cbuf1 = (CVBuffer *)&buf;
		CVBuffer *cbuf2 = (CVBuffer *)&buf2;
#if 0
		int bowSize = histCount(pars.L) * pars.K;
		if (pars.homkermap)
			bowSize *= 3;
		return createNewBuffer(mergeBuffersOrdered(cbuf1->getReferenceMat(), cbuf2->getReferenceMat(), bowSize), buf);
#else
		if (buf.constPars()->duration < buf.constPars()->duration)
			return createNewBuffer(mergeBuffers(cbuf1->getReferenceMat(), cbuf2->getReferenceMat()), buf);
		return createNewBuffer(mergeBuffers(cbuf2->getReferenceMat(), cbuf1->getReferenceMat()), buf);
#endif
	} else
		buffers.insert(sno, buf);

	return RawBuffer("application/empty", 1);
}

RawBuffer ClassificationPipeline::debugBuffer(const RawBuffer &buf, int priv)
{
	//ffDebug() << buf.size() << priv << buf.constPars()->streamBufferNo;
	return createNewBuffer(Mat(1, 1 + priv, CV_32F), buf);
}

RawBuffer ClassificationPipeline::createMulti(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	//vector<KeyPoint> kpts = extractDenseKeypoints(cbuf->getReferenceMat(), pars.xStep);
	//QString imname = QString::fromUtf8(buf.constPars()->metaData);

	const QList<CaffeCnn *> list = getCurrentThreadCaffe(0);
	QString featureLayer = pars.cnnFeatureLayer;
	vector<Mat> features = list[priv]->extractMulti(cbuf->getReferenceMat(), featureLayer.split("@"), pars.cnnFeatureLayerType.split("@"));

	Mat fts = features[0];
#if 0
	/* bow ids */
	tdlock.lock();
	Pyramids *py = NULL;
	if (priv < threadsData.size())
		py = threadsData[priv]->py;
	VlHomogeneousKernelMap *map = NULL;
	if (priv < threadsData.size())
		map = threadsData[priv]->map;
	tdlock.unlock();
	assert(py);
	assert(map);
	Mat ids;
	vector<DMatch> matches = py->matchFeatures(fts);
	ids = Mat(fts.rows, 1, CV_32S);
	for (uint i = 0; i < matches.size(); i++) {
		int idx = matches[i].trainIdx;
		ids.at<int>(i, 0) = idx;
	}
	/* create bow descriptor from ids */
	int imW = buf.constPars()->videoWidth;
	int imH = buf.constPars()->videoHeight;
	const Mat &ft = Pyramids::makeSpmFromIds(ids, pars.L, imW, imH, kpts, pars.K);
	int bowSize = histCount(pars.L) * pars.K;
#else
	int off = 0;
	Mat ft(1, fts.rows * fts.cols, CV_32F);
	for (int i = 0; i < fts.rows; i++)
		for (int j = 0; j < fts.cols; j++)
			ft.at<float>(0, off++) = fts.at<float>(i, j);
	int bowSize = fts.rows * fts.cols;
#endif

	/* apply homgen kernel mapping */
#if 0
	Mat m = Mat(1, ft.cols * 3, ft.type());
	for (int j = 0; j < ft.cols; j++) {
		float d[3];
		float val = ft.at<float>(0, j);
		vl_homogeneouskernelmap_evaluate_f(map, d, 1, val);
		m.at<float>(0, j * 3) = d[0];
		m.at<float>(0, j * 3 + 1) = d[1];
		m.at<float>(0, j * 3 + 2) = d[2];
	}
	bowSize *= 3;
#else
	Mat m = ft;
#endif

	/* merge 2 features */
	RawBuffer bufout = createNewBuffer(mergeBuffersOrdered(m, features[1], bowSize), buf);
	bufout.pars()->duration = priv;
	return bufout;
}

void ClassificationPipeline::pipelineFinished()
{
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
	foreach (QFile *file, trainFilesMulti)
		file->close();
	foreach (QFile *file, testFilesMulti)
		file->close();
	stop();
	ffDebug() << "quitting";
	QApplication::exit();
}

void ClassificationPipeline::init()
{
	dm = new DatasetManager;
	if (pars.datasetName == "ucf101")
		dm->addUCF101(pars.datasetPath, "/home/amenmd/myfs/tasks/video_analysis/dataset/ucf/ucfTrainTestlist");
	else
		dm->addDataset(pars.datasetName, pars.datasetPath);
	images = dm->dataSetImages(pars.datasetName);
	dm->exportImages(pars.datasetName, QString("%1/image_list.txt").arg(pars.dataPath));
	imageCount = images.size();
	finishedCount = 0;
	expectedFrameCount = imageCount;

	if (pars.maxMemBytes) {
		int fsize = 128;
		if (pars.ft == FEAT_SURF)
			fsize = 64;
		else if (pars.ft == FEAT_SIFT)
			fsize = 128;
		else if (pars.ft == FEAT_CNN)
			fsize = 96;
		int dsize = fsize * 4;
		pars.maxFeaturesPerImage = (double)(pars.maxMemBytes) / imageCount / dsize;
		pars.dictSubSample = 0;
	}

	QDir d = QDir::current();
	d.mkpath(pars.dataPath);

	if (!pars.fileListTxt.isEmpty()) {
		QStringList lines = Common::importText(pars.fileListTxt);
		images.clear();
		foreach (const QString &line, lines) {
			QStringList flds = line.split(" ");
			if (flds.size() < 2)
				continue;
			images << flds.first();
		}
		assert(images.size() == imageCount);
	}

	/* create processing pipeline */
	if (pars.createDict)
		createDictPipeline();
	else if (pars.cl == CLASSIFY_BOW)
		createBOWPipeline();
	else if (pars.cl == CLASSIFY_CNN_SVM)
		createCNNFSVMPipeline();
	else if (pars.cl == CLASSIFY_CNN_BOW)
		createCNNBOWPipeline();
	else if (pars.cl == CLASSIFY_CNN_MULTIFTS)
		createCNNMultiFts();

	for (int i = 0; i < pars.threads; i++) {
		ThreadData *data = new ThreadData;
		data->py = NULL;
		data->map =  NULL;
		threadsData << data;
	}

	if (!pars.createDict) {
		QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(pars.K).arg(pars.ft);
		dict = OpenCV::importMatrix(fname);
		for (int i = 0; i < pars.threads; i++) {
			ThreadData *data = threadsData[i];
			data->py = new Pyramids;
			data->py->setDict(dict);
			data->map =  vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, pars.gamma, 1, -1, VlHomogeneousKernelMapWindowRectangular);
		}

		QString trainSetFileName = QString("%1/train_set.txt")
				.arg(pars.dataPath);
		if (QFile::exists(trainSetFileName) && pars.useExistingTrainSet) {
			QStringList lines = Common::importText(trainSetFileName);
			int ind = 0;
			foreach (const QString &line, lines) {
				QStringList vals = line.split(":");
				if (vals.size() != 3)
					continue;
				TrainInfo *info = new TrainInfo;
				info->label = vals[0].toInt();
				info->useForTrain = vals[1].toInt();
				info->useForTest = vals[2].toInt();
				info->imageFileName = images[ind++];
				trainInfo << info;

				if (info->useForTrain)
					augmentTrainData(trainInfo, info, pars.dataAug);
			}
		} else {
			/* split into train/test */
			if (pars.trainListTxt.isEmpty())
				createTrainTestSplit(trainSetFileName);
			else {
				QHash<QString, int> tthash;
				QHash<QString, int> cats;
				QStringList trainList = Common::importText(pars.trainListTxt);
				for (int j = 0; j < trainList.size(); j++) {
					QStringList flds = trainList[j].trimmed().split(" ");
					QString name = flds[0].remove(".avi");
					if (name.isEmpty())
						continue;
					QStringList vals = name.split("/");
					tthash.insert(vals.last(), 1);
					int cat = flds[1].trimmed().toInt();
					assert(cat);
					cats.insert(vals.first(), cat);
				}
				QStringList testList = Common::importText(pars.testListTxt);
				for (int j = 0; j < testList.size(); j++) {
					QStringList flds = testList[j].trimmed().split(" ");
					QString name = flds[0].remove(".avi");
					if (name.isEmpty())
						continue;
					tthash.insert(name.split("/").last(), 2);
				}
				QStringList lines;
				for (int i = 0; i < images.size(); i++) {
					TrainInfo *info = new TrainInfo;
					info->useForTrain = info->useForTest = false;
					QFileInfo fi(images[i]);
					QStringList flds = fi.baseName().split("_");
					flds.removeLast();
					QString key = flds.join("_");
					int val = tthash[key];
					if (val == 1)
						info->useForTrain = true;
					else if (val == 2)
						info->useForTest = true;
					info->label = cats[fi.dir().absolutePath().split("/").last()];
					info->imageFileName = images[i];
					assert(info->label);
					trainInfo << info;
					lines << QString("%1:%2:%3").arg(info->label).arg(info->useForTrain).arg(info->useForTest);

					if (info->useForTrain)
						augmentTrainData(trainInfo, info, pars.dataAug);
				}
				lines << "";
				Common::exportText(lines.join("\n"), trainSetFileName);
			}
		}

		trainFile = new QFile(QString("%1/svm_train_ftype%2_K%3_step%4_L%5_gamma%6.txt")
							  .arg(pars.dataPath)
							  .arg(pars.ft)
							  .arg(pars.K)
							  .arg(pars.xStep)
							  .arg(pars.L)
							  .arg(pars.gamma)
							  );
		trainFile->open(QIODevice::WriteOnly);
		testFile = new QFile(QString("%1/svm_test_ftype%2_K%3_step%4_L%5_gamma%6.txt")
							  .arg(pars.dataPath)
							  .arg(pars.ft)
							  .arg(pars.K)
							  .arg(pars.xStep)
							  .arg(pars.L)
							  .arg(pars.gamma)
							  );
		testFile->open(QIODevice::WriteOnly);
		expectedFrameCount = 0;
		datasetIndex = 0;
		for (int i = 0; i < trainInfo.size(); i++)
			if (trainInfo[i]->useForTest || trainInfo[i]->useForTrain)
				expectedFrameCount++;
	}
}

void ClassificationPipeline::createDictPipeline()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	BaseLmmElement *node1 = p1->getPipe(p1->getPipeCount() - 1);
	QList<BaseLmmElement *> join1;
	for (int i = 0; i < pars.threads; i++) {
		p1->insert(node1, createEl(detectKeypoints, i));
		p1->append(createEl(extractFeatures, i));
		p1->append(createEl(addToDictPool, i));
		join1 << p1->getPipe(p1->getPipeCount() - 1);
	}
	p1->end(join1);
	//p1->appendJoin(createEl(addToDictPool, 0), join1);
	//p1->end();
}

void ClassificationPipeline::createBOWPipeline()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	BaseLmmElement *node1 = p1->getPipe(p1->getPipeCount() - 1);
	QList<BaseLmmElement *> join1;
	for (int i = 0; i < pars.threads; i++) {
		p1->insert(node1, createEl(detectKeypoints, i));
		p1->append(createEl(extractFeatures, i));
		p1->append(createEl(createIDs, i));
		p1->append(createEl(createImageDescriptor, i));
		if (pars.homkermap)
			p1->append(createEl(mapDescriptor, i));
		join1 << p1->getPipe(p1->getPipeCount() - 1);
	}
	p1->appendJoin(createEl(exportForSvm, 0), join1);
	p1->end();
}

void ClassificationPipeline::createCNNFSVMPipeline()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(createEl(cnnExtract, 0));
	if (pars.homkermap)
		p1->append(createEl(mapDescriptor, 0));
	p1->append(createEl(exportForSvm, 0));
	p1->end();
}

void ClassificationPipeline::createCNNBOWPipeline()
{
#if 0
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(new BufferCloner);

	/* create split node */
	BaseLmmElement *node1 = p1->getPipe(p1->getPipeCount() - 1);
	QList<BaseLmmElement *> join1;

	/* create bow part */
	//p1->insert(node1, createEl(debugBuffer, 0));
	p1->insert(node1, createEl(detectKeypoints, 0));
	p1->append(createEl(extractFeatures, 0));
	p1->append(createEl(createIDs, 0));
	p1->append(createEl(createImageDescriptor, 0));
	if (pars.homkermap)
		p1->append(createEl(mapDescriptor, 0));
	join1 << p1->getPipe(p1->getPipeCount() - 1);

	/* create cnn part */
	//p1->insert(node1, createEl(debugBuffer, 1), 1);
	p1->insert(node1, createEl(cnnExtract, 0), 1);
	if (pars.homkermap)
		p1->append(createEl(mapDescriptor, 0));
	join1 << p1->getPipe(p1->getPipeCount() - 1);

	/* now join */
	//p1->appendJoin(createEl(debugBuffer, 2), join1);
	p1->appendJoin(createEl(mergeFeatures, 0), join1);
	p1->append(createEl(exportForSvm, 0), 0);
	p1->end();
#elif 0
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(createEl(createMulti, 0));
	p1->append(createEl(exportForSvm, 0), 0);
	p1->end();
#else
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(new BufferCloner);

	/* create split node */
	BaseLmmElement *node1 = p1->getPipe(p1->getPipeCount() - 1);
	QList<BaseLmmElement *> join1;

	p1->insert(node1, createEl(createMulti, 0), 0);
	join1 << p1->getPipe(p1->getPipeCount() - 1);

	p1->insert(node1, createEl(createMulti, 1), 1);
	join1 << p1->getPipe(p1->getPipeCount() - 1);

	p1->appendJoin(createEl(mergeFeatures, 0), join1);
	p1->append(createEl(exportForSvm, 0), 0);
	p1->end();
#endif
}

void ClassificationPipeline::createCNNMultiFts()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(createEl(cnnExtractMultiFts, 0));
	p1->append(createEl(exportForSvmMulti, 0));
	p1->end();
}

void ClassificationPipeline::createTrainTestSplit(const QString &trainSetFileName)
{
	QStringList cats;
	Mat labels(images.size(), 1, CV_32F);
	Mat classPos(images.size(), 1, CV_32F);
	int cp = 0;
	QHash<int, int> sampleCount;
	for (int i = 0; i < images.size(); i++) {
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
		/* special case: if trcnt = -1 and tscnt = 0, all will be used for training */
		if (trcnt > 0)
			trainSet.push_back(idx.rowRange(0, trcnt));
		else
			trainSet.push_back(idx);
		if (tscnt > 0)
			testSet.push_back(idx.rowRange(trcnt, idx.rows > total ? total : idx.rows));
	}

	QStringList lines;
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
		info->imageFileName = images[i];
		trainInfo << info;
		lines << QString("%1:%2:%3").arg(info->label).arg(info->useForTrain).arg(info->useForTest);

		if (info->useForTrain)
			augmentTrainData(trainInfo, info, pars.dataAug);
	}
	lines << "";
	Common::exportText(lines.join("\n"), trainSetFileName);
}

QString ClassificationPipeline::getExportFilename(const QString &imname, const QString &suffix)
{
	QFileInfo fi(imname);
	return QString("%1/%2_ftype%3.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix);
}

std::vector<KeyPoint> ClassificationPipeline::extractDenseKeypoints(const Mat &m, int step)
{
	vector<KeyPoint> keypoints;
	if (pars.ft == FEAT_CNN) {
		for (int i = 0; i < pars.spatialSize; i++) {
			for (int j = 0; j < pars.spatialSize; j++) {
				KeyPoint kpt;
				int step = m.cols / pars.spatialSize;
				kpt.pt.x = j * step;
				kpt.pt.y = i * step;
				keypoints.push_back(kpt);
			}
		}
	} else {
		DenseFeatureDetector dec(11.f, 1, 0.1f, step, 0);
		dec.detect(m, keypoints);
	}
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
	} else if (pars.ft == FEAT_CNN) {
		for (int i = 0; i < 55; i++) {
			for (int j = 0; j < 55; j++) {
				KeyPoint kpt;
				kpt.pt.x = j * 4;
				kpt.pt.y = i * 4;
				keypoints.push_back(kpt);
			}
		}
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
	} else if (pars.ft == FEAT_CNN) {
		const QList<CaffeCnn *> list = getCurrentThreadCaffe(0);
		QString featureLayer = pars.cnnFeatureLayer;
		return list[0]->extract(m, featureLayer);
	}
	return features;
}

const QList<CaffeCnn *> ClassificationPipeline::getCurrentThreadCaffe(int priv)
{
#if 0
	tdlock.lock();
	if (priv < threadsData.size()) {
		if (!threadsData[priv]->cnns.size()) {
			QString cbase = pars.caffeBaseDir;
			QString deployProto = pars.caffeDeployProto;
			QString modelFile = pars.caffeModelFile;
			QString imageMeanProto = pars.caffeImageMeanProto;

			if (!deployProto.contains(",")) {
				CaffeCnn *c = new CaffeCnn;
				c->load(cbase + deployProto,
					   cbase + modelFile,
					   cbase + imageMeanProto,
					   cbase + "data/ilsvrc12/synset_words.txt");
				c->printLayerInfo();
				threadsData[priv]->cnns << c;
			} else {
				QStringList l1 = deployProto.split(",");
				QStringList l2 = modelFile.split(",");
				QStringList l3 = imageMeanProto.split(",");
				assert(l1.size() == l2.size());
				assert(l1.size() == l3.size());
				for (int i = 0; i < l1.size(); i++) {
					CaffeCnn *c = new CaffeCnn;
					assert(c->load(cbase + l1[i],
						   cbase + l2[i],
						   cbase + l3[i],
						   cbase + "data/ilsvrc12/synset_words.txt") == 0);
					c->printLayerInfo();
					threadsData[priv]->cnns << c;
				}
			}
		}
	}
	QList<CaffeCnn *> list = threadsData[priv]->cnns;
	tdlock.unlock();
	return list;
#else
	if (!threadData) {
		threadData = new ThreadData;
		QString cbase = pars.caffeBaseDir;
		QString deployProto = pars.caffeDeployProto;
		QString modelFile = pars.caffeModelFile;
		QString imageMeanProto = pars.caffeImageMeanProto;

		if (!deployProto.contains(",")) {
			CaffeCnn *c = new CaffeCnn;
			c->load(cbase + deployProto,
				   cbase + modelFile,
				   cbase + imageMeanProto,
				   cbase + "data/ilsvrc12/synset_words.txt");
			c->printLayerInfo();
			threadData->cnns << c;
		} else {
			QStringList l1 = deployProto.split(",");
			QStringList l2 = modelFile.split(",");
			QStringList l3 = imageMeanProto.split(",");
			assert(l1.size() == l2.size());
			assert(l1.size() == l3.size());
			for (int i = 0; i < l1.size(); i++) {
				CaffeCnn *c = new CaffeCnn;
				assert(c->load(cbase + l1[i],
					   cbase + l2[i],
					   cbase + l3[i],
					   cbase + "data/ilsvrc12/synset_words.txt") == 0);
				c->printLayerInfo();
				threadData->cnns << c;
			}
		}
	}
	QList<CaffeCnn *> list = threadData->cnns;
	return list;
#endif
}

int ClassificationPipeline::pipelineOutput(BaseLmmPipeline *p, const RawBuffer &buf)
{
	Q_UNUSED(buf);
	if (++finishedCount == expectedFrameCount) {
		ffDebug() << "finished";
		emit pipelineFinished();
	}
	if (pars.debug) {
		static int cnt = 0;
		//if (++cnt % 100 == 0 || cnt > 9140)
			//ffDebug() << buf.constPars()->streamBufferNo << cnt;
		ffDebug() << buf.constPars()->streamBufferNo << cnt++ << p->getOutputQueue(0)->getFps();
	}

	return 0;
}

int SourceLmmElement::processBlocking(int ch)
{
	Q_UNUSED(ch);
	return newOutputBuffer(ch, RawBuffer("video/x-raw-yuv", 1024));
}

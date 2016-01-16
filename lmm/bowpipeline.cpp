#include "bowpipeline.h"
#include "datasetmanager.h"
#include "opencv/cvbuffer.h"
#include "vision/pyramids.h"
#include "vlfeat/vlfeat.h"
#include "common.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>

#include <QDir>
#include <QFileInfo>
#include <QApplication>

#if CV_MAJOR_VERSION > 2
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/nonfree/features2d.hpp>
#endif
#if CV_MAJOR_VERSION > 2
using namespace xfeatures2d;
#endif

#define createEl(_func, _priv) new OpElement<BowPipeline>(this, &BowPipeline::_func, _priv,  #_func)
#define createEl2(_func) new OpSrcElement<BowPipeline>(this, &BowPipeline::_func, #_func)

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

class BowThreadData
{
public:
	BowThreadData()
	{
		py = NULL;
		map =  NULL;
	}

	Pyramids *py;
	VlHomogeneousKernelMap *map;
};

BowPipeline::BowPipeline(QObject *parent) :
	ClassificationPipeline(parent)
{
}

BowPipeline::BowPipeline(const ClassificationPipeline::parameters &params, QObject *parent)
	: ClassificationPipeline(params, parent)
{

}

void BowPipeline::createPipeline()
{
	if (pars.createDict)
		createDictPipeline();
	else if (pars.cl == CLASSIFY_BOW)
		createBOWPipeline();

	if (!pars.createDict) {
		QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(pars.K).arg(pars.ft);
		dict = OpenCV::importMatrix(fname);
		for (int i = 0; i < pars.threads; i++) {
			BowThreadData *data = threadsData[i];
			data->py = new Pyramids;
			data->py->setDict(dict);
			data->map =  vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, pars.gamma, 1, -1, VlHomogeneousKernelMapWindowRectangular);
		}

#if 0
		trainFile = new QFile(QString("%1/svm_train_ftype%2_K%3_step%4_L%5_gamma%6.txt")
							  .arg(pars.dataPath)
							  .arg(pars.ft)
							  .arg(pars.K)
							  .arg(pars.xStep)
							  .arg(pars.L)
							  .arg(pars.gamma)
							  );

		testFile = new QFile(QString("%1/svm_test_ftype%2_K%3_step%4_L%5_gamma%6.txt")
							 .arg(pars.dataPath)
							 .arg(pars.ft)
							 .arg(pars.K)
							 .arg(pars.xStep)
							 .arg(pars.L)
							 .arg(pars.gamma)
							 );
		testFile->open(QIODevice::WriteOnly);
#else
		trainFile = new QFile(QString("train%1.txt").arg(pars.runId));
		trainFile->open(QIODevice::WriteOnly);
		testFile = new QFile(QString("test%1.txt").arg(pars.runId));
		testFile->open(QIODevice::WriteOnly);
#endif
	}
	initTrainTest();
}

void BowPipeline::createDictPipeline()
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
}

void BowPipeline::createBOWPipeline()
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

RawBuffer BowPipeline::createIDs(const RawBuffer &buf, int priv)
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

	return CVBuffer::createNewBuffer(ids, buf);
}

RawBuffer BowPipeline::createImageDescriptor(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &ids = cbuf->getReferenceMat();
	if (pars.L == 0)
		return CVBuffer::createNewBuffer(getBow(ids, dict.rows), buf);

	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString fname = getExportFilename(imname, "kpts");
	vector<KeyPoint> kpts = OpenCV::importKeyPoints(fname);

	int imW = buf.constPars()->videoWidth;
	int imH = buf.constPars()->videoHeight;
	const Mat &pyr = Pyramids::makeSpmFromIds(ids, pars.L, imW, imH, kpts, pars.K);
	return CVBuffer::createNewBuffer(pyr, buf);
}

void BowPipeline::pipelineFinished()
{
	if (pars.createDict) {
		Mat dict = Pyramids::clusterFeatures(dictPool, pars.K);
		QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(pars.K).arg(pars.ft);
		ffDebug() << fname << dict.rows << dict.cols;
		OpenCV::exportMatrix(fname, dict);
	}

	ClassificationPipeline::pipelineFinished();
}

RawBuffer BowPipeline::mapDescriptor(const RawBuffer &buf, int priv)
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

	return CVBuffer::createNewBuffer(m, buf);
}

RawBuffer BowPipeline::addToDictPool(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	/* do not use non-training data for dictionary creating */
	int index = buf.constPars()->streamBufferNo;
	TrainInfo *info = trainInfo[index];
	if (!info->useForTrain)
		return buf;

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

RawBuffer BowPipeline::detectKeypoints(const RawBuffer &buf, int priv)
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
		OpenCV::exportKeyPoints(fname, kpts);
	}

	return CVBuffer::createNewBuffer(kpts, cbuf->getReferenceMat(), buf);
}

std::vector<KeyPoint> BowPipeline::extractKeypoints(const Mat &m)
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

std::vector<KeyPoint> BowPipeline::extractDenseKeypoints(const Mat &m, int step)
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

Mat BowPipeline::computeFeatures(const Mat &m, std::vector<KeyPoint> &keypoints)
{
	Mat features;
	if (pars.ft == FEAT_SIFT) {
		SiftDescriptorExtractor ex;
		ex.compute(m, keypoints, features);
	} else if (pars.ft == FEAT_SURF) {
		SurfDescriptorExtractor ex;
		ex.compute(m, keypoints, features);
	} /*else if (pars.ft == FEAT_CNN) {
		const QList<CaffeCnn *> list = getCurrentThreadCaffe(0);
		QString featureLayer = pars.cnnFeatureLayer;
		return list[0]->extract(m, featureLayer);
	}*/
	return features;
}

RawBuffer BowPipeline::extractFeatures(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-kpts")
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

	return CVBuffer::createNewBuffer(features, buf);
}

void BowPipeline::createThreadData()
{
	for (int i = 0; i < pars.threads; i++)
		threadsData << new BowThreadData;
}

int BowPipeline::checkParameters()
{
	assert(pars.createDict || pars.cl == CLASSIFY_BOW);
	assert(pars.threads);
	assert(pars.K);
	assert(pars.ft == FEAT_SIFT || pars.ft == FEAT_SURF);
	return 0;
}

QString BowPipeline::getExportFilename(const QString &imname, const QString &suffix)
{
	QFileInfo fi(imname);
	if (suffix == "ids")
		return QString("%1/%2_ftype%3_s%5_K%6.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix).arg(pars.xStep).arg(pars.K);
	return QString("%1/%2_ftype%3_s%5.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix).arg(pars.xStep);
}


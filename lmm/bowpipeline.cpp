#include "bowpipeline.h"
#include "datasetmanager.h"
#include "opencv/cvbuffer.h"
#include "vision/pyramids.h"
#include "vlfeat/vlfeat.h"
#include "common.h"
#include "pipelinesettings.h"

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

static Mat getBow(const Mat &ids, int cols, const Mat &ids2, const Mat &corr)
{
	Mat py = Mat::zeros(1, cols, CV_32F);
	for (int j = 0; j < ids.rows; j++) {
		uint id1 = ids.at<uint>(j);
		py.at<float>(id1) += 1;

		uint id2 = ids2.at<uint>(j);
		Mat c = corr.col(id2);
		c /= OpenCV::getL1Norm(c);

#if 0
		Mat mask = Mat::ones(c.rows, c.cols, CV_8U);
		Point minl; Point maxl;
		double min, max;
		minMaxLoc(c, &min, &max, &minl, &maxl, mask);
		int cw1 = maxl.y;
		float ccw1 = max;

		mask.at<uchar>(maxl.y) = 0;
		minMaxLoc(c, &min, &max, &minl, &maxl, mask);
		int cw2 = maxl.y;
		float ccw2 = max;

		py.at<float>(cw1) += (ccw1) / (ccw1 + ccw2);
		py.at<float>(cw2) += (ccw2) / (ccw1 + ccw2);

		//for (int i = 0; i < cols; i++)
			//py.at<float>(i) += c.at<float>(i);
#else
		double min, max;
		Point minl; Point maxl;
		minMaxLoc(c, &min, &max, &minl, &maxl);
		py.at<float>(maxl.y) += 0.9;
		/*Mat ct;
		transpose(c, ct);
		py += ct;*/
		//qDebug() << id1 << id2 << c.at<float>(id1);
#endif
	}
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
	if (pars.createDict && ps->get("features.lib") == "vlfeat")
		createDictPipelineVlFeat();
	else if (pars.createDict)
		createDictPipeline();
	else if (pars.cl == CLASSIFY_BOW && ps->get("features.lib") == "opencv")
		createBOWPipeline();
	else if (pars.cl == CLASSIFY_BOW && ps->get("features.lib") == "vlfeat")
		createBOWPipelineVlFeat();
	else if (pars.cl == CLASSIFY_BOW_CORR)
		createCorrPipeline();

	if (!pars.createDict) {
		QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(pars.K).arg(pars.ft);
		dict = OpenCV::importMatrix(fname);
		mDebug("dict size: %dx%d", dict.rows, dict.cols);
		for (int i = 0; i < pars.threads; i++) {
			BowThreadData *data = threadsData[i];
			data->py = new Pyramids;
			data->py->setDict(dict);
			data->map =  vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, pars.gamma, 1, -1, VlHomogeneousKernelMapWindowRectangular);
		}
		initSvmFiles();
	}
	initTrainTest();

	//if (pars.cl == CLASSIFY_BOW_CORR)
	corrData.confHash = OpenCV::importMatrix(QString("%1/confmat%2.bin").arg(pars.dataPath).arg(3));
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

void BowPipeline::createDictPipelineVlFeat()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	BaseLmmElement *node1 = p1->getPipe(p1->getPipeCount() - 1);
	QList<BaseLmmElement *> join1;
	for (int i = 0; i < pars.threads; i++) {
		p1->insert(node1, createEl(vlDenseSift, i));
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

void BowPipeline::createBOWPipelineVlFeat()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	BaseLmmElement *node1 = p1->getPipe(p1->getPipeCount() - 1);
	QList<BaseLmmElement *> join1;
	for (int i = 0; i < pars.threads; i++) {
		p1->insert(node1, createEl(vlDenseSift, i));
		if (ps->get("encoding.type") == "vq") {
			p1->append(createEl(createIDs, i));
			p1->append(createEl(createImageDescriptor, i));
		} else if (ps->get("encoding.type") == "vlad") {
			p1->append(createEl(createImageVladDescriptor, i));
		} else
			assert(0);
		if (pars.homkermap)
			p1->append(createEl(mapDescriptor, i));
		join1 << p1->getPipe(p1->getPipeCount() - 1);
	}
	p1->appendJoin(createEl(exportForSvm, 0), join1);
	p1->end();
}

void BowPipeline::createCorrPipeline()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(createEl(calcCorr, 0));
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
	const Mat &pyr = Pyramids::makeSpmFromIds(ids, pars.L, imW, imH, kpts, dict.rows);
	return CVBuffer::createNewBuffer(pyr, buf);
}

RawBuffer BowPipeline::createImageDescriptor2(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &ids = cbuf->getReferenceMat();

	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	//QString fname2 = getExportFilename(imname, "ids").replace("ftype0", "ftype1");
	QString fname2 = getExportFilename(imname, "ids").replace("K1000", "K1001");
	Mat ids1 = OpenCV::importMatrix(fname2);

	if (pars.L == 0)
		return CVBuffer::createNewBuffer(getBow(ids, dict.rows, ids1, corrData.confHash), buf);

	QString fname = getExportFilename(imname, "kpts");
	vector<KeyPoint> kpts = OpenCV::importKeyPoints(fname);

	int imW = buf.constPars()->videoWidth;
	int imH = buf.constPars()->videoHeight;
	const Mat &pyr = Pyramids::makeSpmFromIds(ids, pars.L, imW, imH, kpts, dict.rows, ids1, corrData.confHash);
	//const Mat &pyr = Pyramids::makeSpmFromIds(ids, pars.L, imW, imH, kpts, pars.K);
	return CVBuffer::createNewBuffer(pyr, buf);
}

RawBuffer BowPipeline::createImageVladDescriptor(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
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
	QString kname = getExportFilename(imname, "kpts");
	vector<KeyPoint> kpts = OpenCV::importKeyPoints(kname);
	const Mat &fts = cbuf->getReferenceMat();
	int imW = buf.constPars()->videoWidth;
	int imH = buf.constPars()->videoHeight;
	Mat linear = py->makeVladSpm(fts, pars.L, imW, imH, kpts, ps->get("encoding.vlad.knn_count").toInt(),
								 ps->get("encoding.vlad.flags").toInt());

	return CVBuffer::createNewBuffer(linear, buf);
}

RawBuffer BowPipeline::calcCorr(const RawBuffer &buf, int priv)
{
	Q_UNUSED(buf);

	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString fname = getExportFilename(imname, "ids");
	QString fname2;
	Mat ids0, ids1;
#if 1
	if (fname.contains("ftype101")) {
		fname2 = getExportFilename(imname, "ids").replace("ftype1", "ftype0");
		ids1 = OpenCV::importMatrix(fname);
		ids0 = OpenCV::importMatrix(fname2);
	} else {
		fname2 = getExportFilename(imname, "ids").replace("ftype100", "ftype101");
		ids0 = OpenCV::importMatrix(fname);
		ids1 = OpenCV::importMatrix(fname2);
	}
#else
	fname2 = getExportFilename(imname, "ids").replace("K1000", "K1001");
	ids0 = OpenCV::importMatrix(fname);
	ids1 = OpenCV::importMatrix(fname2);
#endif


	int bins = ids0.rows;
	CV_Assert(ids0.rows == ids1.rows);
	if (corrData.confHash.rows == 0)
		corrData.confHash = Mat::zeros(993, 993, CV_32F);
	for (int i = 0; i < bins; ++i) {
		int d1 = ids0.at<int>(i, 0);
		int d2 = ids1.at<int>(i, 0);

		/* corr matrix */
		double min, max;
		corrData.confHash.at<float>(d1, d2)++;
		Point minl; Point maxl;
		minMaxLoc(corrData.confHash.row(d1), &min, &max, &minl, &maxl);
	}
	return buf;
}

RawBuffer BowPipeline::vlDenseSift(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	Mat mat = cbuf->getReferenceMat();
	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString kname = getExportFilename(imname, "kpts");
	QString fname = getExportFilename(imname, "bin");

	vector<KeyPoint> kpts2;
	Mat features;

	/* use existing ones if exists */
	if (pars.useExisting && QFile::exists(kname)) {
		kpts2 = OpenCV::importKeyPoints(kname);
		features = OpenCV::importMatrix(fname);
	} else {
		vector<float> img;
		for (int i = 0; i < mat.rows; ++i)
			for (int j = 0; j < mat.cols; ++j)
				img.push_back(mat.at<unsigned char>(i, j));

#if 0
		VlDsiftFilter *dsift = vl_dsift_new_basic(mat.cols, mat.rows, pars.xStep, 16);
		vl_dsift_set_flat_window(dsift, true);
		vl_dsift_process(dsift, &img[0]);
		const float *desc = vl_dsift_get_descriptors(dsift);
		const VlDsiftKeypoint *kpts = vl_dsift_get_keypoints(dsift);
		int cnt = vl_dsift_get_keypoint_num(dsift);

		features = Mat(cnt, vl_dsift_get_descriptor_size(dsift), CV_32F);
		for (int i = 0; i < cnt; i++) {
			KeyPoint pt;
			pt.pt.x = kpts[i].x;
			pt.pt.y = kpts[i].y;
			kpts2.push_back(pt);
			for (int j = 0; j < features.cols; j++)
				features.at<float>(i, j) = qMin(desc[i * features.cols + j] * 512.0, 255.0);
		}
		vl_dsift_delete(dsift);
#else
		VlDsiftFilter *dsift = vl_dsift_new_basic(mat.cols, mat.rows, pars.xStep, 16);
		vl_dsift_set_flat_window(dsift, true);
		features = Mat(0, vl_dsift_get_descriptor_size(dsift), CV_32F);

		QStringList l = ps->get("features.step_sizes").toString().split(",");
		QList<int> binSizes;
		foreach (QString s, l)
			binSizes << s.toInt();
		QList<double> scales;
		double magnif = 3;
		for (int i = 0; i < binSizes.size(); i++)
			scales << binSizes[i] / magnif;
		for (int i = 0; i < binSizes.size(); i++) {
			double sigma = sqrt(pow(scales[i], 2) - 0.25);
			//smooth float array image
			float* img_vec_smooth = (float*)malloc(mat.rows * mat.cols * sizeof(float));
			vl_imsmooth_f(img_vec_smooth, mat.cols, &img[0], mat.cols, mat.rows, mat.cols, sigma, sigma);
			vl_dsift_process(dsift, img_vec_smooth);

			const float *desc = vl_dsift_get_descriptors(dsift);
			const VlDsiftKeypoint *kpts = vl_dsift_get_keypoints(dsift);
			int cnt = vl_dsift_get_keypoint_num(dsift);

			Mat fts = Mat(cnt, vl_dsift_get_descriptor_size(dsift), CV_32F);
			for (int k = 0; k < cnt; k++) {
				KeyPoint pt;
				pt.pt.x = kpts[k].x;
				pt.pt.y = kpts[k].y;
				kpts2.push_back(pt);
				for (int j = 0; j < fts.cols; j++)
					fts.at<float>(k, j) = qMin(desc[k * fts.cols + j] * 512.0, 255.0);
				if (ps->isEqual("features.vlfeat.sift.normalization", "l2"))
					fts.row(k) /= OpenCV::getL2Norm(fts.row(k));
				else if (ps->isEqual("features.vlfeat.sift.normalization", "l1"))
					fts.row(k) /= OpenCV::getL1Norm(fts.row(k));
			}
			features.push_back(fts);
			free(img_vec_smooth);
		}

		vl_dsift_delete(dsift);
#endif

		OpenCV::exportKeyPoints(kname, kpts2);
		if (pars.exportData)
			OpenCV::exportMatrix(fname, features);
	}

	assert(features.rows);
	return CVBuffer::createNewBuffer(features, buf);
}

void BowPipeline::pipelineFinished()
{
	if (pars.createDict) {
		if (pars.K) {
			Mat dict = Pyramids::clusterFeatures(dictPool, pars.K);
			QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(pars.K).arg(pars.ft);
			ffDebug() << fname << dict.rows << dict.cols;
			OpenCV::exportMatrix(fname, dict);
		} else {
			QList<int> list;
			list << 100;
			list << 200;
			list << 300;
			list << 400;
			list << 500;
			list << 600;
			list << 700;
			list << 800;
			list << 900;
			list << 1000;
			list << 1500;
			list << 2000;
			list << 2500;
			list << 3000;
			list << 5000;
			list << 10000;
			list << 15000;
			list << 20000;
			list << 25000;
			for (int i = 0; i < list.size(); i++) {
				Mat dict = Pyramids::clusterFeatures(dictPool, list[i]);
				QString fname = QString("%1/dict_ftype%3_K%2.bin").arg(pars.dataPath).arg(list[i]).arg(pars.ft);
				ffDebug() << fname << dict.rows << dict.cols;
				OpenCV::exportMatrix(fname, dict);
			}
		}
	} else if (pars.cl == CLASSIFY_BOW_CORR)
		OpenCV::exportMatrix(QString("%1/confmat%2.bin").arg(pars.dataPath).arg(pars.runId), corrData.confHash);

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
	} else
		assert(0);
	return keypoints;
}

std::vector<KeyPoint> BowPipeline::extractDenseKeypoints(const Mat &m, int step)
{
	vector<KeyPoint> keypoints;
	if (pars.ft >= FEAT_CNN) {
		assert(0);
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
	} else
		assert(0);
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
	assert(pars.createDict || pars.cl == CLASSIFY_BOW || pars.cl == CLASSIFY_BOW_CORR);
	assert(pars.threads);
	assert(pars.K);
	return 0;
}

QString BowPipeline::getExportFilename(const QString &imname, const QString &suffix)
{
	QFileInfo fi(imname);
	if (suffix == "ids")
		return QString("%1/%2_ftype%3_s%5_K%6.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix).arg(pars.xStep).arg(pars.K);
	if (pars.ft < FEAT_CNN)
		return QString("%1/%2_ftype%3_s%5.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix).arg(pars.xStep);
	return QString("%1/%2_ftype%3.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix);
}


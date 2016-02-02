#include "cnnpipeline.h"
#include "datasetmanager.h"
#include "opencv/cvbuffer.h"
#include "common.h"
#include "caffe/caffecnn.h"
#include "buffercloner.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>

#include <QDir>
#include <QFileInfo>
#include <QApplication>

#define createEl(_func, _priv) new OpElement<CnnPipeline>(this, &CnnPipeline::_func, _priv,  #_func)
#define createEl2(_func) new OpSrcElement<CnnPipeline>(this, &CnnPipeline::_func, #_func)

class CnnThreadData
{
public:
	CnnThreadData()
	{
	}

	QList<CaffeCnn *> cnns;
	QStringList inetCats;
};

static __thread CnnThreadData *threadData = NULL;

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

CnnPipeline::CnnPipeline(QObject *parent) :
	ClassificationPipeline(parent)
{
}

CnnPipeline::CnnPipeline(const ClassificationPipeline::parameters &params, QObject *parent)
	: ClassificationPipeline(params, parent)
{

}

const RawBuffer CnnPipeline::readNextLMDBImageFeature()
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

void CnnPipeline::createPipeline()
{
	if (pars.cl == CLASSIFY_CNN_SVM)
		createCNNFSVMPipeline();
	else if (pars.cl == CLASSIFY_CNN_BOW)
		createExtractionPipeline();
	else if (pars.cl == CLASSIFY_CNN_MULTIFTS)
		createCNNMultiFts();

	initSvmFiles();
	initTrainTest();
}

const QList<CaffeCnn *> CnnPipeline::getCurrentThreadCaffe(int priv)
{
	if (!threadData) {
		threadData = new CnnThreadData;
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
}

void CnnPipeline::createExtractionPipeline()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(createEl(cnnExtract, 0));
	p1->append(createEl(exportFeature, 0));
	p1->end();
}

void CnnPipeline::createThreadData()
{
}

int CnnPipeline::checkParameters()
{
	assert(!pars.caffeBaseDir.isEmpty());
	assert(!pars.caffeDeployProto.isEmpty());
	assert(!pars.caffeModelFile.isEmpty());
	return 0;
}

QString CnnPipeline::getExportFilename(const QString &imname, const QString &suffix)
{
	QFileInfo fi(imname);
	return QString("%1/%2_ftype%3.%4").arg(fi.absolutePath()).arg(fi.baseName()).arg(pars.ft).arg(suffix);
}

void CnnPipeline::createCNNFSVMPipeline()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(createEl(cnnExtract, 0));
	//if (pars.homkermap)
		//p1->append(createEl(mapDescriptor, 0));
	p1->append(createEl(exportForSvm, 0));
	p1->end();
}

void CnnPipeline::createCNNBOWPipeline()
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

void CnnPipeline::createCNNMultiFts()
{
	BaseLmmPipeline *p1 = addPipeline();
	p1->append(createEl2(readNextImage));
	p1->append(createEl(cnnExtractMultiFts, 0));
	p1->append(createEl(exportForSvmMulti, 0));
	p1->end();
}

RawBuffer CnnPipeline::cnnExtract(const RawBuffer &buf, int priv)
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

		return CVBuffer::createNewBuffer(merged, buf);
	}


	/* use single model */
	CaffeCnn *c = list[pars.targetCaffeModel];
	if (!c)
		return buf;
	if (featureLayer.contains("&"))
		return CVBuffer::createNewBuffer(c->extractLinear(img, featureLayer.split("&")), buf);
	if (pars.spatialSize == 0) {
		const Mat &m = c->getLayerDimensions(featureLayer);
		pars.spatialSize = m.at<float>(0, 0);
	}
	return CVBuffer::createNewBuffer(c->extractLinear(img, featureLayer), buf);
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

RawBuffer CnnPipeline::cnnExtractMultiFts(const RawBuffer &buf, int priv)
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
			return CVBuffer::createNewBuffer(OpenCV::merge(channels), buf);
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
			return CVBuffer::createNewBuffer(OpenCV::merge(channels), buf);
		}
		assert(0);
	}

	/* use single model */
	CaffeCnn *c = list[pars.targetCaffeModel];
	vector<Mat> fts = extractCnnFeatures(img, pars.cnnFeatureLayer, pars.cnnFeatureLayerType, c, info->useForTest ? pars.dataAug : 0);
	if (pars.featureMergingMethod == 0) //concat
		return CVBuffer::createNewBuffer(OpenCV::merge(fts), buf);
	else if (pars.featureMergingMethod == 1) //sum pooling
		return CVBuffer::createNewBuffer(OpenCV::merge(fts, OpenCV::MM_SUM), buf);
	else if (pars.featureMergingMethod == 2) //max pooling
		return CVBuffer::createNewBuffer(OpenCV::merge(fts, OpenCV::MM_MAX), buf);
	return CVBuffer::createNewBuffer(fts, buf);
}

RawBuffer CnnPipeline::exportFeature(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &desc = cbuf->getReferenceMat();
	QString imname = QString::fromUtf8(buf.constPars()->metaData);
	QString fname = getExportFilename(imname, "bin");
	assert(pars.spatialSize);
	int fsize = desc.cols / pars.spatialSize / pars.spatialSize;
	Mat fts(pars.spatialSize * pars.spatialSize, fsize, CV_32F);
	const float *bdata = (const float *)desc.data;
	for (int i = 0; i < desc.cols; i++) {
		int row = i % fts.rows;
		int col = i / fts.rows;
		fts.at<float>(row, col) = bdata[i];
	}
	OpenCV::exportMatrix(fname, fts);

	vector<KeyPoint> keypoints;
	for (int i = 0; i < pars.spatialSize; i++) {
		for (int j = 0; j < pars.spatialSize; j++) {
			KeyPoint kpt;
			int step = 1;//.cols / pars.spatialSize;
			kpt.pt.x = j * step;
			kpt.pt.y = i * step;
			keypoints.push_back(kpt);
		}
	}
	OpenCV::exportKeyPoints(fname.replace(".bin", ".kpts"), keypoints);

	return buf;
}

RawBuffer CnnPipeline::createMulti(const RawBuffer &buf, int priv)
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
	RawBuffer bufout = CVBuffer::createNewBuffer(mergeBuffersOrdered(m, features[1], bowSize), buf);
	bufout.pars()->duration = priv;
	return bufout;
}


#ifndef CLASSIFICATIONPIPELINE_H
#define CLASSIFICATIONPIPELINE_H

#ifdef HAVE_LMM

#include <lmm/pipeline/pipelinemanager.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <errno.h>

#include <QSemaphore>
#include <QStringList>

class QFile;
class TrainInfo;
class ThreadData;
class DatasetManager;

#define objstr(_x, _id) QString("%1%2").arg(cl##_x).arg(_id)

template <class T>
class OpElement : public BaseLmmElement
{
public:
	typedef RawBuffer (T::*elementOp)(const RawBuffer &, int);
	OpElement(T *parent, elementOp op, int priv, QString objName = "BaseLmmElement")
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
		this->priv = priv;
		setObjectName(QString("%1%2").arg(objName).arg(priv));
	}
	virtual int processBuffer(const RawBuffer &buf)
	{
		RawBuffer buf2 = (enc->*mfunc)(buf, priv);
		return newOutputBuffer(0, buf2);
	}

private:
	T *enc;
	elementOp mfunc;
	int priv;
};

template <class T>
class OpSrcElement : public BaseLmmElement
{
public:
	typedef const RawBuffer (T::*elementOp)();
	OpSrcElement(T *parent, elementOp op, QString objName = "BaseLmmElement")
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
		setObjectName(objName);
	}
	virtual int processBuffer(const RawBuffer &) { return 0; }
	int processBlocking(int ch)
	{
		RawBuffer buf = (enc->*mfunc)();
		return newOutputBuffer(ch, buf);
	}

private:
	T *enc;
	elementOp mfunc;
};

class SourceLmmElement : public BaseLmmElement
{
	Q_OBJECT
public:
	SourceLmmElement(QObject *parent = NULL)
		: BaseLmmElement(parent)
	{}

	int processBlocking(int ch);
	virtual int processBuffer(const RawBuffer &) { return 0; }
};

class ClassificationPipeline : public PipelineManager
{
	Q_OBJECT
public:
	enum ftype {
		FEAT_SIFT,
		FEAT_SURF,
	};
	enum cltype {
		CLASSIFY_BOW,
		CLASSIFY_CNN,
	};

	struct parameters {
		ftype ft;
		int xStep;
		int yStep;
		bool exportData;
		int threads;
		int K;
		int dictSubSample;
		bool useExisting;
		bool createDict;
		QString dataPath;
		double gamma;
		int trainCount;
		int testCount;
		int L;
		quint64 maxMemBytes;
		int maxFeaturesPerImage;
		bool useExistingTrainSet;
		QString datasetPath;
		QString datasetName;
		cltype cl;
		int imFlags;
	};
	parameters pars;

	explicit ClassificationPipeline(QObject *parent = 0);
	explicit ClassificationPipeline(const struct parameters &params, QObject *parent = 0);

	virtual const RawBuffer readNextImage();
	virtual RawBuffer detectKeypoints(const RawBuffer &buf, int priv);
	virtual RawBuffer extractFeatures(const RawBuffer &buf, int priv);
	virtual RawBuffer addToDictPool(const RawBuffer &buf, int priv);
	virtual RawBuffer createIDs(const RawBuffer &buf, int priv);
	virtual RawBuffer createImageDescriptor(const RawBuffer &buf, int priv);
	virtual RawBuffer mapDescriptor(const RawBuffer &buf, int priv);
	virtual RawBuffer exportForSvm(const RawBuffer &buf, int priv);
	virtual RawBuffer cnnClassify(const RawBuffer &buf, int priv);
signals:

protected slots:
	void pipelineFinished();
protected:
	void init();
	void createDictPipeline();
	void createBOWPipeline();
	void createCNNPipeline();
	void createTrainTestSplit(const QString &trainSetFileName);
	QString getExportFilename(const QString &imname, const QString &suffix);
	std::vector<cv::KeyPoint> extractDenseKeypoints(const cv::Mat &m, int step);
	std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat &m);
	cv::Mat computeFeatures(const cv::Mat &m, std::vector<cv::KeyPoint> &keypoints);

	virtual int pipelineOutput(BaseLmmPipeline *, const RawBuffer &buf);

	DatasetManager *dm;

	QStringList images;
	QMutex dplock;
	cv::Mat dictPool;
	cv::Mat dict;
	QMutex exlock;
	QFile *trainFile;
	QFile *testFile;
	int imageCount;
	int finishedCount;
	int expectedFrameCount;

	QMutex tdlock;
	QList<ThreadData *> threadsData;
	QList<TrainInfo *> trainInfo;
};

#endif

#endif // CLASSIFICATIONPIPELINE_H

#ifndef CLASSIFICATIONPIPELINE_H
#define CLASSIFICATIONPIPELINE_H

#ifdef HAVE_LMM

#include <lmm/pipeline/pipelinemanager.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <errno.h>

class QFile;
class TrainInfo;
class ThreadData;
class DatasetManager;

template <class T>
class OpElement : public BaseLmmElement
{
public:
	typedef RawBuffer (T::*elementOp)(const RawBuffer &, int);
	OpElement(T *parent, elementOp op, int priv)
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
		this->priv = priv;
	}
	virtual int processBuffer(const RawBuffer &buf)
	{
		RawBuffer buf2 = (enc->*mfunc)(buf, priv);
		if (buf2.isEOF())
			return -ENOENT;
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
	OpSrcElement(T *parent, elementOp op)
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
	}
	virtual int processBuffer(const RawBuffer &) { return 0; }
	int processBlocking(int ch)
	{
		RawBuffer buf = (enc->*mfunc)();
		if (buf.isEOF()) {
			newOutputBuffer(ch, buf);
			return -ENODATA;
		}
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
	};
	parameters pars;

	explicit ClassificationPipeline(QObject *parent = 0);

	virtual const RawBuffer readNextImage();
	virtual RawBuffer detectKeypoints(const RawBuffer &buf, int priv);
	virtual RawBuffer extractFeatures(const RawBuffer &buf, int priv);
	virtual RawBuffer addToDictPool(const RawBuffer &buf, int priv);
	virtual RawBuffer createIDs(const RawBuffer &buf, int priv);
	virtual RawBuffer createImageDescriptor(const RawBuffer &buf, int priv);
	virtual RawBuffer mapDescriptor(const RawBuffer &buf, int priv);
	virtual RawBuffer exportForSvm(const RawBuffer &buf, int priv);
signals:

protected slots:
	void pipelineFinished();
protected:
	QString getExportFilename(const QString &imname, const QString &suffix);
	std::vector<cv::KeyPoint> extractDenseKeypoints(const cv::Mat &m, int step);
	std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat &m);
	cv::Mat computeFeatures(const cv::Mat &m, std::vector<cv::KeyPoint> &keypoints);

	DatasetManager *dm;

	QStringList images;
	QMutex dplock;
	cv::Mat dictPool;
	cv::Mat dict;
	QMutex exlock;
	QFile *trainFile;
	QFile *testFile;

	QMutex tdlock;
	QList<ThreadData *> threadsData;
	QList<TrainInfo *> trainInfo;
};

#endif

#endif // CLASSIFICATIONPIPELINE_H

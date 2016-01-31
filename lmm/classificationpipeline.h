#ifndef CLASSIFICATIONPIPELINE_H
#define CLASSIFICATIONPIPELINE_H

#ifdef HAVE_LMM

#include <lmm/debug.h>
#include <lmm/pipeline/pipelinemanager.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <errno.h>
#include <unistd.h>

#include <QSemaphore>
#include <QStringList>

#include "lmmelements.h"

class QFile;
class DatasetManager;

//#define objstr(_x, _id) QString("%1%2").arg(cl##_x).arg(_id)

class ClassificationPipeline : public PipelineManager
{
	Q_OBJECT
public:
	enum ftype {
		FEAT_SIFT,
		FEAT_SURF,
		FEAT_CNN,
	};
	enum cltype {
		CLASSIFY_BOW,
		CLASSIFY_CNN_SVM = 3,
		CLASSIFY_CNN_BOW,
		CLASSIFY_CNN_MULTIFTS,
	};

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
		QString fileListTxt;
		QString trainListTxt;
		QString testListTxt;
		QString lmdbFeaturePath;
		QString cnnFeatureLayer;
		int debug;
		int spatialSize;
		bool homkermap;
		QString caffeBaseDir;
		QString caffeDeployProto;
		QString caffeModelFile;
		QString caffeImageMeanProto;
		int targetCaffeModel;
		int featureMergingMethod;
		int dataAug;
		int rotationDegree;
		QString cnnFeatureLayerType;
		int runId;
	};
	parameters pars;

	explicit ClassificationPipeline(QObject *parent = 0);
	explicit ClassificationPipeline(const struct parameters &params, QObject *parent = 0);
	void init();

	/* buffer operations */
	virtual const RawBuffer readNextImage();
	virtual RawBuffer exportForSvm(const RawBuffer &buf, int priv);
	virtual RawBuffer exportForSvmMulti(const RawBuffer &buf, int priv);
	virtual RawBuffer mergeFeatures(const RawBuffer &buf, int priv);
	virtual RawBuffer debugBuffer(const RawBuffer &buf, int priv);
signals:

protected slots:
	virtual void pipelineFinished();
protected:
	virtual void createPipeline() = 0;
	virtual void createThreadData() = 0;
	virtual int checkParameters() = 0;
	virtual QString getExportFilename(const QString &imname, const QString &suffix) = 0;
	virtual void createDatasetInfo();
	virtual void initTrainTest();
	virtual void initSvmFiles();

	void augmentTrainData(QList<TrainInfo *> &trainInfo, TrainInfo *info, int dataAug);
	void createTrainTestSplit(const QString &trainSetFileName);

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
	QList<QFile *> trainFilesMulti;
	QList<QFile *> testFilesMulti;

	QMutex tdlock;
	QList<TrainInfo *> trainInfo;
	int datasetIndex;
};

#endif

#endif // CLASSIFICATIONPIPELINE_H

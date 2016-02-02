#ifndef BOWPIPELINE_H
#define BOWPIPELINE_H

#include <lmm/classificationpipeline.h>

class BowThreadData;

class BowPipeline : public ClassificationPipeline
{
	Q_OBJECT
public:
	explicit BowPipeline(QObject *parent = 0);
	explicit BowPipeline(const struct parameters &params, QObject *parent = 0);

	virtual RawBuffer detectKeypoints(const RawBuffer &buf, int priv);
	virtual RawBuffer extractFeatures(const RawBuffer &buf, int priv);
	virtual RawBuffer addToDictPool(const RawBuffer &buf, int priv);
	virtual RawBuffer createIDs(const RawBuffer &buf, int priv);
	virtual RawBuffer createImageDescriptor(const RawBuffer &buf, int priv);
	virtual RawBuffer createImageDescriptor2(const RawBuffer &buf, int priv);
	virtual RawBuffer calcCorr(const RawBuffer &buf, int priv);

signals:

protected slots:
	virtual void pipelineFinished();
protected:
	virtual void createPipeline();
	virtual void createThreadData();
	virtual int checkParameters();
	virtual QString getExportFilename(const QString &imname, const QString &suffix);

	void createDictPipeline();
	void createBOWPipeline();
	void createCorrPipeline();
	std::vector<cv::KeyPoint> extractDenseKeypoints(const cv::Mat &m, int step);
	std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat &m);
	cv::Mat computeFeatures(const cv::Mat &m, std::vector<cv::KeyPoint> &keypoints);
	virtual RawBuffer mapDescriptor(const RawBuffer &buf, int priv);

	QList<BowThreadData *> threadsData;

	class CorrData {
	public:
		CorrData()
		{
			cnt = 0;
			sum = 0;
			createCorr = false;
		}

		int cnt;
		double sum;
		bool createCorr;
		cv::Mat confHash;
	};
	CorrData corrData;
};

#endif // BOWPIPELINE_H

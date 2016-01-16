#ifndef CNNPIPELINE_H
#define CNNPIPELINE_H

#include <lmm/classificationpipeline.h>

class CaffeCnn;
class CnnThreadData;

class CnnPipeline : public ClassificationPipeline
{
	Q_OBJECT
public:
	explicit CnnPipeline(QObject *parent = 0);
	explicit CnnPipeline(const struct parameters &params, QObject *parent = 0);

	virtual const RawBuffer readNextLMDBImageFeature();
	RawBuffer createMulti(const RawBuffer &buf, int priv);
	virtual RawBuffer cnnExtract(const RawBuffer &buf, int priv);
	virtual RawBuffer cnnExtractMultiFts(const RawBuffer &buf, int priv);
signals:

public slots:
protected:
	virtual void createPipeline();
	virtual void createThreadData();
	virtual int checkParameters();
	virtual QString getExportFilename(const QString &imname, const QString &suffix);

	const QList<CaffeCnn *> getCurrentThreadCaffe(int priv);

	void createCNNFC7Pipeline();
	void createCNNFSVMPipeline();
	void createCNNBOWPipeline();
	void createCNNMultiFts();

	QList<CnnThreadData *> threadsData;
};

#endif // CNNPIPELINE_H

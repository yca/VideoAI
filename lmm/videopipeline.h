#ifndef VIDEOPIPELINE_H
#define VIDEOPIPELINE_H

#ifdef HAVE_LMM

#include <lmm/pipeline/pipelinemanager.h>

class CaffeCnn;
class BaseLmmDemux;
class QtVideoOutput;
class X11VideoOutput;
class PipelineSettings;

class VideoPipeline : public PipelineManager
{
	Q_OBJECT
public:
	explicit VideoPipeline(QObject *parent = 0);

	void init(PipelineSettings *s);

	RawBuffer cnnExtract(const RawBuffer &buf, int priv);
signals:

protected slots:
	virtual void pipelineFinished();

protected:
	virtual int pipelineOutput(BaseLmmPipeline *p, const RawBuffer &buf);
	const QList<CaffeCnn *> getCurrentThreadCaffe(const QString &cbase, const QString &deployProto, const QString &modelFile, const QString imageMeanProto);

	PipelineSettings *ps;
	int expectedFrameCount;
	int finishedCount;
	BaseLmmDemux *demux;
	X11VideoOutput *vout;
	QtVideoOutput *vout2;
};

#endif

#endif // VIDEOPIPELINE_H

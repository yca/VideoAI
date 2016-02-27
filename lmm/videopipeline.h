#ifndef VIDEOPIPELINE_H
#define VIDEOPIPELINE_H

#ifdef HAVE_LMM

#include <lmm/pipeline/pipelinemanager.h>

class PipelineSettings;

class VideoPipeline : public PipelineManager
{
	Q_OBJECT
public:
	explicit VideoPipeline(QObject *parent = 0);

	void init(PipelineSettings *s);
signals:

protected slots:
	virtual void pipelineFinished();

protected:
	virtual int pipelineOutput(BaseLmmPipeline *, const RawBuffer &buf);

	PipelineSettings *ps;
};

#endif

#endif // VIDEOPIPELINE_H

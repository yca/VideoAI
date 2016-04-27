#ifndef VLADPIPELINE_H
#define VLADPIPELINE_H

#include <lmm/pipeline/pipelinemanager.h>

class VladPipeline : public PipelineManager
{
	Q_OBJECT
public:
	explicit VladPipeline(QObject *parent = 0);

protected slots:
	void pipelineFinished();

protected:
	virtual int pipelineOutput(BaseLmmPipeline *p, const RawBuffer &buf);

private:
	int finishedCount;
	int expectedFrameCount;
};

#endif // VLADPIPELINE_H

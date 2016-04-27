#include "vladpipeline.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>

#include <QApplication>

VladPipeline::VladPipeline(QObject *parent) :
	PipelineManager(parent)
{
}

void VladPipeline::pipelineFinished()
{
	stop();
	ffDebug() << "quitting";
	QApplication::exit();
}

int VladPipeline::pipelineOutput(BaseLmmPipeline *p, const RawBuffer &buf)
{
	Q_UNUSED(buf);
	if (++finishedCount == expectedFrameCount) {
		ffDebug() << "finished";
		emit pipelineFinished();
	}
	if (1) {
		static int cnt = 0;
		//if (++cnt % 100 == 0 || cnt > 9140)
			//ffDebug() << buf.constPars()->streamBufferNo << cnt;
		ffDebug() << buf.constPars()->streamBufferNo << ++cnt << p->getOutputQueue(0)->getFps() << expectedFrameCount;
	}

	return 0;
}

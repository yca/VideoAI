#include "videopipeline.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>
#include <lmm/ffmpeg/baselmmdemux.h>
#include <lmm/ffmpeg/ffmpegdecoder.h>

VideoPipeline::VideoPipeline(QObject *parent) :
	PipelineManager(parent)
{
}

void VideoPipeline::init(PipelineSettings *s)
{
	ps = s;

	BaseLmmPipeline *p1 = addPipeline();
	BaseLmmDemux *demux = new BaseLmmDemux;
	demux->setSource("/home/amenmd/Downloads/00006_3dfail.avi");
	p1->append(demux);
	FFmpegDecoder *dec = new FFmpegDecoder;
	p1->append(dec);
	p1->end();
}

void VideoPipeline::pipelineFinished()
{
	ffDebug() << "finished";
}

int VideoPipeline::pipelineOutput(BaseLmmPipeline *, const RawBuffer &buf)
{
	ffDebug() << buf.size();
	return 0;
}

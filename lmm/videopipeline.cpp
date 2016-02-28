#include "videopipeline.h"
#include "lmmelements.h"
#include "opencv/opencv.h"
#include "qtvideooutput.h"
#include "caffe/caffecnn.h"
#include "pipelinesettings.h"
#include "opencv/cvbuffer.h"

#include <lmm/debug.h>
#include <lmm/x11videooutput.h>
#include <lmm/baselmmpipeline.h>
#include <lmm/ffmpeg/baselmmdemux.h>
#include <lmm/ffmpeg/ffmpegdecoder.h>

#include <QApplication>

#define createEl(_func, _priv) new OpElement<VideoPipeline>(this, &VideoPipeline::_func, _priv,  #_func)

class VideoThreadData
{
public:
	VideoThreadData()
	{
	}

	QList<CaffeCnn *> cnns;
	QStringList inetCats;
};

static __thread VideoThreadData *threadData = NULL;

VideoPipeline::VideoPipeline(QObject *parent) :
	PipelineManager(parent)
{
	finishedCount = expectedFrameCount = 0;
}

void VideoPipeline::init(PipelineSettings *s)
{
	ps = s;

	BaseLmmPipeline *p1 = addPipeline();
	demux = new BaseLmmDemux;
	//demux->setSource("/home/amenmd/Downloads/00006_3dfail.avi");
	//demux->setSource(QString("%1/%2").arg("/home/amenmd/Videos/").arg("Arsenal vs Barcelona 0-2 - 2016 Highlights and Full goals Champions League 23_02_2016 HD-mPu85sMxu6M.mp4"));
	demux->setSource(QString("%1/%2").arg("/home/amenmd/Videos/").arg("Paris Saint-Germain PSG vs Chelsea 2-1 ALL GOALS & Highlights Champions League 2016-KCTBf0-UIac.mp4"));
	p1->append(demux);
	FFmpegDecoder *dec = new FFmpegDecoder;
	/* caffe uses bgr channel order */
	dec->setRgbOutput(false);
	dec->setBgrOutput(true);
	p1->append(dec);
	p1->append(createEl(cnnExtract, 0));
	p1->end();

	vout = new X11VideoOutput;
	vout->start();
	vout2 = new QtVideoOutput;
	vout2->show();
	vout2->setInteractive(true);
}

static RawBuffer getFeatureMapsBuffer(const vector<Mat> &maps, const RawBuffer &buf)
{
	assert(maps.size());
	int rows = 16;
	int cols = maps.size() / rows;
#if 1
	RawBuffer out("qt/qimage", maps[0].rows * maps[0].cols * maps.size() * 3);
	out.pars()->videoWidth = maps[0].cols * cols;
	out.pars()->videoHeight = maps[0].rows * rows;
	out.pars()->avPixelFormat = 2;
	out.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	memset(out.data(), 0, out.size());
	QImage im((uchar *)out.data(), cols * maps[0].cols, rows * maps[0].rows, QImage::Format_RGB888);
	out.pars()->metaData = QByteArray((const char *)buf.constData(), buf.size());
	out.pars()->captureTime = buf.constPars()->videoWidth * 65536 + buf.constPars()->videoHeight;
#else
	QByteArray ba(maps[0].rows * maps[0].cols * maps.size() * 3, 0);
	QImage im((uchar *)ba.data(), cols * maps[0].cols, rows * maps[0].rows, QImage::Format_RGB888);
	buf.pars()->metaData = ba;
#endif

	for (uint i = 0; i < maps.size(); i++) {
		const Mat &m = maps[i];

		double min, max;
		minMaxLoc(m, &min, &max);
		m -= min;
		if (max != 0)
			m /= max;

		int mode = 1;
		/*double sum = cv::sum(m)[0];
		double avg = sum / m.rows / m.cols;
		if (avg < 0.3)
			mode = 0;
		else
			mode = 255;*/

		int c = i % cols;
		int r = i / cols;
		int offX = c * m.cols;
		int offY = r * m.rows;
		for (int j = 0; j < m.cols; j++) {
			for (int k = 0; k < m.rows; k++) {
				int val = m.at<float>(j, k) * 255;
				if (mode == 0)
					val = 0;
				else if (mode == 255)
					val = 255;
				im.setPixel(offX + j, offY + k, qRgb(val, val, val));
			}
		}
	}

	return out;
}

RawBuffer VideoPipeline::cnnExtract(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	const Mat &img = OpenCV::loadImage((void *)buf.constData(), buf.constPars()->videoWidth, buf.constPars()->videoHeight);
	QString cbase = PipelineSettings::getInstance()->get("data.caffe.base_dir").toString();
	QString deployProto = PipelineSettings::getInstance()->get("data.caffe.deploy_proto").toString();
	QString modelFile = PipelineSettings::getInstance()->get("data.caffe.weights_file").toString();
	QString imageMeanProto = PipelineSettings::getInstance()->get("data.caffe.image_mean").toString();
	const QList<CaffeCnn *> list = getCurrentThreadCaffe(cbase, deployProto, modelFile, imageMeanProto);
	CaffeCnn *caffe = list[0];

	caffe->forwardImage(img);

	//const vector<Mat> &maps = caffe->getFeatureMaps("conv5_4");
	//return getFeatureMapsBuffer(maps, buf);

	Mat m = Mat::zeros(1, 1000, CV_32F);
	m.at<float>(0, 981) = 1;
	caffe->setBlobDiff("fc8", m);
	caffe->backward();

	Mat smap = caffe->getSaliencyMap();
	CVBuffer cbuf = CVBuffer::createNewBuffer(smap, buf);
	cbuf.pars()->metaData = QByteArray((const char *)buf.constData(), buf.size());
	cbuf.pars()->captureTime = buf.constPars()->videoWidth * 65536 + buf.constPars()->videoHeight;
	cbuf.pars()->avPixelFormat = 2;
	return cbuf;
}

void VideoPipeline::pipelineFinished()
{
	stop();
	if (!vout2->isInteractive()) {
		ffDebug() << "quitting";
		QApplication::exit();
	}
}

int VideoPipeline::pipelineOutput(BaseLmmPipeline *p, const RawBuffer &buf)
{
	Q_UNUSED(p);
	Q_UNUSED(buf);
	if (buf.constPars()->avPixelFormat == 2 || buf.constPars()->avPixelFormat == 3) {
		vout2->showFrame(buf);
	} else {
		vout->addBuffer(0, buf);
		vout->processBlocking(0);
	}
	expectedFrameCount = demux->getDemuxedCount();
	if (++finishedCount == expectedFrameCount || buf.constPars()->streamBufferNo + 1 == expectedFrameCount) {
		ffDebug() << "finished";
		emit pipelineFinished();
	}
	ffDebug() << buf.constPars()->streamBufferNo << p->getOutputQueue(0)->getFps() << expectedFrameCount << finishedCount;
	return 0;
}

const QList<CaffeCnn *> VideoPipeline::getCurrentThreadCaffe(const QString &cbase, const QString &deployProto, const QString &modelFile, const QString imageMeanProto)
{
	if (!threadData) {
		threadData = new VideoThreadData;

		if (!deployProto.contains(",")) {
			CaffeCnn *c = new CaffeCnn;
			c->load(cbase + deployProto,
				   cbase + modelFile,
				   cbase + imageMeanProto,
				   cbase + "data/ilsvrc12/synset_words.txt");
			c->printLayerInfo();
			threadData->cnns << c;
		} else {
			QStringList l1 = deployProto.split(",");
			QStringList l2 = modelFile.split(",");
			QStringList l3 = imageMeanProto.split(",");
			assert(l1.size() == l2.size());
			assert(l1.size() == l3.size());
			for (int i = 0; i < l1.size(); i++) {
				CaffeCnn *c = new CaffeCnn;
				assert(c->load(cbase + l1[i],
					   cbase + l2[i],
					   cbase + l3[i],
					   cbase + "data/ilsvrc12/synset_words.txt") == 0);
				c->printLayerInfo();
				threadData->cnns << c;
			}
		}
	}
	QList<CaffeCnn *> list = threadData->cnns;
	return list;
}

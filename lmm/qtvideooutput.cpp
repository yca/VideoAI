#include "qtvideooutput.h"
#include "debug.h"
#include "opencv/opencv.h"
#include "opencv/cvbuffer.h"

#include <QPainter>
#include <QKeyEvent>
#include <QMutexLocker>

QtVideoOutput::QtVideoOutput(QWidget *parent) :
	QWidget(parent)
{
	interval = 33;
	timer.start(interval);
	connect(&timer, SIGNAL(timeout()), SLOT(timeout()));
	interactive = false;
}

void QtVideoOutput::showFrame(const RawBuffer &buf)
{
	lock.lock();
	frames << buf;
	lock.unlock();
}

void QtVideoOutput::advance(int step)
{
	stepsLeft = step;
	timer.start(interval);
}

void QtVideoOutput::setInteractive(bool enable)
{
	interactive = enable;
	if (interactive)
		timer.stop();
	else
		timer.start(interval);
}

void QtVideoOutput::timeout()
{
	QMutexLocker ml(&lock);

	if (!frames.size())
		return;

	const RawBuffer &buf = frames.takeFirst();
	/*QRect g = geometry();
	g.setWidth(buf.constPars()->videoWidth);
	g.setHeight(buf.constPars()->videoHeight);
	setGeometry(g);*/
	currentFrame = buf;

	if (interactive) {
		if (--stepsLeft <= 0) {
			timer.stop();
			stepsLeft = 0;
		}
	}

	repaint();
}

void QtVideoOutput::keyPressEvent(QKeyEvent *kev)
{
	if (kev->key() == Qt::Key_Space)
		advance(1);
	else if (kev->key() == Qt::Key_Right)
		advance(10);
}

void QtVideoOutput::paintEvent(QPaintEvent *)
{
	QPainter p(this);
	if (currentFrame.getMimeType() == "application/qtimage") {
		QImage im = QImage((const uchar *)currentFrame.constData(), currentFrame.constPars()->videoWidth, currentFrame.constPars()->videoHeight, QImage::Format_RGB888);
		if (currentFrame.constPars()->metaData.size()) {
			int w = currentFrame.constPars()->captureTime >> 16;
			int h = currentFrame.constPars()->captureTime & 0xffff;
			QImage im2 = QImage((const uchar *)currentFrame.constPars()->metaData.constData(), w, h, QImage::Format_RGB888);
			p.drawImage(0, 0, im);
			p.drawImage(im.width(), 0, im2);
		} else
			p.drawImage(rect(), im);
	} else if (currentFrame.getMimeType() == "application/cv-mat") {
		CVBuffer *cbuf = (CVBuffer *)&currentFrame;
		int w = currentFrame.constPars()->captureTime >> 16;
		int h = currentFrame.constPars()->captureTime & 0xffff;
		QImage im = OpenCV::toQImage(cbuf->getReferenceMat());
		QImage im2 = QImage((const uchar *)currentFrame.constPars()->metaData.constData(), w, h, QImage::Format_RGB888);
		p.drawImage(0, 0, im);
		p.drawImage(im.width(), 0, im2);
	} else
		ffDebug() << "un-supported mimetype" << currentFrame.getMimeType();
	currentFrame = RawBuffer();
}

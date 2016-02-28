#ifndef QTVIDEOOUTPUT_H
#define QTVIDEOOUTPUT_H

#include <QTimer>
#include <QMutex>
#include <QImage>
#include <QWidget>

#include <lmm/rawbuffer.h>

class QtVideoOutput : public QWidget
{
	Q_OBJECT
public:
	explicit QtVideoOutput(QWidget *parent = 0);

	void showFrame(const RawBuffer &buf);
	void advance(int step = 1);
	void setInteractive(bool enable);
	bool isInteractive() { return interactive; }
signals:

protected slots:
	void timeout();
	void keyPressEvent(QKeyEvent *);
protected:
	void paintEvent(QPaintEvent *);

	QList<RawBuffer> frames;
	QMutex lock;
	QTimer timer;
	RawBuffer currentFrame;
	bool interactive;
	int interval;
	int stepsLeft;
};

#endif // QTVIDEOOUTPUT_H

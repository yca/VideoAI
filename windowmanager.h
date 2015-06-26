#ifndef WINDOWMANAGER_H
#define WINDOWMANAGER_H

#include <QObject>

class ImageWidget;

class WindowManager : public QObject
{
	Q_OBJECT
public:
	explicit WindowManager(QObject *parent = 0);
signals:

public slots:
	void createImageWindow(int rows = 1, int cols = 1);
	void setCurrentImageWindow(int curr);
	ImageWidget * getCurrentImageWindow();

protected:
	QList<ImageWidget *> imageWidgets;
	int current;
};

#endif // WINDOWMANAGER_H

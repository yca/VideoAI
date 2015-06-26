#include "windowmanager.h"

#include "widgets/imagewidget.h"

WindowManager::WindowManager(QObject *parent) :
	QObject(parent)
{
}

void WindowManager::createImageWindow(int rows, int cols)
{
	ImageWidget *iw = new ImageWidget(rows, cols);
	iw->show();
	imageWidgets << iw;
	current = imageWidgets.size() - 1;
}

void WindowManager::setCurrentImageWindow(int curr)
{
	if (curr >= 0 && curr < imageWidgets.size())
		current = curr;
}

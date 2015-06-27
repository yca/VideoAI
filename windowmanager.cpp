#include "windowmanager.h"
#include "scriptmanager.h"

#include "widgets/imagewidget.h"

WindowManager::WindowManager(QObject *parent) :
	QObject(parent)
{
}

void WindowManager::createImageWindow(int rows, int cols)
{
	if (rows == 0 || cols == 0)
		return;
	ImageWidget *iw = new ImageWidget(rows, cols);
	iw->show();
	imageWidgets << iw;
	current = imageWidgets.size() - 1;
	ScriptManager::instance()->setCurrentWindow(-1);
}

void WindowManager::setCurrentImageWindow(int curr)
{
	if (curr < 0)
		curr = imageWidgets.size() + curr;
	if (curr >= 0 && curr < imageWidgets.size())
		current = curr;
}

ImageWidget *WindowManager::getCurrentImageWindow()
{
	return imageWidgets[current];
}

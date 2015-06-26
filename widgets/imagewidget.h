#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QImage>
#include <QWidget>

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class ImageWidgetGridItem;

class ImageWidget : public QWidget
{
	Q_OBJECT
public:
	explicit ImageWidget(QWidget *parent = 0);
	explicit ImageWidget(int rows, int cols, QWidget *parent = 0);
	~ImageWidget();
signals:

public slots:
	void setCurrentCell(int row, int col);
	void showImageMat(const Mat &image);
	void showImage(const QString &filename);

protected:
	void paintEvent(QPaintEvent *);

	QList<ImageWidgetGridItem *> grid;
	QImage im;
	Mat imgSource;
	int curr;
	int gridRows;
	int gridCols;
};

#endif // IMAGEWIDGET_H

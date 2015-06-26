#include "imagewidget.h"
#include "debug.h"

#include <QPainter>

/*
pyr.setDict("/home/amenmd/myfs/tasks/hilal_tez/work/ox_complete/oxbuild_images_512.dict"); spm = pyr.makeSpm(dm.getImage(0), 0); spmi = pyr.makeHistImage(spm); iw.showImageMat(spmi);
*/

class ImageWidgetGridItem {
public:
	enum ScaleMode {
		SCALE_NONE,
		SCALE_FILL,
		SCALE_WIDTH,
		SCALE_HEIGHT,
	};
	void show(const QString &filename)
	{
		imgSource = Mat();
		im = QImage(filename);
		if (im.isNull())
			qDebug("Null image from %s", qPrintable(filename));
	}
	void showMat(const Mat &image)
	{
		im = QImage((const uchar *)image.data, image.cols,  image.rows, QImage::Format_RGB888);
		imgSource = image;
	}
	void draw(QPainter *p, QRect r)
	{
		p->drawImage(r, im);
	}
	void draw(QPainter *p)
	{
		p->drawImage(rect, im);
	}

	QImage im;
	Mat imgSource;
	ScaleMode mode;
	QRect rect;
};

ImageWidget::ImageWidget(QWidget *parent) :
	QWidget(parent)
{
	ImageWidgetGridItem *item = new ImageWidgetGridItem;
	item->mode = ImageWidgetGridItem::SCALE_NONE;
	grid << item;
	curr = 0;
	gridCols = 1;
	gridRows = 1;
}

ImageWidget::ImageWidget(int rows, int cols, QWidget *parent)
	: QWidget(parent)
{
	for (int i = 0; i < rows * cols; i++) {
		ImageWidgetGridItem *item = new ImageWidgetGridItem;
		item->mode = ImageWidgetGridItem::SCALE_NONE;
		grid << item;
	}
	gridRows = rows;
	gridCols = cols;
	curr = 0;
}

ImageWidget::~ImageWidget()
{
	qDeleteAll(grid);
	grid.clear();
}

void ImageWidget::setCurrent(int row, int col)
{
	curr = row * gridCols + col;
}

void ImageWidget::showImageMat(const Mat &image)
{
	mDebug("size: %dx%d", image.cols, image.rows);
	grid[curr]->showMat(image);
	update();
}

void ImageWidget::showImage(const QString &filename)
{
	grid[curr]->show(filename);
	update();
}

void ImageWidget::paintEvent(QPaintEvent *)
{
	QPainter p(this);
	QRect r = rect();
	int dx = r.width() / gridCols;
	int dy = r.height() / gridRows;
	for (int i = 0; i < gridCols; i++) {
		for (int j = 0; j < gridRows; j++) {
			ImageWidgetGridItem *cell = grid[i + j * gridCols];
			cell->rect = QRect(dx * i, j * dy, dx, dy);
			cell->draw(&p);
		}
	}
}

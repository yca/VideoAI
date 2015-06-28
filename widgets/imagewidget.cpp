#include "imagewidget.h"
#include "debug.h"

#include <QPainter>
#include <QFontMetrics>

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

		/* draw OSD */
		if (osd.size()) {
			p->setFont(QFont("Arial", 12));
			p->setBrush(QBrush(Qt::yellow));
			p->setPen(QPen(Qt::yellow));
			/*QFontMetrics fm(p->font(), p->device());
			QRectF rf;
			rf.setX(r.x());
			rf.setY(r.y());
			rf.setHeight(fm.height() * osd.size());
			int max = 0;
			foreach (QString l, osd)
				if (fm.width(l) > max)
					max = fm.width(l);
			rf.setWidth(max);*/
			p->drawText(r, osd.join("\n"));
		}
	}
	void draw(QPainter *p)
	{
		draw(p, rect);
	}

	QImage im;
	Mat imgSource;
	ScaleMode mode;
	QRect rect;
	QStringList osd;
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

void ImageWidget::addText(const QString &text)
{
	grid[curr]->osd << text;
}

void ImageWidget::clearText()
{
	grid[curr]->osd.clear();
}

void ImageWidget::setCurrentCell(int row, int col)
{
	curr = row * gridCols + col;
}

void ImageWidget::showImageMat(const Mat &image)
{
	mInfo("size: %dx%d", image.cols, image.rows);
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

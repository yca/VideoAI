#include "imagewidget.h"
#include "debug.h"

#include <QPainter>
#include <QFontMetrics>

static double interpolate( double val, double y0, double x0, double y1, double x1 ) {
	return (val-x0)*(y1-y0)/(x1-x0) + y0;
}

static double base( double val ) {
	if ( val <= -0.75 ) return 0;
	else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
	else if ( val <= 0.25 ) return 1.0;
	else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
	else return 0.0;
}

static double red( double gray ) {
	return base( gray - 0.5 );
}
static double green( double gray ) {
	return base( gray );
}
static double blue( double gray ) {
	return base( gray + 0.5 );
}

static QImage createIndexed(int w, int h)
{
	QImage im(w, h, QImage::Format_Indexed8);
	QVector<QRgb> jet;
	for (int i = 0; i < 256; i++) {
		double ival = i / 128.0 - 1;
		jet << qRgb(red(ival) * 255, green(ival) * 255, blue(ival) * 255);
	}
	im.setColorTable(jet);
	return im;
}

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
		if (image.type() == CV_32F) {
			double min, max;
			minMaxLoc(image, &min, &max);
			Mat image2 = image.clone();
			image2 -= min;
			image2 /= max;
			im = createIndexed(image.cols, image.rows);
			for (int i = 0; i < image2.cols; i++) {
				for (int j = 0; j < image2.rows; j++) {
					im.setPixel(i, j, image2.at<float>(j, i) * 255);
				}
			}
		} else
			im = QImage((const uchar *)image.data, image.cols,  image.rows, QImage::Format_RGB888);
		imgSource = image;
	}
	void draw(QPainter *p, QRect r)
	{
		p->setPen(Qt::red);
		p->setBrush(QBrush(Qt::red));
		p->drawRect(r);
		p->drawImage(QRect(r.left() + 1, r.top() + 1, r.right() - 1, r.bottom() - 1), im);

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

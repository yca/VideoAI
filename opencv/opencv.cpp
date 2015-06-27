#include "opencv.h"
#include "debug.h"

#include <QFile>
#include <QDataStream>

#include <errno.h>

#include <opencv2/opencv.hpp>

OpenCV::OpenCV(QObject *parent) :
	QObject(parent)
{
}

OpenCV::OpenCV(const Mat &m, QObject *parent)
	: QObject(parent)
{
	refMat = m;
}

OpenCV::~OpenCV()
{
	qDebug("deleting my object");
}

Mat OpenCV::blendImages(const Mat &im1, const Mat &im2, double alpha, double beta)
{
	Mat dst;
	addWeighted(im1, alpha, im2, beta, 0.0, dst);
	return dst;
}

void OpenCV::saveImage(const QString &filename, const Mat &m)
{
	imwrite(qPrintable(filename), m);
}

Mat OpenCV::loadImage(const QString &filename)
{
	return imread(qPrintable(filename), IMREAD_GRAYSCALE);
}

void OpenCV::printMatInfo(const Mat &m)
{
	qDebug("rows: %d cols: %d", m.rows, m.cols);
}

int OpenCV::exportMatrix(QString filename, const Mat &m)
{
	if (m.channels() != 1) {
		fDebug("unsupported channel count %d", m.channels());
		return -EINVAL;
	}
	if (m.dims >= 3) {
		fDebug("unsupported dimension count %d", m.dims);
		return -EINVAL;
	}
	QFile f(filename);
	f.open(QIODevice::WriteOnly);
	QDataStream out(&f);
	out.setByteOrder(QDataStream::LittleEndian);
	out << (int)1;
	out << m.dims;
	out << m.rows;
	out << m.cols;
	out << m.type();
	out.writeBytes((const char *)m.data, m.rows * m.cols * m.elemSize());
	f.close();
	return 0;
}

Mat OpenCV::importMatrix(QString filename)
{
	QFile f(filename);
	f.open(QIODevice::ReadOnly);
	QDataStream in(&f);
	in.setByteOrder(QDataStream::LittleEndian);
	int size; in >> size;
	int dims; in >> dims;
	int rows; in >> rows;
	int cols; in >> cols;
	int type; in >> type;
	Mat m(rows, cols, type);
	char *data;
	uint l;
	in.readBytes(data, l);
	if (l != rows * cols * m.elemSize()) {
		fDebug("error in data length: exptected=%d got=%d", uint(rows * cols * m.elemSize()), l);
	}
	memcpy(m.data, data, l);
	delete []data;
	return m;
}

int OpenCV::exportMatrixTxt(const QString &filename, const Mat &m)
{
	QFile f(filename);
	if (!f.open(QIODevice::WriteOnly))
		return -EPERM;
	for (int i = 0; i < m.rows; i++) {
		QString line;
		for (int j = 0; j < m.cols - 1; j++) {
			line.append(QString("%1,").arg(m.at<float>(i, j)));
		}
		line.append(QString("%1\n").arg(m.at<float>(i, m.cols - 1)));
		f.write(line.toUtf8());
	}
	f.close();
	return 0;
}

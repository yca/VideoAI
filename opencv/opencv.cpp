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

float OpenCV::getL1Norm(const Mat &m1)
{
	return norm(m1, NORM_L1);
}

float OpenCV::getL2Norm(const Mat &m1)
{
	return norm(m1, NORM_L2);
}

float OpenCV::getL1Norm(const Mat &m1, const Mat &m2)
{
	return normL1_((float *)m1.data, (float *)m2.data, m1.cols * m1.rows);
}

float OpenCV::getL2Norm(const Mat &m1, const Mat &m2)
{
	return normL2Sqr_((float *)m1.data, (float *)m2.data, m1.cols * m1.rows);
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
	out << (int)2;
	out << m.dims;
	out << m.rows;
	out << m.cols;
	out << m.type();
	quint64 len = (quint64)m.rows * m.cols * m.elemSize();
	out << len;
	uint bs = 1024 * 1024 * 1024;
	quint64 off = 0;
	while(len > 0) {
		int res = out.writeRawData((const char *)m.data + off, len < bs ? len : bs);
		len -= res;
		off += res;
	}
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
	if (size == 1) {
		char *data;
		uint l;
		in.readBytes(data, l);
		if (l != rows * cols * m.elemSize()) {
			fDebug("error in data length: exptected=%d got=%d", uint(rows * cols * m.elemSize()), l);
		}
		memcpy(m.data, data, l);
		delete []data;
	} else {
		quint64 len; in >> len;
		uint bs = 1024 * 1024 * 1024;
		quint64 off = 0;
		while(len > 0) {
			int res = in.readRawData((char *)m.data + off, len < bs ? len : bs);
			len -= res;
			off += res;
		}
	}
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

Mat OpenCV::subSampleRandom(const Mat &m, int count)
{
	if (count > m.rows)
		return m;
	Mat sub(0, m.cols, CV_32F);
	QList<int> l;
	for (int i = 0; i < count; i++)
		l << i;
	srand(time(NULL));
	while (l.size()) {
		int ind = rand() % l.size();
		sub.push_back(m.row(l.takeAt(ind)));
	}
	return sub;
}

Mat OpenCV::createRandomized(int start, int size)
{
	Mat m(size, 1, CV_32F);
	QList<int> list;
	for (int i = start; i < start + size; i++)
		list << i;
	srand(time(NULL));
	for (int i = 0; i < m.rows; i++)
		m.at<float>(i) = list.takeAt(rand() % list.size());
	return m;
}

bool OpenCV::matContains(const Mat &m, int val)
{
	for (int i = 0; i < m.rows; i++)
		if (m.at<float>(i) == val)
			return true;
	return false;
}

QString OpenCV::toSvmLine(const Mat &spm, int label)
{
	QString line;
	for (int i = 0; i < spm.rows; i++) {
		line += QString("%1 ").arg(label);
		for (int j = 0; j < spm.cols; j++) {
			line += QString("%1:%2 ").arg(j + 1).arg(spm.at<float>(i, j));
		}
		line.append("\n");
	}
	return line;
}

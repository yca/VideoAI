#include "opencv.h"
#include "debug.h"
#include "common.h"

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

Mat OpenCV::merge(const vector<Mat> &vec, MergeMethod mm)
{
	if (mm == MM_CONCAT) {
		int cols = 0;
		for (uint i = 0; i < vec.size(); ++i)
			cols += vec[i].cols;
		Mat m = Mat::zeros(1, cols, CV_32F);
		int off = 0;
		for (uint i = 0; i < vec.size(); ++i)
			for (int j = 0; j < vec[i].cols; j++)
				m.at<float>(0, off++) = vec[i].at<float>(0, j);
		return m;
	} else if (mm == MM_SUM) {
		Mat m = Mat::zeros(1, vec[0].cols, CV_32F);
		for (uint i = 0; i < vec.size(); i++)
			for (int j = 0; j < m.cols; j++)
				m.at<float>(0, j) += vec[i].at<float>(0, j) / vec.size();
		return m;
	}
	Mat m = Mat::zeros(1, vec[0].cols, CV_32F);
	for (uint i = 0; i < vec.size(); i++)
		for (int j = 0; j < m.cols; j++)
			m.at<float>(0, j) = qMax<float>(vec[i].at<float>(0, j), m.at<float>(0, j));
	return m;
}

Mat OpenCV::rotate(const Mat &img, float angle)
{
	if (angle == 90) {
		Mat dst;
		transpose(img, dst);
		flip(dst, dst, 1);
		return dst;
	} else if (angle == -90 || angle == 270) {
		Mat dst;
		transpose(img, dst);
		flip(dst, dst, 0);
		return dst;
	}
	int len = std::max(img.cols, img.rows);
	cv::Point2f pt(len/2., len/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
	Mat dst;
	cv::warpAffine(img, dst, r, cv::Size(img.rows, img.cols));
	return dst;
}

Mat OpenCV::gammaCorrection(const Mat &img, float gamma)
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
	Mat dst = img.clone();
	const int channels = dst.channels();
	switch (channels) {
		case 1: {
			MatIterator_<uchar> it, end;
			for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
				*it = lut[(*it)];
			break;
		}
		case 3: {
			MatIterator_<Vec3b> it, end;
			for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++) {
				(*it)[0] = lut[((*it)[0])];
				(*it)[1] = lut[((*it)[1])];
				(*it)[2] = lut[((*it)[2])];
			}
			break;
		}
	}
	return dst;
}

float OpenCV::getL1Norm(const Mat &m1)
{
	return norm(m1, NORM_L1);
}

float OpenCV::getL2Norm(const Mat &m1)
{
	return norm(m1, NORM_L2);
}

float OpenCV::getHammingNorm(const Mat &m1)
{
	return norm(m1, NORM_HAMMING);
}

float OpenCV::getL1Norm(const Mat &m1, const Mat &m2)
{
	return normL1_((float *)m1.data, (float *)m2.data, m1.cols * m1.rows);
}

float OpenCV::getL2Norm(const Mat &m1, const Mat &m2)
{
	return normL2Sqr_((float *)m1.data, (float *)m2.data, m1.cols * m1.rows);
}

float OpenCV::getHammingNorm(const Mat &m1, const Mat &m2)
{
	return 0;//normHamming((float *)m1.data, (float *)m2.data, m1.cols * m1.rows);
}

float OpenCV::getCosineNorm(const Mat &m1, const Mat &m2)
{
	double ab = m1.dot(m2);
	double aa = m1.dot(m1);
	double bb = m2.dot(m2);
	return -ab / sqrt(aa * bb);
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

Mat OpenCV::loadImage(const QString &filename, int flags)
{
	return imread(qPrintable(filename), flags);
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
	if (!f.open(QIODevice::ReadOnly))
		return Mat();
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
			assert(res > 0);
			len -= res;
			off += res;
		}
	}
	return m;
}

Mat OpenCV::importMatrixTxt(QString filename)
{
	Mat data;
	const QStringList lines = Common::importText(filename);
	foreach (const QString &line, lines) {
		QStringList vals = line.split(",");
		if (data.cols == 0)
			data = Mat(0, vals.size(), CV_32F);
		Mat m(1, vals.size(), CV_32F);
		for (int i = 0; i < vals.size(); i++)
			m.at<float>(i) = vals[i].toFloat();
		if (m.cols == data.cols)
			data.push_back(m);
	}
	return data;
}

void OpenCV::exportKeyPoints(QString filename, const vector<KeyPoint> &keypoints)
{
	QFile f(filename);
	f.open(QIODevice::WriteOnly);
	QDataStream out(&f);
	out.setByteOrder(QDataStream::LittleEndian);
	out << 1;
	const vector<KeyPoint> &v = keypoints;
	out << (int)v.size();
	for (uint j = 0; j < v.size(); j++) {
		KeyPoint k = v.at(j);
		out << k.pt.x;
		out << k.pt.y;
		out << k.size;
		out << k.angle;
		out << k.response;
		out << k.octave;
		out << k.class_id;
	}
	f.close();
}

void OpenCV::exportVector2(QString filename, const vector<vector<int> > &v)
{
	QFile f(filename);
	f.open(QIODevice::WriteOnly);
	QDataStream out(&f);
	out.setByteOrder(QDataStream::LittleEndian);
	out << 1;
	out << (int)v.size();
	for (uint j = 0; j < v.size(); j++) {
		const vector<int> v2 = v.at(j);
		out << (int)v2.size();
		for (uint i = 0; i < v2.size(); i++)
			out << (int)v2[i];
	}
	f.close();
}

void OpenCV::exportVector2f(QString filename, const vector<vector<float> > &v)
{
	QFile f(filename);
	f.open(QIODevice::WriteOnly);
	QDataStream out(&f);
	out.setByteOrder(QDataStream::LittleEndian);
	out.setFloatingPointPrecision(QDataStream::DoublePrecision);
	out << 2; //1:single 2:double
	out << (int)v.size();
	for (uint j = 0; j < v.size(); j++) {
		const vector<float> v2 = v.at(j);
		out << (int)v2.size();
		for (uint i = 0; i < v2.size(); i++)
			out << (float)v2[i];
	}
	f.close();
}

void OpenCV::exportVector2d(QString filename, const vector<vector<double> > &v)
{
	QFile f(filename);
	f.open(QIODevice::WriteOnly);
	QDataStream out(&f);
	out.setByteOrder(QDataStream::LittleEndian);
	out.setFloatingPointPrecision(QDataStream::DoublePrecision);
	out << 2; //1:single 2:double
	out << (int)v.size();
	for (uint j = 0; j < v.size(); j++) {
		const vector<double> v2 = v.at(j);
		out << (int)v2.size();
		for (uint i = 0; i < v2.size(); i++)
			out << (double)v2[i];
	}
	f.close();
}

const vector<vector<int> > OpenCV::importVector2(QString filename)
{
	vector<vector<int> > v;
	QFile f(filename);
	if (!f.open(QIODevice::ReadOnly))
		return v;
	QDataStream in(&f);
	in.setByteOrder(QDataStream::LittleEndian);
	int size; in >> size;
	in >> size; // real size
	for (int i = 0; i < size; i++) {
		vector<int> v2;
		int vsize; in >> vsize;
		for (int j = 0; j < vsize; j++) {
			int k; in >> k;
			v2.push_back(k);
		}
		v.push_back(v2);
	}
	return v;
}

const vector<vector<float> > OpenCV::importVector2f(QString filename)
{
	vector<vector<float> > v;
	QFile f(filename);
	if (!f.open(QIODevice::ReadOnly))
		return v;
	QDataStream in(&f);
	in.setByteOrder(QDataStream::LittleEndian);
	in.setFloatingPointPrecision(QDataStream::SinglePrecision);
	int size; in >> size;
	if (size > 1) {
		ffDebug() << "reading in double precision";
		in.setFloatingPointPrecision(QDataStream::DoublePrecision);
	}
	in >> size; // real size
	for (int i = 0; i < size; i++) {
		vector<float> v2;
		int vsize; in >> vsize;
		for (int j = 0; j < vsize; j++) {
			float k; in >> k;
			v2.push_back(k);
		}
		v.push_back(v2);
	}
	return v;
}

const vector<vector<double> > OpenCV::importVector2d(QString filename)
{
	vector<vector<double> > v;
	QFile f(filename);
	if (!f.open(QIODevice::ReadOnly))
		return v;
	QDataStream in(&f);
	in.setByteOrder(QDataStream::LittleEndian);
	in.setFloatingPointPrecision(QDataStream::SinglePrecision);
	int size; in >> size;
	if (size > 1) {
		ffDebug() << "reading in double precision";
		in.setFloatingPointPrecision(QDataStream::DoublePrecision);
	}
	in >> size; // real size
	for (int i = 0; i < size; i++) {
		vector<double> v2;
		int vsize; in >> vsize;
		for (int j = 0; j < vsize; j++) {
			double k; in >> k;
			v2.push_back(k);
		}
		v.push_back(v2);
	}
	return v;
}

const vector<KeyPoint> OpenCV::importKeyPoints(QString filename)
{
	vector<KeyPoint> v;
	QFile f(filename);
	if (!f.open(QIODevice::ReadOnly))
		return v;
	QDataStream in(&f);
	in.setByteOrder(QDataStream::LittleEndian);
	int size; in >> size;
	//for (int i = 0; i < size; i++) {
		int vsize; in >> vsize;
		for (int j = 0; j < vsize; j++) {
			KeyPoint k;
			in >> k.pt.x;
			in >> k.pt.y;
			in >> k.size;
			in >> k.angle;
			in >> k.response;
			in >> k.octave;
			in >> k.class_id;
			v.push_back(k);
		}
		return v;
	//}
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
	if (count >= m.rows)
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

Mat OpenCV::createRandomized(int start, int max, int size)
{
	Mat m(size, 1, CV_32F);
	QList<int> list;
	for (int i = start; i < start + size; i++)
		list << i;
	srand(time(NULL));
	for (int i = 0; i < m.rows; i++)
		m.at<float>(i) = list.takeAt(rand() % max);
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

Mat OpenCV::histIntersect(const Mat &m1, const Mat &m2)
{
	Mat m = Mat::zeros(m1.rows, m1.cols, CV_32F);
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
			m.at<float>(i, j) = qMin(m1.at<float>(i, j), m2.at<float>(i, j));
	return m;
}

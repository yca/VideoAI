#ifndef OPENCV_H
#define OPENCV_H

#include <QObject>

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class OpenCV : public QObject
{
	Q_OBJECT
public:
	explicit OpenCV(QObject *parent = 0);
	explicit OpenCV(const Mat &m, QObject *parent = 0);
	~OpenCV();

	Mat getRefMat() { return refMat; }
signals:

public slots:
	static float getL1Norm(const Mat &m1);
	static float getL2Norm(const Mat &m1);
	static float getL1Norm(const Mat &m1, const Mat &m2);
	static float getL2Norm(const Mat &m1, const Mat &m2);
	static Mat blendImages(const Mat &im1, const Mat &im2, double alpha = 0.5, double beta = 0.5);
	static void saveImage(const QString &filename, const Mat &m);
	static Mat loadImage(const QString &filename);
	static void printMatInfo(const Mat &m);
	static int exportMatrix(QString filename, const Mat &m);
	static Mat importMatrix(QString filename);
	static Mat importMatrixTxt(QString filename);
	static int exportMatrixTxt(const QString &filename, const Mat &m);
	static Mat subSampleRandom(const Mat &m, int count);
	static Mat createRandomized(int start, int size);
	static bool matContains(const Mat &m, int val);
	static QString toSvmLine(const Mat &spm, int label);
protected:
	Mat refMat;
};

#endif // OPENCV_H

#ifndef OPENCV_H
#define OPENCV_H

#include <QObject>

#include <opencv2/opencv.hpp>
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

	enum MergeMethod {
		MM_CONCAT,
		MM_SUM,
		MM_MAX,
	};
	static Mat merge(const vector<Mat> &vec, MergeMethod mm = MM_CONCAT);
	static Mat rotate(const Mat &img, float angle);
	static Mat gammaCorrection(const Mat &img, float gamma);
	static vector<Mat> subImages(const Mat &img, int rows, int cols);
signals:

public slots:
	static float getL1Norm(const Mat &m1);
	static float getL2Norm(const Mat &m1);
	static float getHammingNorm(const Mat &m1);
	static float getL1Norm(const Mat &m1, const Mat &m2);
	static float getL2Norm(const Mat &m1, const Mat &m2);
	static float getHammingNorm(const Mat &m1, const Mat &m2);
	static float getCosineNorm(const Mat &m1, const Mat &m2);
	static Mat blendImages(const Mat &im1, const Mat &im2, double alpha = 0.5, double beta = 0.5);
	static void saveImage(const QString &filename, const Mat &m);
	static Mat loadImage(const QString &filename, int flags = IMREAD_GRAYSCALE);
	static void printMatInfo(const Mat &m);
	static int exportMatrix(QString filename, const Mat &m);
	static Mat importMatrix(QString filename);
	static Mat importMatrixTxt(QString filename);
	static void exportKeyPoints(QString filename, const vector<KeyPoint> &keypoints);
	static void exportVector2(QString filename, const vector<vector<int> > &v);
	static void exportVector2f(QString filename, const vector<vector<float> > &v);
	static void exportVector2d(QString filename, const vector<vector<double> > &v);
	static const vector<vector<int> > importVector2(QString filename);
	static const vector<vector<float> > importVector2f(QString filename);
	static const vector<vector<double> > importVector2d(QString filename);
	static const vector<KeyPoint> importKeyPoints(QString filename);
	static int exportMatrixTxt(const QString &filename, const Mat &m);
	static Mat subSampleRandom(const Mat &m, int count);
	static Mat createRandomized(int start, int size);
	static Mat createRandomized(int start, int max, int size);
	static bool matContains(const Mat &m, int val);
	static QString toSvmLine(const Mat &spm, int label);
	static Mat histIntersect(const Mat &m1, const Mat &m2);
protected:
	Mat refMat;
};

#endif // OPENCV_H

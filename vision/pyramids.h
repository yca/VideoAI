#ifndef PYRAMIDS_H
#define PYRAMIDS_H

#include <QObject>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

class Pyramids : public QObject
{
	Q_OBJECT
public:
	explicit Pyramids(QObject *parent = 0);

signals:

public slots:
	static vector<KeyPoint> extractDenseKeypoints(const Mat &m, int step);
	static vector<KeyPoint> extractKeypoints(const Mat &m);
	static Mat computeFeatures(const Mat &m, vector<KeyPoint> &keypoints);
	static Mat clusterFeatures(const Mat &features, int clusterCount);

	void createDictionary(const QStringList &images, int clusterCount);
	void createDictionary(int clusterCount);
	void computeImageFeatures(const QStringList &images, int samplesPerImage = -1);
	Mat calculatePyramids(const QStringList &images, int L, int step);
	Mat calculatePyramidsH(const QStringList &images, int L, int H, int step);

	Mat makeSpm(const QString &filename, int L, int step = -1);
	Mat makeSpmH(const QString &filename, int L, int H, int step = -1);
	Mat makeSpmFromMat(const Mat &im, int L, int step = -1);
	Mat makeSpmFromMatH(const Mat &im, int L, int H, int step = -1);
	Mat makeHistImage(const Mat &hist, int scale = 0, int foreColor = Qt::white, int backColor = Qt::black);
	void setDict(const QString &filename);
	void setDict(const Mat &codewords);
	Mat getDict();
	Mat getImageFeatures();
	void setImageFeatures(const Mat &features);

protected:
	virtual Mat extractFeatures(const Mat &im, vector<KeyPoint> &keypoints, int step);

	static int histCount(int L);
	static int findPointContribution(int x, int y, int level, int width, int height);
	static int findPointContributionHor(int x, int binsH, int width);
	static Mat findPointContributions(int x, int y, int level, int width, int height);

	Mat dict;
	Mat imageFeatures;
};

#endif // PYRAMIDS_H

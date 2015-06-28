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
	void computeImageFeatures(const QStringList &images);

	Mat makeSpm(const QString &filename, int L);
	Mat makeSpmFromMat(const Mat &im, int L);
	Mat makeHistImage(const Mat &hist, int scale = 0, int foreColor = Qt::white, int backColor = Qt::black);
	void setDict(const QString &filename);
	void setDict(const Mat &codewords);
	Mat getDict();
	Mat getImageFeatures();

protected:
	Mat dict;
	Mat imageFeatures;
};

#endif // PYRAMIDS_H

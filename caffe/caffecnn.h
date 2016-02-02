#ifndef CAFFECNN_H
#define CAFFECNN_H

#include <QMutex>
#include <QObject>
#include <QStringList>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class CaffeCnnPriv;

class CaffeCnn : public QObject
{
	Q_OBJECT
public:
	explicit CaffeCnn(QObject *parent = 0);

	int load(const QString &modelFile, const QString &trainedFile, const QString &meanFile, const QString &labelFile);
	int load(const QString &lmdbFolder);
	int setMean(const QString &meanFile);
	QStringList classify(const QString &filename, int N = 5);
	QStringList classify(const Mat &img, int N = 5);
	Mat readNextFeature(QString &key);
	Mat extract(const Mat &img, const QString &layerName);
	Mat extractLinear(const Mat &img, const QString &layerName);
	Mat extractLinear(const Mat &img, const QStringList &layers);
	vector<Mat> extractMulti(const Mat &img, const QStringList &layers, const QStringList &featureFlags);
	vector<Mat> extractMulti(const Mat &img, const QStringList &layers, const QStringList &featureFlags, int augFlags);
	vector<Mat> getFeatureMaps(const QString &layerName);
	int forwardImage(const QString &filename);

	void printLayerInfo();
	Mat getLayerDimensions(const QString &layer);
	static void printLayerInfo(const QString &modelFile, bool printEmpty = false);
	void printLayerInfo(const QStringList &layers);
	QStringList getBlobbedLayerNames();

signals:

public slots:

protected:
	void forwardImage(const Mat &img);
	CaffeCnnPriv *p;
	static QMutex lock;
};

#endif // CAFFECNN_H

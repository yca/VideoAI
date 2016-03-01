#ifndef DARKNET_H
#define DARKNET_H

#include <QString>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
//using namespace std;

class DarkNetPriv;

class Darknet
{
public:
	Darknet();

	int init(const QString &darkNetRoot);
	int loadNetwork(const QString &cfg, const QString &weights);
	void predict(const QString &filename, float thresh = 0.2);
	Mat predict(const Mat &ori, float thresh = 0.2);
	Mat predictFile(const QString &filename, float thresh = 0.2);

	void yoloImage(const QString &cfg, const QString &weights, const QString &filename, float thresh);
	void yoloImage(const QString &filename, float thresh);

protected:
	QString getAbs(const QString &path);

	QString darkNetRootPath;
	DarkNetPriv *priv;
};

#endif // DARKNET_H

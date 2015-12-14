#ifndef CAFFECNN_H
#define CAFFECNN_H

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
signals:

public slots:

protected:
	CaffeCnnPriv *p;
};

#endif // CAFFECNN_H

#ifndef PYRAMIDS_H
#define PYRAMIDS_H

#include <QObject>

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class Pyramids : public QObject
{
	Q_OBJECT
public:
	explicit Pyramids(QObject *parent = 0);

signals:

public slots:
	Mat makeSpm(const QString &filename, int L);
	Mat makeHistImage(const Mat &hist, int scale = 0);
	void setDict(const QString &filename);
	void setDict(const Mat &codewords);

protected:
	Mat dict;
};

#endif // PYRAMIDS_H

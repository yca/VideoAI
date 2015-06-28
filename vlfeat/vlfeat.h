#ifndef VLFEAT_H
#define VLFEAT_H

#include <QObject>
#include <QString>

#include <opencv2/core/core.hpp>

extern "C" {
#include <vl/homkermap.h>
}

using namespace cv;
using namespace std;

class VlFeat : public QObject
{
	Q_OBJECT
public:
	explicit VlFeat(QObject *parent = 0);

	static QString toSvmLine(VlHomogeneousKernelMap *map, const Mat &spm, int label);
	static void exportToSvm(const Mat &pyramids, const Mat &labels, const QString &filename);
signals:

public slots:

};

#endif // VLFEAT_H

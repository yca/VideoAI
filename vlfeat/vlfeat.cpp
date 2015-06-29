#include "vlfeat.h"
#include "debug.h"

#include <QFile>

VlFeat::VlFeat(QObject *parent) :
	QObject(parent)
{
}

QString VlFeat::toSvmLine(VlHomogeneousKernelMap *map, const Mat &spm, int label, bool dense)
{
	float d[3];
	QString line;
	for (int i = 0; i < spm.rows; i++) {
		line += QString("%1 ").arg(label);
		int curr = 1;
		for (int j = 0; j < spm.cols; j++) {
			float val = spm.at<float>(0, j);
			vl_homogeneouskernelmap_evaluate_f(map, d, 1, val);
			if (!dense) {
				QString fts;
				if (d[0] != 0)
					fts.append(QString("%1:%2 ").arg(curr).arg(d[0]));
				if (d[1] != 0)
					fts.append(QString("%1:%2 ").arg(curr + 1).arg(d[1]));
				if (d[2] != 0)
					fts.append(QString("%1:%2 ").arg(curr + 2).arg(d[2]));
				line.append(fts);
			} else {
				line.append(QString("%1:%2 %3:%4 %5:%6 ")
							.arg(curr)
							.arg(d[0])
							.arg(curr + 1)
							.arg(d[1])
							.arg(curr + 2)
							.arg(d[2])
							);
			}
			curr += 3;
		}
		line.append("\n");
	}
	return line;
}

void VlFeat::exportToSvm(const Mat &pyramids, const Mat &labels, const QString &filename, bool dense)
{
	QFile f2(filename);
	f2.open(QIODevice::WriteOnly);
	VlHomogeneousKernelMap *map = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, 0.5, 1, -1, VlHomogeneousKernelMapWindowRectangular);
	float d[3];
	for (int i = 0; i < pyramids.rows; i++) {
		int label = labels.at<float>(i);
		QString line = QString("%1 ").arg(label);
		const Mat &m = pyramids.row(i);
		int curr = 1;
		for (int j = 0; j < pyramids.cols; j++) {
			float val = m.at<float>(0, j);
			vl_homogeneouskernelmap_evaluate_f(map, d, 1, val);
			if (!dense) {
				QString fts;
				if (d[0] != 0)
					fts.append(QString("%1:%2 ").arg(curr).arg(d[0]));
				if (d[1] != 0)
					fts.append(QString("%1:%2 ").arg(curr + 1).arg(d[1]));
				if (d[2] != 0)
					fts.append(QString("%1:%2 ").arg(curr + 2).arg(d[2]));
				line.append(fts);
			} else {
				line.append(QString("%1:%2 %3:%4 %5:%6 ")
							.arg(curr)
							.arg(d[0])
							.arg(curr + 1)
							.arg(d[1])
							.arg(curr + 2)
							.arg(d[2])
							);
			}
			curr += 3;
		}
		f2.write(line.toUtf8());
		f2.write("\n");
		qDebug() << "%" << i * 100.0 / pyramids.rows;
	}
	f2.close();
}

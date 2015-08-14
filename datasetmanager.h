#ifndef DATASETMANAGER_H
#define DATASETMANAGER_H

#include <QHash>
#include <QObject>
#include <QStringList>

#include <opencv2/opencv.hpp>

#include "opencv/opencv.h"

class DatasetManager : public QObject
{
	Q_OBJECT
public:
	explicit DatasetManager(QObject *parent = 0);

signals:

public slots:
	void addDataset(const QString &name, const QString &path);
	QStringList availableDatasets();
	QStringList allImages();
	QStringList dataSetImages(const QString &dataset);
	QString getImage(int pos);

	static QList<QPair<int, QString> > voc2007GetImagesForCateogory(const QString &path, QString key, QString cat);
	static void parseOxfordFeatures(const QString &path, const QString &ftPath, vector<vector<KeyPoint> > &kpts, vector<Mat> &features, vector<Mat> &ids);
	static void checkOxfordMissing(const QStringList &images, const QString &featuresBase);
	void convertOxfordFeatures(const QString &featuresBase);
	void calculateOxfordIdfs(const QStringList &images, const QString ftPaths, int cols);

protected:
	QString currentDataset;
	QHash<QString, QStringList> datasets;
};

#endif // DATASETMANAGER_H

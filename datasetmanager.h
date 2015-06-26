#ifndef DATASETMANAGER_H
#define DATASETMANAGER_H

#include <QHash>
#include <QObject>
#include <QStringList>

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

protected:
	QString currentDataset;
	QHash<QString, QStringList> datasets;
};

#endif // DATASETMANAGER_H

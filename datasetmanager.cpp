#include "datasetmanager.h"

#include <QDir>

static QStringList listDir(QString path, QString suffix)
{
	QStringList list;
	QDir d(path);
	QStringList files = d.entryList(QStringList() << QString("*.%1").arg(suffix)
									, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	foreach (QString f, files)
		list << path + "/" + f;
	QStringList subdirs = d.entryList(QDir::NoDotAndDotDot | QDir::Dirs, QDir::Name);
	foreach (QString s, subdirs)
		list << listDir(d.filePath(s), suffix);
	return list;
}

DatasetManager::DatasetManager(QObject *parent) :
	QObject(parent)
{
	datasets.insert("sample", QStringList() << "testImage");
}

void DatasetManager::addDataset(const QString &name, const QString &path)
{
	QStringList images = listDir(path, "jpg");
	datasets.insert(name, images);
	currentDataset = name;
}

QStringList DatasetManager::availableDatasets()
{
	return datasets.keys();
}

QStringList DatasetManager::allImages()
{
	QStringList images;
	QHashIterator<QString, QStringList> i(datasets);
	while (i.hasNext()) {
		i.next();
		images << i.value();
	}
	return images;
}

QStringList DatasetManager::dataSetImages(const QString &dataset)
{
	if (!dataset.contains(dataset))
		return QStringList();
	return datasets[dataset];
}

QString DatasetManager::getImage(int pos)
{
	return datasets[currentDataset][pos];
}

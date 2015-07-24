#include "datasetmanager.h"
#include "common.h"
#include "debug.h"

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
#if 0
	images.clear();
	QStringList lines = Common::importText("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007/ImageSets/Main/aeroplane_mintrain.txt");
	foreach (QString line, lines) {
		if (line.trimmed().isEmpty())
			continue;
		images << QString("/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages/%1.jpg").arg(line.split(" ").first().trimmed());
	}
	lines = Common::importText("/home/caglar/myfs/tasks/video_analysis/data/VOCdevkit/VOC2007/ImageSets/Main/aeroplane_mintest.txt");
	foreach (QString line, lines) {
		if (line.trimmed().isEmpty())
			continue;
		images << QString("/home/caglar/myfs/tasks/video_analysis/data/vocimages/JPEGImages/%1.jpg").arg(line.split(" ").first().trimmed());
	}
	images.removeDuplicates();
#endif
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

QList<QPair<int, QString> > DatasetManager::voc2007GetImagesForCateogory(const QString &path, QString key, QString cat)
{
	QDir d(path + "/JPEGImages");
	QStringList files = d.entryList(QStringList() << QString("*.jpg")
									, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	QHash<int, QString> fhash;
	foreach (QString file, files) {
		int no = file.split(".jpg").first().toInt();
		fhash.insert(no, d.filePath(file));
	}

	d.setPath(path + "/ImageSets/Main");
	files = d.entryList(QStringList() << QString("*.txt")
									, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	QList<QPair<int, QString> > images;
	foreach (QString file, files) {
		if (file != QString("%1_%2.txt").arg(cat).arg(key))
			continue;
		QStringList lines = Common::importText(d.filePath(file));
		foreach (QString line, lines) {
			line = line.trimmed();
			if (!line.contains(" "))
				continue;
			QStringList vals = line.split(" ", QString::SkipEmptyParts);
			images << QPair<int, QString>(vals[1].toInt(), fhash[vals[0].toInt()]);
		}
		break;
	}
	return images;
}

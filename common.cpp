#include "common.h"

#include <QDir>
#include <QFile>

Common::Common(QObject *parent) :
	QObject(parent)
{
}

void Common::exportText(const QString text, const QString &filename)
{
	QFile file(filename);
	file.open(QIODevice::WriteOnly);
	file.write(text.toUtf8());
	file.close();
}

QStringList Common::importText(const QString &filename)
{
	QFile file(filename);
	if (!file.open(QIODevice::ReadOnly))
		return QStringList();
	QStringList lines = QString::fromUtf8(file.readAll()).split("\n");
	file.close();
	return lines;
}

QStringList Common::listDir(QString path, QString suffix)
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

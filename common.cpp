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

#if 0
p->pyr->setDict("/home/amenmd/myfs/tasks/hilal_tez/work/ox_complete/oxbuild_images_512.dict");
QString path = "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/";
QStringList lines = Common::importText("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/bodleian_1_good.txt");
foreach (QString line, lines) {
	if (line.trimmed().isEmpty())
		continue;
	ffDebug() << "processing" << line;
	Mat spm = p->pyr->makeSpm(path + line + ".jpg", 0);
	Mat im = p->pyr->makeHistImage(spm);
	OpenCV::exportMatrixTxt(QString("/home/amenmd/myfs/temp/oxdata/absent/%1_desc.txt").arg(line), spm);
	OpenCV::saveImage(QString("/home/amenmd/myfs/temp/oxdata/absent/%1_desc.jpg").arg(line), im);
	QFile::copy(path + line + ".jpg", "/home/amenmd/myfs/temp/oxdata/absent/" + line + ".jpg");
}
#elif 0
QString line = "all_souls_000013";
Mat spm = p->pyr->makeSpm(path + line + ".jpg", 0);
Mat im = p->pyr->makeHistImage(spm);
OpenCV::exportMatrixTxt(QString("/home/amenmd/myfs/temp/oxdata/query/%1_desc.txt").arg(line), spm);
OpenCV::saveImage(QString("/home/amenmd/myfs/temp/oxdata/query/%1_desc.jpg").arg(line), im);
QFile::copy(path + line + ".jpg", "/home/amenmd/myfs/temp/oxdata/query/" + line + ".jpg");
QStringList vals = QString("X 136.5 34.1 648.5 955.7").split(" ");
cv::Rect r(Point2f(vals[1].toFloat(), vals[2].toFloat()), Point2f(vals[3].toFloat(), vals[4].toFloat()));
im = OpenCV::loadImage(path + line + ".jpg");
OpenCV::saveImage(QString("/home/amenmd/myfs/temp/oxdata/query/%1_roi.jpg").arg(line), Mat(im, r));
spm = p->pyr->makeSpm(Mat(im, r), 0);
im = p->pyr->makeHistImage(spm);
OpenCV::exportMatrixTxt(QString("/home/amenmd/myfs/temp/oxdata/query/%1_desc_roi.txt").arg(line), spm);
OpenCV::saveImage(QString("/home/amenmd/myfs/temp/oxdata/query/%1_desc_roi.jpg").arg(line), im);
#endif

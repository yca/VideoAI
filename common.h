#ifndef COMMON_H
#define COMMON_H

#include <QObject>
#include <QStringList>

class Common : public QObject
{
	Q_OBJECT
public:
	explicit Common(QObject *parent = 0);

	static int hashMax(const QHash<int, int> &h);
	static bool isNan(float x) { return !(x*0.0 == 0.0); }
signals:

public slots:
	static void exportText(const QString text, const QString &filename);
	static QStringList importText(const QString &filename);
	static QByteArray importData(const QString &filename);
	static QStringList listDir(QString path, QString suffix);
};

#endif // COMMON_H

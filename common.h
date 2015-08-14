#ifndef COMMON_H
#define COMMON_H

#include <QObject>
#include <QStringList>

class Common : public QObject
{
	Q_OBJECT
public:
	explicit Common(QObject *parent = 0);

signals:

public slots:
	static void exportText(const QString text, const QString &filename);
	static QStringList importText(const QString &filename);
	static QByteArray importData(const QString &filename);
	static QStringList listDir(QString path, QString suffix);
};

#endif // COMMON_H

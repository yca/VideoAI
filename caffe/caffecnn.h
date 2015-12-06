#ifndef CAFFECNN_H
#define CAFFECNN_H

#include <QObject>
#include <QStringList>

class CaffeCnnPriv;

class CaffeCnn : public QObject
{
	Q_OBJECT
public:
	explicit CaffeCnn(QObject *parent = 0);

	int load(const QString &modelFile, const QString &trainedFile, const QString &meanFile, const QString &labelFile);
	int setMean(const QString &meanFile);
	QStringList classify(const QString &filename, int N = 5);
signals:

public slots:

protected:
	CaffeCnnPriv *p;
};

#endif // CAFFECNN_H

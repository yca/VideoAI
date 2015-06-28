#ifndef SNIPPETS_H
#define SNIPPETS_H

#include <QObject>

class Pyramids;
class DatasetManager;

class Snippets : public QObject
{
	Q_OBJECT
public:
	explicit Snippets(QObject *parent = 0);

signals:

public slots:
	static void caltech1(Pyramids *pyr, DatasetManager *dm);
	static void pyr2linearsvm(DatasetManager *dm, const QString &trainName, const QString &testName);
	static void pyr2svm(DatasetManager *dm, const QString &trainName, const QString &testName);
	static void oxfordTemp();
};

#endif // SNIPPETS_H

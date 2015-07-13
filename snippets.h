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
	static void caltech1();
	static void voc2007();
	static void vocpyr2linearsvm();
	static void pyr2linearsvm(const QString &trainName, const QString &testName);
	static void pyr2svm(DatasetManager *dm, const QString &trainName, const QString &testName);
	static void oxfordTemp();
	static void getAP(const QString &resultsFile, const QString &predictInputs, const QString &categories);
	static void toVOCKit(const QString &path);
};

#endif // SNIPPETS_H

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
	static void voc2007(int step, int L, int H);
	static void voc2007Min(int step, int L, int H);
	static void vocTrain(const QString &pyramidData, const QString &subFolder, double gamma, double cost);
	static void vocPredict(const QString &pyramidData, const QString &subFolder, double gamma);
	static void vocAP(const QString &subFolder);
	static void vocpyr2linearsvm(const QString &pyramidData, const QString &subFolder, double gamma);
	static void pyr2linearsvm(const QString &trainName, const QString &testName);
	static void pyr2svm(DatasetManager *dm, const QString &trainName, const QString &testName);
	static void oxfordTemp();
	static void getAP(const QString &resultsFile, const QString &predictInputs, const QString &categories);
	static void toVOCKit(const QString &path);
};

#endif // SNIPPETS_H

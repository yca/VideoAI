#ifndef OXFORDRETRIEVAL_H
#define OXFORDRETRIEVAL_H

#include <QObject>

class OxfordRetrieval : public QObject
{
	Q_OBJECT
public:
	explicit OxfordRetrieval(QObject *parent = 0);

	static void convertFeatures();
	static void createFeatures(const QString &base, int K, bool useHessianAffine, const QString &dictFileName);
	static void createPyramids(const QString &base, const QString &dictFileName, const QString &dfFileName);
	static void createInvertedIndex(const QString &base, const QString &dfFileName, const QString &outputFolder);
	static void runAllQueries(const QString &base, const QString &dfFileName, const QString &invertedIndexFolder);
	static void runSingleQuery(const QString &base, const QString &dfFileName, const QString &invertedIndexFolder);
signals:

public slots:
protected:
};

#endif // OXFORDRETRIEVAL_H

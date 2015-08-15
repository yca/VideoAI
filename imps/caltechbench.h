#ifndef CALTECHBENCH_H
#define CALTECHBENCH_H

#include <QObject>

class CaltechBench : public QObject
{
	Q_OBJECT
public:
	explicit CaltechBench(QObject *parent = 0);

	static void createImageFeatures();
	static void createImageDescriptors(const QString &dictFileName);
	static void createImageDescriptors2(const QString &dictFileName, int L, int flags);
	static void createImageIds(const QString &dictFileName);
	static void createDictionary(int K, int subSample = -1);
	static void exportForLibSvm();
	static void exportForLibSvmMulti(int flags);
signals:

public slots:

};

#endif // CALTECHBENCH_H

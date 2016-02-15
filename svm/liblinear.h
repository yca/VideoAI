#ifndef LIBLINEAR_H
#define LIBLINEAR_H

#include <QObject>

#include "opencv/opencv.h"

struct problem;
struct feature_node;
struct parameter;
struct model;

class LibLinear : public QObject
{
	Q_OBJECT
public:
	explicit LibLinear(QObject *parent = 0);
	~LibLinear();

	static int merge(const QString &fileName1, const QString &fileName2, const QString &outputName, int featureCount, int method);

	int setDataSize(int size, int fSize);
	void setCost(double c);
	int addData(const Mat &data, const Mat &label);
	int train();
	int predict(const Mat &m, int *label, double *probs);
	int save(const QString &filename);
	int load(const QString &filename);
signals:

public slots:
protected:
	problem *prob;
	parameter *pars;
	int trainSize;
	struct model *model;
	QList<feature_node *> nodes;
	//feature_node *features;

};

#endif // LIBLINEAR_H

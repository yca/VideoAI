#include "liblinear.h"
#include "linear.h"
#include "debug.h"

#include <errno.h>

static inline int getNonZeroCount(const Mat &data)
{
	int dcount = 0;
	float *d = (float *)data.data;
	for (int i = 0; i < data.cols; i++) {
		if (d[i])
			dcount++;
	}
	return dcount;
}

static void matToFNode(const Mat &data, feature_node *fnode)
{
	int dcount = 0;
	float *d = (float *)data.data;
	for (int i = 0; i < data.cols; i++) {
		if (d[i]) {
			fnode[dcount].index = i + 1; /* 1-based in liblinear */
			fnode[dcount].value = d[i];
			dcount++;
		}
	}
	fnode[dcount].index = -1;
}

LibLinear::LibLinear(QObject *parent) :
	QObject(parent)
{
	prob = new problem;
	prob->l = 0;
	prob->n = 0;
	prob->x = NULL;
	prob->y = NULL;
	prob->bias = -1;

	pars = new parameter;
	pars->solver_type = L2R_L2LOSS_SVC_DUAL;
	pars->eps = HUGE_VAL;
	pars->C = 1;
	pars->p = 0.1;
	pars->nr_weight = 0;
	pars->weight_label = NULL;
	pars->weight = NULL;

	model = NULL;
}

LibLinear::~LibLinear()
{
	if (prob->x)
		delete [] prob->x;
	if (prob->y)
		delete [] prob->y;
	for (int i = 0; i < nodes.size(); i++)
		delete [] nodes[i];
	nodes.clear();
	delete prob;
	delete pars;
}

int LibLinear::setDataSize(int size, int fSize)
{
	if (prob->x)
		return -EEXIST;
	prob->x = new feature_node*[size];
	prob->y = new double[size];
	trainSize = size;
	prob->n = fSize;
	return 0;
}

void LibLinear::setCost(double c)
{
	pars->C = c;
}

int LibLinear::addData(const Mat &data, const Mat &label)
{
	for (int i = 0; i < data.rows; i++) {
		const Mat &m = data.row(i);
		int dcount = getNonZeroCount(m);
		feature_node *fn = new feature_node[dcount + 1];
		matToFNode(m, fn);
		prob->x[prob->l] = fn;
		prob->y[prob->l++] = label.at<float>(i, 0);
		nodes << fn;
	}
	return 0;
}

int LibLinear::train()
{
	if (model)
		return -EEXIST;
	if (trainSize != prob->l)
		return -ENODATA;
	if (check_parameter(prob, pars))
		return -EINVAL;
	model = ::train(prob, pars);
	return 0;
}

int LibLinear::predict(const Mat &m, int *label, double *probs)
{
	if (!model)
		return -ENOENT;
	int dcount = getNonZeroCount(m);
	feature_node fn[dcount + 1];
	matToFNode(m, fn);
	*label = predict_probability(model, fn, probs);
	if (model->label[0] == -1)
		qSwap(probs[0], probs[1]);
	return 0;
}

int LibLinear::save(const QString &filename)
{
	if (!model)
		return -ENOENT;
	return save_model(qPrintable(filename), model);
}

int LibLinear::load(const QString &filename)
{
	if (model)
		return -EEXIST;
	model = load_model(qPrintable(filename));
	return 0;
}

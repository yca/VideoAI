#ifndef PYRAMIDSVL_H
#define PYRAMIDSVL_H

#include "vision/pyramids.h"

class PyramidsVl : public Pyramids
{
	Q_OBJECT
public:
	explicit PyramidsVl(QObject *parent = 0);

protected:
	virtual Mat extractFeatures(const Mat &im, vector<KeyPoint> &keypoints, int step);

};

#endif // PYRAMIDSVL_H

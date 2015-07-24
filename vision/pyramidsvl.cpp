#include "pyramidsvl.h"
#include "debug.h"

#include <vl/dsift.h>
#include <vl/imopv.h>

PyramidsVl::PyramidsVl(QObject *parent) :
	Pyramids(parent)
{
}

Mat PyramidsVl::extractFeatures(const Mat &im, vector<KeyPoint> &keypoints, int step)
{
	Q_UNUSED(step);
	Mat features(0, 128, CV_32F);
	double magnif = 6;
	QList<int> sizes;
	sizes << 4;
	sizes << 6;
	sizes << 8;
	sizes << 10;

	/* convert to float array */
	assert(im.type() == CV_8U);
	float *imdata = new float[im.rows * im.cols];
	for (int i = 0; i < im.rows; i++)
		for (int j = 0; j < im.cols; j++)
			imdata[i * im.cols + j] = im.row(i).data[j];
	float *smoothed = new float[im.rows * im.cols];

	for (int i = 0; i < sizes.size(); i++) {
		int step = sizes[i];

		/* smoothing step */
		double sigma = step / magnif;
		vl_imsmooth_f(smoothed, im.cols, imdata, im.cols, im.rows, im.cols, sigma, sigma);
		memcpy(smoothed, imdata, im.rows * im.cols * 4);

		/* denset sift */
		VlDsiftFilter *dsift = vl_dsift_new_basic(im.cols, im.rows, step, 8);
		vl_dsift_process(dsift, smoothed);
		int cnt = vl_dsift_get_keypoint_num(dsift);
		const float *descs = vl_dsift_get_descriptors(dsift);
		const VlDsiftKeypoint *kpts = vl_dsift_get_keypoints(dsift);
		for (int i = 0; i < cnt; i++) {
			Mat ft(1, 128, CV_32F);
			for (int j = 0; j < 128; j++)
				ft.at<float>(0, j) = qMin(descs[i * 128 + j] * 512, float(255));
			features.push_back(ft);
			KeyPoint kpt;
			kpt.pt.x = kpts[i].x;
			kpt.pt.y = kpts[i].y;
			keypoints.push_back(kpt);
		}
		vl_dsift_delete(dsift);
	}

	delete [] imdata;
	delete [] smoothed;

	return features;
}

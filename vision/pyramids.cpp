#include "pyramids.h"
#include "debug.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#if CV_MAJOR_VERSION > 2
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/nonfree/features2d.hpp>
#endif
#if CV_MAJOR_VERSION > 2
using namespace xfeatures2d;
#endif

#include "opencv/opencv.h"

#include <QFile>
#include <QColor>

static FlannBasedMatcher *matcher = NULL;
#pragma omp threadprivate(matcher)

static int histCount(int L)
{
	int binCount = 0;
	for (int i = 0; i <= L; i++)
		binCount += pow(4, i);
	return binCount;
}

static int findPointContribution(int x, int y, int level, int width, int height)
{
	int binsH = pow(2, level);
	int col = binsH - 1, row = binsH - 1;
	int xSpan = width / binsH;
	int ySpan = height / binsH;
	col = x / xSpan;
	row = y / ySpan;
	if (col >= binsH)
		col = binsH - 1;
	if (row >= binsH)
		row = binsH - 1;
	return col + row * binsH;
}

static Mat findPointContributions(int x, int y, int level, int width, int height)
{
	Mat cont(level + 1, 1, CV_32F);
	int off = 0;
	for (int i = 0; i <= level; i++) {
		cont.at<float>(i, 0) = off + findPointContribution(x, y, i, width, height);
		off += pow(2, i) * pow(2, i);
	}
	return cont;
}

Pyramids::Pyramids(QObject *parent) :
	QObject(parent)
{
}

Mat Pyramids::makeSpm(const QString &filename, int L)
{
	if (!QFile::exists(filename))
		return Mat();
	if (!dict.rows)
		return Mat();
	Mat im(imread(qPrintable(filename), IMREAD_GRAYSCALE));
	return makeSpmFromMat(im, L);
}

Mat Pyramids::makeSpmFromMat(const Mat &im, int L)
{
	int imW = im.cols;
	int imH = im.rows;

	int binCount = histCount(L);
	Mat linear = Mat::zeros(1, binCount * dict.rows, CV_32F);
	Mat hists = Mat(binCount, dict.rows, CV_32F, linear.data);//Mat::zeros(binCount, dict.rows, CV_32F);

	vector<KeyPoint> keypoints;
	Mat features;
	SIFT dec;
	dec.detect(im, keypoints);
	dec.compute(im, keypoints, features);

	std::vector<DMatch> matches;
	matcher->match(features, matches);

	assert(matches.size() == keypoints.size());
	/* calculate histogram values using matches and keypoints */
	for (uint i = 0; i < matches.size(); i++) {
		int idx = matches[i].trainIdx;
		const KeyPoint kpt = keypoints.at(i);
		Mat cont = findPointContributions(kpt.pt.x, kpt.pt.y, L, imW, imH);
		for (int j = 0; j < cont.rows; j++)
			hists.at<float>(cont.at<float>(j), idx) += 1;
	}

	/* normalize histogram */
	linear /= keypoints.size();

	return linear;
}

void Pyramids::setDict(const Mat &codewords)
{
	dict = codewords;
	#pragma omp parallel
	{
	if (!matcher) {
		matcher = new FlannBasedMatcher;
	}
	matcher->clear();
	matcher->add(std::vector<Mat>(1, dict));
	}
}

Mat Pyramids::makeHistImage(const Mat &hist, int scale, int foreColor, int backColor)
{
	Mat img = Mat::zeros(256, hist.cols, CV_8UC3);
	if (backColor != Qt::black) {
		int off = 0;
		if (foreColor == Qt::red)
			off = 0;
		else if (foreColor == Qt::green)
			off = 1;
		else if (foreColor == Qt::blue)
			off = 2;
		for (int i = off; i < img.rows * img.cols * 3; i += 3)
			img.at<uchar>(i) = 255;
	}
	Scalar color = Scalar(255, 255, 255);
	if (foreColor != Qt::white) {
		QColor clr = QColor(Qt::GlobalColor(foreColor));
		color = Scalar(clr.red(), clr.green(), clr.blue());
	}
	int lBottom = img.rows;
	double max = 1.0;
	double hs = 1.0;
	int height = img.rows;
	if (scale == 0) {
		minMaxLoc(hist, NULL, &max);
	} else if (scale > 0)
		hs = scale;
	for (int i = 0; i < hist.cols; i++) {
		int lx = i;
		int ly = lBottom - height * hist.at<float>(i) * hs / max;
		if (ly < 0)
			ly = 0;
		line(img, Point(lx, lBottom), Point(lx, ly), color, 1);
	}
	return img;
}

void Pyramids::setDict(const QString &filename)
{
	setDict(OpenCV::importMatrix(filename));
}

/*Mat Pyramids::makeSPM(const Mat &dists, const vector<KeyPoint> &keypoints, const Mat &dict, int L, int imW, int imH)
{
	QList<int> indices;
	//Mat dists = openmm::gpgpu::l2NormMin(features, dict);
	//#pragma omp parallel for
	for (int i = 0; i < dists.rows; i++) {
		Point idx;
		cv::minMaxLoc(dists.row(i), NULL, NULL, &idx, NULL);
		indices << idx.x;
	}
	return makeSPM(indices, keypoints, dict, L, imW, imH);
}*/

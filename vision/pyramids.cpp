#include "pyramids.h"
#include "debug.h"

#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION > 2
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/nonfree/features2d.hpp>
#endif
#if CV_MAJOR_VERSION > 2
using namespace xfeatures2d;
#endif
#include <opencv2/flann/flann.hpp>

#include "opencv/opencv.h"

#include <QFile>
#include <QColor>

static FlannBasedMatcher *matcher = NULL;
#pragma omp threadprivate(matcher)

int Pyramids::histCount(int L)
{
	int binCount = 0;
	for (int i = 0; i <= L; i++)
		binCount += pow(4, i);
	return binCount;
}

int Pyramids::findPointContribution(int x, int y, int level, int width, int height)
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

int Pyramids::findPointContributionHor(int x, int binsH, int width)
{
	int xSpan = width / binsH;
	int col = x / xSpan;
	if (col >= binsH)
		col = binsH - 1;
	return col;
}

Mat Pyramids::findPointContributions(int x, int y, int level, int width, int height)
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

vector<KeyPoint> Pyramids::extractDenseKeypoints(const Mat &m, int step)
{
	vector<KeyPoint> keypoints;
	DenseFeatureDetector dec(11.f, 1, 0.1f, step, 0);
	dec.detect(m, keypoints);
	return keypoints;
}

vector<KeyPoint> Pyramids::extractKeypoints(const Mat &m)
{
	vector<KeyPoint> keypoints;
	SiftFeatureDetector dec;
	dec.detect(m, keypoints);
	return keypoints;
}

Mat Pyramids::computeFeatures(const Mat &m, vector<KeyPoint> &keypoints)
{
	Mat features;
	SiftDescriptorExtractor ex;
	ex.compute(m, keypoints, features);
	return features;
}

Mat Pyramids::clusterFeatures(const Mat &features, int clusterCount)
{
	fDebug("will cluster %d features", features.rows);
	const cvflann::KMeansIndexParams p;
	Mat centers(clusterCount, features.cols, CV_32F);
	int ccnt = cv::flann::hierarchicalClustering<flann::L2<float> >(features, centers, p);
	return centers.rowRange(Range(0, ccnt));
}

void Pyramids::createDictionary(const QStringList &images, int clusterCount)
{
	computeImageFeatures(images);
	dict = clusterFeatures(OpenCV::subSampleRandom(imageFeatures, clusterCount * 100), clusterCount);
}

void Pyramids::createDictionary(int clusterCount)
{
	dict = clusterFeatures(OpenCV::subSampleRandom(imageFeatures, clusterCount * 100), clusterCount);
}

void Pyramids::computeImageFeatures(const QStringList &images, int samplesPerImage)
{
	Mat features(0, 128, CV_32F);
	for (int i = 0; i < images.size(); i++) {
		mDebug("Processing %s, %d of %d", qPrintable(images[i]), i, images.size());
		Mat img = OpenCV::loadImage(images[i]);
		vector<KeyPoint> kpts = extractKeypoints(img);
		Mat fts = computeFeatures(img, kpts);
		if (samplesPerImage <= 0)
			features.push_back(fts);
		else
			features.push_back(OpenCV::subSampleRandom(fts, samplesPerImage));
	}
	imageFeatures = features;
}

Mat Pyramids::calculatePyramids(const QStringList &images, int L, int step)
{
	int size = images.size();
	Mat pyramids(size, makeSpm(images.first(), L, step).cols, CV_32F);
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		QString iname = images[i];
		const Mat m = makeSpm(iname, L, step);
		m.copyTo(pyramids.row(i));
		ffDebug() << i << size;
	}
	return pyramids;
}

Mat Pyramids::calculatePyramidsH(const QStringList &images, int L, int H, int step)
{
	int size = images.size();
	Mat pyramids(size, makeSpmH(images.first(), L, H, step).cols, CV_32F);
	#pragma omp parallel for
	for (int i = 0; i < pyramids.rows; i++) {
		QString iname = images[i];
		const Mat m = makeSpmH(iname, L, H, step);
		m.copyTo(pyramids.row(i));
		ffDebug() << i << pyramids.rows << iname;
	}
	return pyramids;
}

std::vector<DMatch> Pyramids::matchFeatures(const Mat &features)
{
	std::vector<DMatch> matches;
	matcher->match(features, matches);
	return matches;
}

vector<vector<DMatch> > Pyramids::matchFeatures(const Mat &features, int knn)
{
	vector<vector<DMatch> > matches;
	matcher->knnMatch(features, matches, knn);
	return matches;
}

Mat Pyramids::makeSpm(const QString &filename, int L, int step)
{
	if (!QFile::exists(filename))
		return Mat();
	if (!dict.rows)
		return Mat();
	Mat im(imread(qPrintable(filename), IMREAD_GRAYSCALE));
	return makeSpmFromMat(im, L, step);
}

Mat Pyramids::makeSpmH(const QString &filename, int L, int H, int step)
{
	if (!QFile::exists(filename))
		return Mat();
	if (!dict.rows)
		return Mat();
	Mat im(imread(qPrintable(filename), IMREAD_GRAYSCALE));
	return makeSpmFromMatH(im, L, H, step);
}

Mat Pyramids::makeSpmFromMat(const Mat &im, int L, int step)
{
	int imW = im.cols;
	int imH = im.rows;

	int binCount = histCount(L);
	Mat linear = Mat::zeros(1, binCount * dict.rows, CV_32F);
	Mat hists = Mat(binCount, dict.rows, CV_32F, linear.data);//Mat::zeros(binCount, dict.rows, CV_32F);

	vector<KeyPoint> keypoints;
	Mat features = extractFeatures(im, keypoints, step);

	std::vector<DMatch> matches;
	matcher->match(features, matches);

	assert(matches.size() == keypoints.size());
	/* calculate histogram values using matches and keypoints */
	for (uint i = 0; i < matches.size(); i++) {
		int idx = matches[i].trainIdx;
		int kid = matches[i].queryIdx;
		const KeyPoint kpt = keypoints.at(kid);
		Mat cont = findPointContributions(kpt.pt.x, kpt.pt.y, L, imW, imH);
		for (int j = 0; j < cont.rows; j++)
			hists.at<float>(cont.at<float>(j), idx) += 1;
	}
	for (int i = 0; i < hists.rows; i++) {
		Mat h = hists.row(i);
		h /= OpenCV::getL1Norm(h);
		h.copyTo(hists.row(i));
	}

	return linear / OpenCV::getL1Norm(linear);
}

Mat Pyramids::makeSpmFromMatH(const Mat &im, int L, int H, int step)
{
	int imW = im.cols;
	int imH = im.rows;

	int quadCount = histCount(L);
	int binCount = quadCount + H;
	Mat linear = Mat::zeros(1, binCount * dict.rows, CV_32F);
	Mat hists = Mat(binCount, dict.rows, CV_32F, linear.data);

	vector<KeyPoint> keypoints;
	Mat features = extractFeatures(im, keypoints, step);

	std::vector<DMatch> matches;
	matcher->match(features, matches);

	assert(matches.size() == keypoints.size());
	/* calculate histogram values using matches and keypoints */
	for (uint i = 0; i < matches.size(); i++) {
		int idx = matches[i].trainIdx;
		int kid = matches[i].queryIdx;
		const KeyPoint kpt = keypoints.at(kid);
		Mat cont = findPointContributions(kpt.pt.x, kpt.pt.y, L, imW, imH);
		for (int j = 0; j < cont.rows; j++)
			hists.at<float>(cont.at<float>(j), idx) += 1;

		/* horizontal contributions */
		int contH = findPointContributionHor(kpt.pt.x, H, imW);
		hists.at<float>(quadCount + contH, idx) += 1;
	}

	return linear / OpenCV::getL1Norm(linear);
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

Mat Pyramids::getDict()
{
	return dict;
}

Mat Pyramids::getImageFeatures()
{
	return imageFeatures;
}

void Pyramids::setImageFeatures(const Mat &features)
{
	imageFeatures = features;
}

Mat Pyramids::extractFeatures(const Mat &im, vector<KeyPoint> &keypoints, int step)
{
	Mat features;
	SIFT dec;
	if (step <= 0)
		dec.detect(im, keypoints);
	else
		keypoints = extractDenseKeypoints(im, step);
	dec.compute(im, keypoints, features);
	return features;
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

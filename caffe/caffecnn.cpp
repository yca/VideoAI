#include "caffecnn.h"
#include "debug.h"
#include "opencv/opencv.h"

#define USE_LMDB

#ifdef GPU
#undef GPU
#endif
#include <caffe/caffe.hpp>
#include <caffe/util/db.hpp>
#include <opencv2/opencv.hpp>
#include <caffe/util/db_lmdb.hpp>

#include <QMutexLocker>

using namespace cv;
using namespace std;
using namespace caffe;
using namespace db;

typedef std::pair<string, float> Prediction;

QMutex CaffeCnn::lock;

class CaffeCnnPriv {
public:
	shared_ptr<Net<float> > net;
	vector<string> labels;
	int channelCount;
	Mat mean;
	Size inputGeometry;
	LMDB *imdb;
	LMDBCursor *dbCursor;
};

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> argmax(const std::vector<float>& v, int N)
{
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

static void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, CaffeCnnPriv *p)
{
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && p->channelCount == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && p->channelCount == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && p->channelCount == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && p->channelCount == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != p->inputGeometry)
		cv::resize(sample, sample_resized, p->inputGeometry);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (p->channelCount == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, p->mean, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		  == p->net->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void wrapInputLayer(std::vector<cv::Mat>* input_channels, CaffeCnnPriv *p)
{
	Blob<float>* input_layer = p->net->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

static std::vector<float> predict(const cv::Mat& img, CaffeCnnPriv *p)
{
	Blob<float>* input_layer = p->net->input_blobs()[0];
	input_layer->Reshape(1, p->channelCount, p->inputGeometry.height, p->inputGeometry.width);
	/* Forward dimension change to all layers. */
	p->net->Reshape();

	std::vector<cv::Mat> input_channels;
	wrapInputLayer(&input_channels, p);

	preprocess(img, &input_channels, p);

	p->net->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = p->net->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

CaffeCnn::CaffeCnn(QObject *parent) : QObject(parent)
{
	lock.lock();
	static int once = 0;
	if (!once) {
		::google::InitGoogleLogging("VideoAi");
		once = 1;
	}
	Caffe::set_mode(Caffe::GPU);
	p = new CaffeCnnPriv;
	lock.unlock();
}

int CaffeCnn::load(const QString &modelFile, const QString &trainedFile, const QString &meanFile, const QString &labelFile)
{
	QMutexLocker locker(&lock);

	p->net.reset(new Net<float>(modelFile.toStdString(), TEST));
	p->net->CopyTrainedLayersFrom(trainedFile.toStdString());
	if (p->net->num_inputs() != 1)
		return -EINVAL;
	if (p->net->num_outputs() != 1)
		return -EINVAL;

	/* Load labels. */
	ifstream labels(qPrintable(labelFile));
	string line;
	while (std::getline(labels, line))
		p->labels.push_back(string(line));

	Blob<float> * inputLayer = p->net->input_blobs()[0];
	if (inputLayer->channels() != 3 && inputLayer->channels() != 1)
		return -EINVAL;
	Blob<float> * outputLayer = p->net->output_blobs()[0];
	p->channelCount = inputLayer->channels();
	p->inputGeometry = Size(inputLayer->width(), inputLayer->height());

	if (outputLayer->channels() != (int)p->labels.size()) {
		mDebug("output layer channels and category size mismatch: %d vs %d", outputLayer->channels(), (int)p->labels.size());
	}

	return setMean(meanFile);
}

int CaffeCnn::load(const QString &lmdbFolder)
{
	QMutexLocker locker(&lock);

	p->imdb = new LMDB;
	p->imdb->Open(lmdbFolder.toStdString(), db::READ);
	p->dbCursor = p->imdb->NewCursor();
	return 0;
}

int CaffeCnn::setMean(const QString &meanFile)
{
	if (meanFile.endsWith("_vgg")) {
		//Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].
		//Scalar channelMean = Scalar(103.939, 116.779, 123.68);
		vector<Mat> chnls;
		chnls.push_back(Mat::ones(p->inputGeometry, CV_32FC1) * 103.939);
		chnls.push_back(Mat::ones(p->inputGeometry, CV_32FC1) * 116.779);
		chnls.push_back(Mat::ones(p->inputGeometry, CV_32FC1) * 123.68);
		cv::merge(chnls, p->mean);
		return 0;
	}
	BlobProto blobProto;
	ReadProtoFromBinaryFileOrDie(qPrintable(meanFile), &blobProto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> meanBlob;
	meanBlob.FromProto(blobProto);
	if (meanBlob.channels() != p->channelCount)
		return -EINVAL;

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	vector<Mat> channels;
	float* data = meanBlob.mutable_cpu_data();
	for (int i = 0; i < p->channelCount; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += meanBlob.height() * meanBlob.width();
	}

	/* Merge the separate channels into a single image. */
	Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image illed with this value. */
	Scalar channelMean = cv::mean(mean);
	p->mean = Mat(p->inputGeometry, mean.type(), channelMean);

	return 0;
}

QStringList CaffeCnn::classify(const QString &filename, int N)
{
	Mat img = imread(qPrintable(filename), IMREAD_UNCHANGED);
	return classify(img, N);
}

QStringList CaffeCnn::classify(const Mat &img, int N)
{
	std::vector<float> output = predict(img, p);
	N = std::min<int>(p->labels.size(), N);
	std::vector<int> maxN = argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(p->labels[idx], output[idx]));
	}

	QStringList list;
	/* Print the top N predictions. */
	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		list << QString("%1 %2").arg(QString::fromStdString(p.first)).arg(p.second);
		continue;
		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
				  << p.first << "\"" << std::endl;
	}
	return list;
}

Mat CaffeCnn::readNextFeature(QString &key)
{
	if (!p->dbCursor->valid())
		return Mat();
	Datum d;
	d.ParseFromString(p->dbCursor->value());
	key.append(QString::fromStdString(p->dbCursor->key()));
	Mat m(1, d.float_data_size(), CV_32F);
	for (int i = 0; i < d.float_data_size(); i++)
		m.at<float>(0, i) = d.float_data(i);
	p->dbCursor->Next();
	return m;
}

Mat CaffeCnn::extract(const Mat &img, const QString &layerName)
{
	QMutexLocker locker(&lock);

	forwardImage(img);

	const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
	const float *bdata = blob->cpu_data() + blob->offset(0);

	Mat m(blob->width() * blob->height(), blob->channels(), CV_32F);
#if 0
	int row = 0, off = 0;
	for (int i = 0; i < blob->width(); i++) {
		for (int j = 0; j < blob->height(); j++) {
			for (int k = 0; k < blob->channels(); k++)
				m.at<float>(row, k) = bdata[off++];
			m.row(row) = m.row(row) / OpenCV::getL2Norm(m.row(row));
			row++;
		}
	}
#else
	for (int i = 0; i < blob->count(); i++) {
		int row = i % m.rows;
		int col = i / m.rows;
		m.at<float>(row, col) = bdata[i];
	}
	for (int i = 0; i < m.rows; i++)
		m.row(i) /= OpenCV::getL2Norm(m.row(i));
#endif
	assert(m.rows * m.cols == blob->count());
	return m;
}

static Mat extractBlobSumPooled(const shared_ptr<Blob<float> > blob)
{
	const float *bdata = blob->cpu_data() + blob->offset(0);
	assert(blob->width() * blob->height() * blob->channels() == blob->count());
	Mat m = Mat::zeros(1, blob->channels(), CV_32F);
	int cnt = blob->width() * blob->height();
	for (int i = 0; i < blob->channels(); i++)
		for (int j = 0; j < blob->height(); j++)
			for (int k = 0; k < blob->width(); k++)
				m.at<float>(0, i) += bdata[k + j * blob->width() + i * blob->height() * blob->width()] / cnt;
	//((n * K + k) * H + h) * W + w.
	//(k * H + h) * W + w
	//(k * H * W) + (h * W) + w
	return m / OpenCV::getL2Norm(m);
}

static Mat extractBlobMaxPooled(const shared_ptr<Blob<float> > blob)
{
	const float *bdata = blob->cpu_data() + blob->offset(0);
	assert(blob->width() * blob->height() * blob->channels() == blob->count());
	Mat m = Mat::zeros(1, blob->channels(), CV_32F);
	for (int i = 0; i < blob->channels(); i++)
		for (int j = 0; j < blob->height(); j++)
			for (int k = 0; k < blob->width(); k++)
				m.at<float>(0, i) = qMax(m.at<float>(0, i), bdata[k + j * blob->width() + i * blob->height() * blob->width()]);
	return m / OpenCV::getL2Norm(m);
}

static Mat extractBlobVector(const shared_ptr<Blob<float> > blob)
{
	const float *bdata = blob->cpu_data() + blob->offset(0);
	assert(blob->width() * blob->height() * blob->channels() == blob->count());
	Mat m2(1, blob->width() * blob->height() * blob->channels(), CV_32F);
	for (int i = 0; i < m2.cols; i++)
		m2.at<float>(0, i) = bdata[i];
	return m2 / OpenCV::getL2Norm(m2);
}

static Mat extractBlobMatrix(const shared_ptr<Blob<float> > blob)
{
	const float *bdata = blob->cpu_data() + blob->offset(0);
	Mat m(blob->width() * blob->height(), blob->channels(), CV_32F);
	for (int i = 0; i < blob->count(); i++) {
		int row = i % m.rows;
		int col = i / m.rows;
		m.at<float>(row, col) = bdata[i];
	}
	for (int i = 0; i < m.rows; i++)
		m.row(i) /= OpenCV::getL2Norm(m.row(i));
	assert(m.rows * m.cols == blob->count());
	return m;
}

Mat CaffeCnn::extractLinear(const Mat &img, const QString &layerName)
{
	QMutexLocker locker(&lock);

	forwardImage(img);

	const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
	return extractBlobVector(blob);
}

Mat CaffeCnn::extractLinear(const Mat &img, const QStringList &layers)
{
	QMutexLocker locker(&lock);

	forwardImage(img);

	int cols = 0;
	foreach (const QString &layerName, layers) {
		const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
		cols += blob->width() * blob->height() * blob->channels();
	}
	Mat m(1, cols, CV_32F);
	int off = 0;
	foreach (const QString &layerName, layers) {
		const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
		const float *bdata = blob->cpu_data() + blob->offset(0);
		cols = blob->width() * blob->height() * blob->channels();
		for (int i = 0; i < cols; i++)
			m.at<float>(0, off + i) = bdata[i];
		off += cols;
	}

	return m / OpenCV::getL2Norm(m);
}

vector<Mat> CaffeCnn::extractMulti(const Mat &img, const QStringList &layers, const QStringList &featureFlags)
{
	QMutexLocker locker(&lock);
	forwardImage(img);
	vector<Mat> features;
	for (int i = 0; i < layers.size(); i++) {
		const QString &layer = layers[i];
		const QString &info = featureFlags[i];
		const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layer.toStdString());
		if (info == "concat")
			features.push_back(extractBlobVector(blob));
		else if (info == "sum")
			features.push_back(extractBlobSumPooled(blob));
		else if (info == "max")
			features.push_back(extractBlobMaxPooled(blob));
		else
			features.push_back(extractBlobMatrix(blob));
	}

	return features;
}

static void augKri(const Mat &img, Size sz, vector<Mat> &images)
{
	int W = sz.width;
	int H = sz.height;
	int X = 256, Y = 256;
	Mat imgr;
	cv::resize(img, imgr, Size(X, Y));
	images.push_back(img);
	images.push_back(imgr(Rect(0, 0, W, H)));
	images.push_back(imgr(Rect(X - W, 0, W, H)));
	images.push_back(imgr(Rect(0, Y - W, W, H)));
	images.push_back(imgr(Rect(X - W, Y - W, W, H)));
	images.push_back(imgr(Rect((X - W) / 2, (Y - W) / 2, W, H)));
	int size = images.size();
	for (int i = 0; i < size; i++) {
		Mat flipped;
		cv::flip(images[i], flipped, 1);
		images.push_back(flipped);
	}
}

static void augRaz(const Mat &img, Size sz, vector<Mat> &images)
{
	int w = img.cols;
	int h = img.rows;
	int tw = sz.width < w ? sz.width : w / 2;
	int th = sz.height < h ? sz.height : h / 2;
	int x = w - tw;
	int y = h - th;
	images.push_back(img);
	images.push_back(img(Rect(0, 0, tw, th)));
	images.push_back(img(Rect(x, 0, tw, th)));
	images.push_back(img(Rect(0, y, tw, th)));
	images.push_back(img(Rect(x, y, tw, th)));
	images.push_back(img(Rect(x / 2, y / 2, tw, th)));
	images.push_back(OpenCV::rotate(img, 5));
	images.push_back(OpenCV::rotate(img, -5));
	int size = images.size();
	for (int i = 0; i < size; i++) {
		Mat flipped;
		cv::flip(images[i], flipped, 1);
		images.push_back(flipped);
	}
}

static void augPT(const Mat &img, Size sz, vector<Mat> &images)
{
	Q_UNUSED(sz);
	/*assert (img.type() == CV_8UC3);
	Mat m;
	img.convertTo(m, CV_32FC3);
	qDebug() << img.type() <<
	assert(0);*/
	images.push_back(OpenCV::gammaCorrection(img, 0.5));
}

vector<Mat> CaffeCnn::extractMulti(const Mat &img, const QStringList &layers, const QStringList &featureFlags, int augFlags)
{
	if ((augFlags & 0xff00) == 0)
		return extractMulti(img, layers, featureFlags);
	vector<Mat> images;
	if (augFlags & 0x0100)
		augKri(img, p->inputGeometry, images);
	if (augFlags & 0x0200)
		augRaz(img, p->inputGeometry, images);
	if (augFlags & 0x0400)
		augPT(img, p->inputGeometry, images);
	vector<Mat> ftsM;
	for (uint i = 0; i < images.size(); i++) {
		vector<Mat> fts = extractMulti(images[i], layers, featureFlags);
		for (uint j = 0; j < fts.size(); j++) {
			if (i == 0)
				ftsM.push_back(Mat::zeros(1, fts[j].cols, CV_32F));
			ftsM[j] += fts[j] / images.size();
		}
	}
	return ftsM;
}

vector<Mat> CaffeCnn::getFeatureMaps(const QString &layerName)
{
	vector<Mat> maps;

	const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
	const float *bdata = blob->cpu_data() + blob->offset(0);

	int off = 0;
	for (int i = 0; i <blob->channels(); i++) {
		Mat m2(blob->height(), blob->width(), CV_32F);
		for (int k = 0; k < m2.rows; k++)
			for (int j = 0; j < m2.cols; j++)
				m2.at<float>(k, j) = bdata[off++];
		maps.push_back(m2);
	}
	return maps;
}

Mat CaffeCnn::getGradients(const QString &layerName)
{
#if 0
	const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
	const float *bdiff = blob->cpu_data() + blob->offset(0);
	qDebug() << blob->width() << blob->height() << blob->channels();
	Mat m(blob->height(), blob->width(), CV_32FC3);
	int off = 0;
	for (int i = 0; i < blob->channels(); i++) {
		for (int k = 0; k < m.rows; k++)
			for (int j = 0; j < m.cols; j++)
				m.at<Vec2f>(k, j)[i] = bdiff[off++];
	}
	return m;
#else
	const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
	//Blob<float>* blob = p->net->input_blobs()[0];

	int width = blob->width();
	int height = blob->height();
	float* bdiff = blob->mutable_cpu_diff() + blob->offset(0);
	vector<Mat> channels2;
	for (int i = 0; i < blob->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, bdiff);
		channels2.push_back(channel);
		bdiff += width * height;
	}
	//Mat m;
	//cv::merge(channels2, m);
	//return m;
#endif

	std::vector<cv::Mat> channels8;
	for (uint i = 0; i < channels2.size(); i++) {
	//for (int i = channels2.size() - 1; i >= 0; i--) {
		double min, max;
		minMaxLoc(channels2[i], &min, &max);
		channels2[i] -= min;
		channels2[i] /= max;
		channels2[i] *= 255;
		cv::Mat sample(height, width, CV_8U);
		/*for (int j = 0; j < width; j++)
			for (int k = 0; k < height; k++)
				sample.at<uchar>(j, k) = qMin(channels2[i].at<float>(j, k), 255.0f);*/
		channels2[i].convertTo(sample, CV_8U);
		channels8.push_back(sample);
	}
	Mat merged;
	cv::merge(channels8, merged);
	//OpenCV::saveImage("test.jpg", merged);
	return merged;

	/*std::vector<cv::Mat> channels;
	std::vector<cv::Mat> channels8;
	cv::split(m, channels);
	for (uint i = 0; i < channels.size(); i++) {
		double min, max;
		minMaxLoc(channels[i], &min, &max);
		channels[i] -= min;
		channels[i] /= max;
		channels[i] *= 255;
		cv::Mat sample;
		channels[i].convertTo(sample, CV_8U);
		channels8.push_back(sample);
	}
	Mat merged;// = Mat::zeros(224, 224, CV_8UC3);
	cv::merge(channels8, merged);
	return merged;*/
}

Mat CaffeCnn::getSaliencyMap()
{
	Mat smap = getSaliencyDiff();

	double min, max;
	minMaxLoc(smap, &min, &max);
	smap -= min;
	smap /= max / 255;
	Mat sample;
	smap.convertTo(sample, CV_8U);
	return sample;
}

vector<Mat> CaffeCnn::getSaliencyMapVect()
{
	const shared_ptr<Blob<float> > blob = p->net->blob_by_name("data");
	int width = blob->width();
	int height = blob->height();
	float* bdiff = blob->mutable_cpu_diff() + blob->offset(0);
	vector<Mat> channels;
	for (int i = 0; i < blob->channels(); ++i) {
		double min, max;
		cv::Mat channel(height, width, CV_32FC1, bdiff);
		minMaxLoc(channel, &min, &max);
		channel -= min;
		channel /= max;
		channel *= 255;
		cv::Mat sample(height, width, CV_8U);
		channel.convertTo(sample, CV_8U);
		channels.push_back(sample);
		bdiff += width * height;
	}
	return channels;
}

QImage CaffeCnn::getSaliencyMapGray()
{
#if 0
	Mat smap = getSaliencyMapRgb();
	QImage im(smap.cols, smap.rows, QImage::Format_RGB888);
	for (int i = 0; i < smap.cols; i++) {
		for (int j = 0; j < smap.rows; j++) {
			float b = smap.at<Vec3f>(j, i)[0];
			float g = smap.at<Vec3f>(j, i)[1];
			float r = smap.at<Vec3f>(j, i)[2];
			//float val = qMax(, smap.at<Vec3f>(j, i)[1]);
			//val = qMax(val, smap.at<Vec3f>(j, i)[2]);
			im.setPixel(i, j, qRgb(r, g, b));
		}
	}
#endif

	vector<Mat> channels = getSaliencyMapVect();
	Mat smap = channels[0] + channels[1];
	smap = channels[1];//cv::max(channels[2], smap);
	QImage im(smap.cols, smap.rows, QImage::Format_RGB888);
	for (int i = 0; i < smap.cols; i++)
		for (int j = 0; j < smap.rows; j++)
			im.setPixel(i, j, qRgb(smap.at<uchar>(j, i), smap.at<uchar>(j, i), smap.at<uchar>(j, i)));
	return im;
}

Mat CaffeCnn::getSaliencyMapRgb()
{
	vector<Mat> channels = getSaliencyMapVect();
	Mat merged;
	cv::merge(channels, merged);
	return merged;
}

Mat CaffeCnn::getSaliencyDiff()
{
	const shared_ptr<Blob<float> > blob = p->net->blob_by_name("data");

	int width = blob->width();
	int height = blob->height();
	float* bdiff = blob->mutable_cpu_diff() + blob->offset(0);
	Mat smap = Mat::zeros(height, width, CV_32F);
	for (int i = 0; i < blob->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, bdiff);
		smap = cv::max(smap, channel);
		bdiff += width * height;
	}

	return smap;
}

int CaffeCnn::forwardImage(const QString &filename)
{
	Mat img = OpenCV::loadImage(filename, -1);
	forwardImage(img);
	return 0;
}

void CaffeCnn::printLayerInfo()
{
	for (uint i = 0; i < p->net->layer_names().size(); i++) {
		QString layer = QString::fromStdString(p->net->layer_names()[i]);
		if (p->net->has_blob(layer.toStdString())) {
			const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layer.toStdString());
			ffDebug() << layer << blob->width() << blob->height() << blob->channels() << blob->count() << blob->num();
		} else
			ffDebug() << layer;
	}
}

Mat CaffeCnn::getLayerDimensions(const QString &layer)
{
	const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layer.toStdString());
	Mat m(1, 4, CV_32F);
	m.at<float>(0, 0) = blob->width();
	m.at<float>(0, 1) = blob->height();
	m.at<float>(0, 2) = blob->channels();
	m.at<float>(0, 3) = blob->num();
	return m;
}

void CaffeCnn::printLayerInfo(const QStringList &layers)
{
	foreach (const QString &layer, layers) {
		const shared_ptr<Blob<float> > blob = p->net->blob_by_name(layer.toStdString());
		ffDebug() << layer << blob->width() << blob->height() << blob->channels() << blob->count() << blob->num();
	}
}

void CaffeCnn::printLayerInfo(const QString &modelFile, bool printEmpty)
{
	shared_ptr<Net<float> > net;
	net.reset(new Net<float>(modelFile.toStdString(), TEST));
	for (uint i = 0; i < net->layer_names().size(); i++) {
		QString layer = QString::fromStdString(net->layer_names()[i]);
		if (net->has_blob(layer.toStdString())) {
			const shared_ptr<Blob<float> > blob = net->blob_by_name(layer.toStdString());
			qDebug() << layer << blob->width() << blob->height() << blob->channels() << blob->count() << blob->num();
		} else if (printEmpty)
			qDebug() << layer;
	}
}

QStringList CaffeCnn::getBlobbedLayerNames()
{
	QStringList list;
	for (uint i = 0; i < p->net->layer_names().size(); i++) {
		QString layer = QString::fromStdString(p->net->layer_names()[i]);
		if (p->net->has_blob(layer.toStdString()))
			list << layer;
	}
	return list;
}

void CaffeCnn::forwardImage(const Mat &img)
{
	Blob<float>* input_layer = p->net->input_blobs()[0];
	input_layer->Reshape(1, p->channelCount, p->inputGeometry.height, p->inputGeometry.width);
	/* Forward dimension change to all layers. */
	p->net->Reshape();

	std::vector<cv::Mat> input_channels;
	wrapInputLayer(&input_channels, p);

	preprocess(img, &input_channels, p);

	p->net->ForwardPrefilled();
}

void CaffeCnn::backward()
{
	p->net->Backward();
}

int CaffeCnn::setBlobDiff(const QString &layerName, const Mat &m)
{
	shared_ptr<Blob<float> > blob = p->net->blob_by_name(layerName.toStdString());
	float *bdiff = blob->mutable_cpu_diff() + blob->offset(0);
	assert(m.rows * m.cols == blob->height() * blob->width() * blob->channels());
	int off = 0;
	for (int i = 0; i < blob->channels(); i++) {
		for (int k = 0; k < m.rows; k++)
			for (int j = 0; j < m.cols; j++)
				bdiff[off++] = m.at<float>(k, j);
	}
	return 0;
}

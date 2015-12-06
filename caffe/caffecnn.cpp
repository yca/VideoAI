#include "caffecnn.h"

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace caffe;

typedef std::pair<string, float> Prediction;

class CaffeCnnPriv {
public:
	shared_ptr<Net<float> > net;
	vector<string> labels;
	int channelCount;
	Mat mean;
	Size inputGeometry;
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
	::google::InitGoogleLogging("VideoAi");
	Caffe::set_mode(Caffe::GPU);
	p = new CaffeCnnPriv;
}

int CaffeCnn::load(const QString &modelFile, const QString &trainedFile, const QString &meanFile, const QString &labelFile)
{
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

	if (outputLayer->channels() != (int)p->labels.size())
		return -EINVAL;

	return setMean(meanFile);
}

int CaffeCnn::setMean(const QString &meanFile)
{
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
	Mat img = imread(qPrintable(filename), -1);
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


#include "darknet.h"
#include "debug.h"

#include "darknet/image.h"
#include "darknet/parser.h"
#include "darknet/network.h"
#include "darknet/detection_layer.h"
#include "darknet/cost_layer.h"
#include "darknet/utils.h"
#include "darknet/box.h"

#include "opencv/opencv.h"

extern "C" {
extern char *voc_names[];
extern image voc_labels[];
extern void yolo_image(const char *cfg, const char *weights, const char *filename, float thresh);
}

class DarkNetPriv
{
public:
	network net;
	box *boxes;
	float **probs;
	detection_layer l;
};

static void mat2Buf(const Mat &m, float * buf)
{
	for (int i = 0; i < m.rows; i++) {
		const uchar *row = m.row(i).data;
		for (int j = 0; j < m.cols; j++)
			buf[i * m.cols + j] = row[j] / 255.0;
	}
}

static void buf2Mat(Mat &m, const float * buf)
{
	for (int i = 0; i < m.rows; i++) {
		uchar *row = m.row(i).data;
		for (int j = 0; j < m.cols; j++)
			row[j] = qMin(buf[i * m.cols + j] * 255.0, 255.0);
	}
}

static image toDarkImage(const Mat &img)
{
	int h = img.rows;
	int w = img.cols;
	int c = img.channels();
	assert(c == 3);

	image out = make_image(w, h, c);

	std::vector<Mat> vec;
	cv::split(img, vec);

	/* convert bgr to rgb */
	mat2Buf(vec[2], &out.data[0 * out.w * out.h]);
	mat2Buf(vec[1], &out.data[1 * out.w * out.h]);
	mat2Buf(vec[0], &out.data[2 * out.w * out.h]);

	return out;
}

static Mat toCVImage(const image &im)
{
	Mat r(im.h, im.w, CV_8U);
	Mat g(im.h, im.w, CV_8U);
	Mat b(im.h, im.w, CV_8U);

	buf2Mat(r, &im.data[0 * im.w * im.h]);
	buf2Mat(g, &im.data[1 * im.w * im.h]);
	buf2Mat(b, &im.data[2 * im.w * im.h]);

	/* convert rgb to bgr */
	vector<Mat> l;
	l.push_back(b);
	l.push_back(g);
	l.push_back(r);

	Mat merged;
	cv::merge(l, merged);
	return merged;
}

static void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
	int i,j,n;
	//int per_cell = 5*num+classes;
	for (i = 0; i < side*side; ++i){
		int row = i / side;
		int col = i % side;
		for(n = 0; n < num; ++n){
			int index = i*num + n;
			int p_index = side*side*classes + i*num + n;
			float scale = predictions[p_index];
			int box_index = side*side*(classes + num) + (i*num + n)*4;
			boxes[index].x = (predictions[box_index + 0] + col) / side * w;
			boxes[index].y = (predictions[box_index + 1] + row) / side * h;
			boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
			boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
			for(j = 0; j < classes; ++j){
				int class_index = i*classes;
				float prob = scale*predictions[class_index+j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if(only_objectness){
				probs[index][0] = scale;
			}
		}
	}
}

static void yolo_image2(const char *cfg, const char *weights, const char *filename, float thresh)
{
	int i;
	for (i = 0; i < 20; ++i){
		char buff[256];
		sprintf(buff, "/home/amenmd/myfs/source-codes/oss/darknet/data/labels/%s.png", voc_names[i]);
		voc_labels[i] = load_image_color(buff, 0, 0);
	}

	network net = parse_network_cfg((char *)cfg);
	load_weights(&net, (char *)weights);
	detection_layer l = net.layers[net.n-1];
	set_batch_network(&net, 1);

	int j;
	float nms=.5;

	box *boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
	float **probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
	for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

	image im = load_image_color((char *)filename, 0, 0);
	image sized = resize_image(im, net.w, net.h);
	float *X = sized.data;
	float *predictions = network_predict(net, X);

	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
	//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
	draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
	show_image(im, "predictions");
	save_image(im, "predictions");

	show_image(sized, "resized");
	free_image(im);
	free_image(sized);
}

Darknet::Darknet()
{
	priv = NULL;
}

int Darknet::init(const QString &darkNetRoot)
{
	darkNetRootPath = darkNetRoot;
	for (int i = 0; i < 20; ++i){
		char buff[256];
		sprintf(buff, "%s/data/labels/%s.png", qPrintable(darkNetRoot), voc_names[i]);
		voc_labels[i] = load_image_color(buff, 0, 0);
	}
	priv = new DarkNetPriv;

	return 0;
}

int Darknet::loadNetwork(const QString &cfg, const QString &weights)
{
	priv->net = parse_network_cfg((char *)qPrintable(getAbs(cfg)));
	load_weights(&priv->net, (char *)qPrintable(getAbs(weights)));
	set_batch_network(&priv->net, 1);

	priv->l = priv->net.layers[priv->net.n-1];
	priv->boxes = (box *)calloc(priv->l.side*priv->l.side*priv->l.n, sizeof(box));
	priv->probs = (float **)calloc(priv->l.side*priv->l.side*priv->l.n, sizeof(float *));
	for(int j = 0; j < priv->l.side * priv->l.side * priv->l.n; ++j)
		priv->probs[j] = (float *)calloc(priv->l.classes, sizeof(float *));

	return 0;
}

void Darknet::predict(const QString &filename, float thresh)
{
	const Mat ori = OpenCV::loadImage(getAbs(filename), -1);
	Mat img;
	cv::resize(ori, img, cv::Size(priv->net.w, priv->net.h));
	image im = toDarkImage(img);

	float nms = 0.5;
	/* NOTE: network_predict doesn't allocate any data */
	float *predictions = network_predict(priv->net, im.data);
	const detection_layer l = priv->l;
	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, priv->probs, priv->boxes, 0);
	if (nms)
		do_nms_sort(priv->boxes, priv->probs, l.side*l.side*l.n, l.classes, nms);
	draw_detections(im, l.side*l.side*l.n, thresh, priv->boxes, priv->probs, voc_names, 0, 20);
	save_image(im, (char *)qPrintable(getAbs(QString(filename).replace(".jpg", "_pred.jpg"))));
	//ffDebug() << priv->net.n << priv->net.outputs << priv->l.classes;
}

Mat Darknet::predict(const Mat &ori, float thresh)
{
	Mat img;
	cv::resize(ori, img, cv::Size(priv->net.w, priv->net.h));
	image im = toDarkImage(img);

	float nms = 0.5;
	/* NOTE: network_predict doesn't allocate any data */
	float *predictions = network_predict(priv->net, im.data);
	const detection_layer l = priv->l;
	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, priv->probs, priv->boxes, 0);
	if (nms)
		do_nms_sort(priv->boxes, priv->probs, l.side*l.side*l.n, l.classes, nms);
	draw_detections(im, l.side*l.side*l.n, thresh, priv->boxes, priv->probs, voc_names, 0, 20);
	Mat ret = toCVImage(im);
	free_image(im);
	return ret;
}

Mat Darknet::predictFile(const QString &filename, float thresh)
{
	const Mat ori = OpenCV::loadImage(getAbs(filename), -1);
	return predict(ori, thresh);
}

void Darknet::yoloImage(const QString &cfg, const QString &weights, const QString &filename, float thresh)
{
	yolo_image2(qPrintable(cfg), qPrintable(weights), qPrintable(filename), thresh);
}

void Darknet::yoloImage(const QString &filename, float thresh)
{
	const Mat ori = OpenCV::loadImage(getAbs(filename), -1);
	Mat img;
	cv::resize(ori, img, cv::Size(priv->net.w, priv->net.h));
	image im = toDarkImage(img);
	//image im = load_image_color((char *)qPrintable(getAbs(filename)), 0, 0);

	//image sized = resize_image(im, priv->net.w, priv->net.h);
	//float *X = sized.data;
	float *X = im.data;
	float *predictions = network_predict(priv->net, X);

	float nms=.5;
	const detection_layer l = priv->l;
	convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, priv->probs, priv->boxes, 0);
	if (nms) do_nms_sort(priv->boxes, priv->probs, l.side*l.side*l.n, l.classes, nms);
	draw_detections(im, l.side*l.side*l.n, thresh, priv->boxes, priv->probs, voc_names, 0, 20);
	show_image(im, "predictions");
	save_image(im, "predictions");

	//show_image(sized, "resized");
	free_image(im);
	//free_image(sized);
}

QString Darknet::getAbs(const QString &path)
{
	return QString("%1/%2").arg(darkNetRootPath).arg(path);
}

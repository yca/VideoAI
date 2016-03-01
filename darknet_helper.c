#if 0

#include "darknet/image.h"
#include "darknet/parser.h"
#include "darknet/network.h"
#include "darknet/detection_layer.h"
#include "darknet/cost_layer.h"
#include "darknet/utils.h"
#include "darknet/box.h"

#include "darknet_helper.h"

extern char *voc_names[];
extern image voc_labels[];

struct darknet_priv {
	network net;
	box *boxes;
	float **probs;
};

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

void yolo_image(const char *cfg, const char *weights, const char *filename, float thresh)
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

struct darknet_helper * darknet_init(const char *darknet_root)
{
	struct darknet_helper *dnet = (struct darknet_helper *)malloc(sizeof(struct darknet_helper));
	dnet->priv = (struct darknet_priv *)malloc(sizeof(struct darknet_priv));
	int i;
	for (i = 0; i < 20; ++i){
		char buff[256];
		sprintf(buff, "%s/data/labels/%s.png", darknet_root, voc_names[i]);
		voc_labels[i] = load_image_color(buff, 0, 0);
	}
	return dnet;
}

int darknet_load_network(struct darknet_helper *dnet, const char *cfg, const char *weights)
{
	dnet->priv->net = parse_network_cfg((char *)cfg);
	load_weights(&dnet->priv->net, (char *)weights);
	set_batch_network(&dnet->priv->net, 1);

	detection_layer l = dnet->priv->net.layers[dnet->priv->net.n-1];
	dnet->priv->boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
	dnet->priv->probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
	int j;
	for(j = 0; j < l.side * l.side * l.n; ++j)
		dnet->priv->probs[j] = (float *)calloc(l.classes, sizeof(float *));

	return 0;
}

/*cv::Mat darknet_predict(darknet_helper *dnet, const cv::Mat &img)
{
	return cv::Mat();
}*/

#endif

#include "cnnvisualizer.h"
#include "ui_cnnvisualizer.h"
#include "imagewidget.h"
#include "debug.h"

#include "caffe/caffecnn.h"
#include "opencv/opencv.h"

#include <QFileDialog>
#include <QPlainTextEdit>

static QList<CaffeCnn *> loadModel(QString cbase, QString deployProto, QString modelFile, QString imageMeanProto)
{
	QList<CaffeCnn *> list;
	if (!deployProto.contains(",")) {
		CaffeCnn *c = new CaffeCnn;
		c->load(cbase + deployProto,
			   cbase + modelFile,
			   cbase + imageMeanProto,
			   cbase + "data/ilsvrc12/synset_words.txt");
		list << c;
	} else {
		QStringList l1 = deployProto.split(",");
		QStringList l2 = modelFile.split(",");
		QStringList l3 = imageMeanProto.split(",");
		assert(l1.size() == l2.size());
		assert(l1.size() == l3.size());
		for (int i = 0; i < l1.size(); i++) {
			CaffeCnn *c = new CaffeCnn;
			assert(c->load(cbase + l1[i],
				   cbase + l2[i],
				   cbase + l3[i],
				   cbase + "data/ilsvrc12/synset_words.txt") == 0);
			list << c;
		}
	}
	return list;
}

CNNVisualizer::CNNVisualizer(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::CNNVisualizer)
{
	ui->setupUi(this);

	on_pushLoad_clicked();
	on_pushLoadImage_clicked();
}

CNNVisualizer::~CNNVisualizer()
{
	delete ui;
}

void CNNVisualizer::on_pushLoad_clicked()
{
#if 0
	models = loadModel("/home/amenmd/myfs/source-codes/oss/caffe/",
			  "models/bvlc_reference_caffenet/deploy.prototxt,models/bvlc_googlenet/deploy.prototxt,models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt",
			  "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel,models/bvlc_googlenet/bvlc_googlenet.caffemodel,models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel",
			  "data/ilsvrc12/imagenet_mean.binaryproto,data/ilsvrc12/imagenet_mean.binaryproto,_vgg");
	ui->comboModels->addItems(QString("CaffeNet,GoogleNet,VGG19").split(","));
#else
	models = loadModel("/home/amenmd/myfs/source-codes/oss/caffe/",
			  "models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt",
			  "models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel",
			  "_vgg");
	ui->comboModels->addItems(QString("VGG19").split(","));
#endif
	cmodel = models[0];
}

void CNNVisualizer::on_pushShowLayers_clicked()
{
	QPlainTextEdit e;
	e.setPlainText(cmodel->getBlobbedLayerNames().join("\n"));
	e.show();
	while (e.isVisible())
		QApplication::processEvents();
}

void CNNVisualizer::on_pushLoadImage_clicked()
{
	/*QStringList filenames = QFileDialog::getOpenFileNames(this, trUtf8("File selection dialog"), trUtf8("Please select image(s)"));
	if (filenames.isEmpty())
		return;*/
#if 0
	QStringList filenames;
	filenames << "/home/amenmd/myfs/tasks/video_analysis/dataset/101_ObjectCategories/accordion/image_0001.jpg";
	foreach (const QString filename, filenames) {
		cmodel->forwardImage(filename);
		const vector<Mat> &maps = cmodel->getFeatureMaps("conv5");
		int rows = 16, cols = 16;
		ImageWidget w(rows, cols);
		for (uint i = 0; i < maps.size(); i++) {
			const Mat &m = maps[i];
			w.setCurrentCell(i / cols, i % cols);
			w.showImageMat(m);
		}
		w.show();
		while (w.isVisible())
			QApplication::processEvents();
		break;
	}
#else
	QStringList filenames;
	filenames << "/home/amenmd/Downloads/cat2.jpg";
	foreach (const QString filename, filenames) {
		cmodel->forwardImage(filename);
		Mat m = Mat::zeros(1, 1000, CV_32F);
		m.at<float>(0, 283) = 1;
		cmodel->setBlobDiff("fc8", m);
		cmodel->backward();
		//Mat smap = cmodel->getGradients("data");
		//Mat smap = cmodel->getSaliencyMapRgb();
		//QImage im = cmodel->getSaliencyMapGray();
		Mat smap = cmodel->getSaliencyMap();
		QImage im = OpenCV::toQImage(smap);
		ImageWidget w(1, 2);
		w.setCurrentCell(0, 0);
		w.showImage(filename);
		w.setCurrentCell(0, 1);
		//w.showImageMat(smap);
		w.showImage(im);
		w.show();
		while (w.isVisible())
			QApplication::processEvents();
		break;
	}

#endif
}

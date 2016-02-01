#include "cnnvisualizer.h"
#include "ui_cnnvisualizer.h"
#include "imagewidget.h"
#include "debug.h"

#include "caffe/caffecnn.h"

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
}

CNNVisualizer::~CNNVisualizer()
{
	delete ui;
}

void CNNVisualizer::on_pushLoad_clicked()
{
	models = loadModel("/home/amenmd/myfs/tasks/cuda/caffe_master/caffe/",
			  "models/bvlc_reference_caffenet/deploy.prototxt,models/bvlc_googlenet/deploy.prototxt",
			  "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel,models/bvlc_googlenet/bvlc_googlenet.caffemodel",
			  "data/ilsvrc12/imagenet_mean.binaryproto,data/ilsvrc12/imagenet_mean.binaryproto");
	ui->comboModels->addItems(QString("CaffeNet,GoogleNet").split(","));
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
}

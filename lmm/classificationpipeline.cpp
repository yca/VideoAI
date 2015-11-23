#include "classificationpipeline.h"
#include "datasetmanager.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>

#define DPATH "/home/caglar/myfs/tasks/video_analysis/data/101_ObjectCategories/"
#define DATASET "caltech"

#include <errno.h>

ClassificationPipeline::ClassificationPipeline(QObject *parent) :
	PipelineManager(parent)
{
	BaseLmmPipeline *p1 = addPipeline();
	//p1->append(new SourceLmmElement);
	p1->append(new OpSrcElement<ClassificationPipeline>(this, &ClassificationPipeline::readNextImage));
	p1->append(new OpElement<ClassificationPipeline>(this, &ClassificationPipeline::detectKeypoints));
	p1->end();

	dm = new DatasetManager;
	dm->addDataset(DATASET, DPATH);
	images = dm->dataSetImages(DATASET);


/*#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		ffDebug() << i << size;
		QString iname = images[i];
		const Mat &m = OpenCV::loadImage(iname);
		vector<KeyPoint> kpts = Pyramids::extractDenseKeypoints(m, 4);
		const Mat features = Pyramids::computeFeatures(m, kpts);
		OpenCV::exportKeyPoints(QString(iname).replace(".jpg", ".kpts"), kpts);
		OpenCV::exportMatrix(QString(iname).replace(".jpg", ".bin"), features);
	}*/
}

const RawBuffer ClassificationPipeline::readNextImage()
{
	if (!images.size())
		return RawBuffer::eof(this);
	const QString iname = images.takeFirst();
	const Mat &m = OpenCV::loadImage(iname);
	ffDebug() << iname << images.size();
	return RawBuffer("video/x-raw-yuv", 1024);
}

int ClassificationPipeline::detectKeypoints(const RawBuffer &buf)
{
	ffDebug() << buf.size();
	return 0;
}


int SourceLmmElement::processBlocking(int ch)
{
	Q_UNUSED(ch);
	return newOutputBuffer(ch, RawBuffer("video/x-raw-yuv", 1024));
}

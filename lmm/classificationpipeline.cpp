#include "classificationpipeline.h"
#include "datasetmanager.h"
#include "opencv/cvbuffer.h"
#include "vision/pyramids.h"
#include "vlfeat/vlfeat.h"
#include "common.h"
#include "caffe/caffecnn.h"
#include "buffercloner.h"

#include <lmm/debug.h>
#include <lmm/baselmmpipeline.h>

#include <QDir>
#include <QFile>
#include <QApplication>

#include <errno.h>

#define createEl(_func, _priv) new OpElement<ClassificationPipeline>(this, &ClassificationPipeline::_func, _priv,  #_func)
#define createEl2(_func) new OpSrcElement<ClassificationPipeline>(this, &ClassificationPipeline::_func, #_func)

static Mat mergeBuffers(const Mat &m1, const Mat &m2)
{
	Mat m(1, m1.cols + m2.cols, CV_32F);
	for (int i = 0; i < m1.cols; i++)
		m.at<float>(0, i) = m1.at<float>(0, i);
	for (int i = 0; i < m2.cols; i++)
		m.at<float>(0, m1.cols + i) = m2.at<float>(0, i);
	return m;
}

#if 0
static vector<Mat> subSampleImage(const Mat &img, int L)
{
	vector<Mat> images;
	for (int i = 0; i <= L; i++) {
		int cnt = pow(2, i);
		int w = img.cols / cnt;
		int h = img.rows / cnt;
		for (int j = 0; j < cnt * cnt; j++) {
			int x = (j % cnt) * w;
			int y = (j / cnt) * h;
			images.push_back(img(Rect(x, y, w, h)));
		}
	}
	return images;
}
#endif

#define TRANS_CROP1			0x01
#define TRANS_CROP2			0x02
#define TRANS_CROP3			0x04
#define TRANS_CROP4			0x08
#define TRANS_CROP5			0x10
#define TRANS_ROTATE_CW		0x20
#define TRANS_ROTATE_CCW	0x40
#define TRANS_PT			0x80
#define TRANS_FLIP_HOR		0x100

void ClassificationPipeline::augmentTrainData(QList<TrainInfo *> &trainInfo, TrainInfo *info, int dataAug)
{
	if (dataAug & 0x02) {
		/* Razavian style */
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR);
		trainInfo << new TrainInfo(info, TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_CROP5);
		trainInfo << new TrainInfo(info, TRANS_ROTATE_CW);
		trainInfo << new TrainInfo(info, TRANS_ROTATE_CCW);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP5);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_ROTATE_CW);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_ROTATE_CCW);
	}
	if (dataAug & 0x01) {
		/* Kriz style */
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR);
		trainInfo << new TrainInfo(info, TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_CROP5);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP1);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP2);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP3);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP4);
		trainInfo << new TrainInfo(info, TRANS_FLIP_HOR | TRANS_CROP5);
	}
	if (dataAug & 0x04) {
		trainInfo << new TrainInfo(info, TRANS_PT);
	}
}

ClassificationPipeline::ClassificationPipeline(QObject *parent) :
	PipelineManager(parent)
{
	pars.datasetName = "odtu";
	pars.datasetPath = "/home/caglar/myfs/tasks/video_analysis/data/odtu/annotation/dataset/";
	pars.ft = FEAT_SURF;
	pars.xStep = pars.yStep = 3;
	pars.exportData = true;
	pars.threads = 4;
	pars.K = 2048;
	pars.dictSubSample = 0;
	pars.useExisting = true;
	pars.createDict = false;
	pars.dataPath = "dataset2";
	pars.gamma = 1;
	pars.trainCount = 15;
	pars.testCount = 15;
	pars.L = 2;
	pars.maxFeaturesPerImage = 0;
	pars.maxMemBytes = (quint64)1024 * 1024 * 1024 * 2;
	pars.useExistingTrainSet = true;
	pars.cl = CLASSIFY_BOW;
	pars.imFlags = IMREAD_GRAYSCALE;
	pars.runId = 0;
}

ClassificationPipeline::ClassificationPipeline(const ClassificationPipeline::parameters &params, QObject *parent) :
	PipelineManager(parent)
{
	pars.runId = 0;
	pars = params;
}

const RawBuffer ClassificationPipeline::readNextImage()
{
	if (datasetIndex >= trainInfo.size())
		return RawBuffer(this);
	TrainInfo *info = trainInfo[datasetIndex++];
	if (info->useForTest == false && info->useForTrain == false)
		return RawBuffer("application/empty", 1);
	Mat img = OpenCV::loadImage(info->imageFileName, pars.imFlags);
	int w = img.cols;
	int h = img.rows;
	int modelW = 227 * 2;
	int modelH = 227 * 2;
	int tw = modelW < w ? modelW : w / 2;
	int th = modelH < h ? modelH : h / 2;
	int x = w - tw;
	int y = h - th;
	if (info->preprocess & TRANS_CROP1)
		img = img(Rect(0, 0, tw, th));
	if (info->preprocess & TRANS_CROP2)
		img = img(Rect(x, 0, tw, th));
	if (info->preprocess & TRANS_CROP3)
		img = img(Rect(0, y, tw, th));
	if (info->preprocess & TRANS_CROP4)
		img = img(Rect(x, y, tw, th));
	if (info->preprocess & TRANS_CROP5)
		img = img(Rect(x / 2, y / 2, tw, th));
	if (info->preprocess & TRANS_ROTATE_CW)
		img = OpenCV::rotate(img, pars.rotationDegree);
	if (info->preprocess & TRANS_ROTATE_CCW)
		img = OpenCV::rotate(img, -1 * pars.rotationDegree);
	if (info->preprocess & TRANS_PT)
		img = OpenCV::gammaCorrection(img, 0.5);
	if (info->preprocess & TRANS_FLIP_HOR)
		cv::flip(img, img, 1);

	CVBuffer buf(img);
	buf.pars()->metaData = info->imageFileName.toUtf8();
	buf.pars()->streamBufferNo = datasetIndex - 1;
	buf.pars()->videoWidth = buf.getReferenceMat().cols;
	buf.pars()->videoHeight = buf.getReferenceMat().rows;
	return buf;
}

RawBuffer ClassificationPipeline::exportForSvm(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	if (buf.getMimeType() != "application/cv-mat")
		return RawBuffer();

	int index = buf.constPars()->streamBufferNo;
	TrainInfo *info = trainInfo[index];
	CVBuffer *cbuf = (CVBuffer *)&buf;
	const Mat &desc = cbuf->getReferenceMat() / OpenCV::getL2Norm(cbuf->getReferenceMat());
	int label = info->label;
	QString line = QString("%1 ").arg(label);
	float *data = (float *)desc.row(0).data;
	for (int j = 0; j < desc.cols; j++) {
		if (data[j] != 0)
			line.append(QString("%1:%2 ").arg(j + 1).arg(data[j]));
	}
	if (info->useForTrain) {
		trainFile->write(line.toUtf8());
		trainFile->write("\n");
	} else if (info->useForTest) {
		testFile->write(line.toUtf8());
		testFile->write("\n");
	}

	return buf;
}

RawBuffer ClassificationPipeline::exportForSvmMulti(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);

	if (buf.getMimeType() == "application/cv-mat")
		return exportForSvm(buf, priv);

	if (buf.getMimeType() != "application/cv-matv")
		return RawBuffer();

	int index = buf.constPars()->streamBufferNo;
	TrainInfo *info = trainInfo[index];
	CVBuffer *cbuf = (CVBuffer *)&buf;
	int label = info->label;
	const vector<Mat> &fts = cbuf->getVector();
	for (uint k = 0; k < fts.size(); k++) {
		const Mat &desc = fts[k] / OpenCV::getL2Norm(fts[k]);

		QString line = QString("%1 ").arg(label);
		float *data = (float *)desc.row(0).data;
		for (int j = 0; j < desc.cols; j++) {
			if (data[j] != 0)
				line.append(QString("%1:%2 ").arg(j + 1).arg(data[j]));
		}
		int multiIndex = k;

		if (multiIndex == trainFilesMulti.size()) {
			QFile *file = new QFile(QString("%1/svm_train_ftype%2_K%3_step%4_L%5_gamma%6.txt")
									.arg(pars.dataPath)
									.arg(pars.ft)
									.arg(pars.K + multiIndex)
									.arg(pars.xStep)
									.arg(pars.L)
									.arg(pars.gamma)
									);
			file->open(QIODevice::WriteOnly);
			trainFilesMulti << file;
			file = new QFile(QString("%1/svm_test_ftype%2_K%3_step%4_L%5_gamma%6.txt")
							 .arg(pars.dataPath)
							 .arg(pars.ft)
							 .arg(pars.K + multiIndex)
							 .arg(pars.xStep)
							 .arg(pars.L)
							 .arg(pars.gamma)
							 );
			file->open(QIODevice::WriteOnly);
			testFilesMulti << file;
		}

		QFile *trainFile = trainFilesMulti[multiIndex];
		QFile *testFile = testFilesMulti[multiIndex];
		if (info->useForTrain) {
			trainFile->write(line.toUtf8());
			trainFile->write("\n");
		} else if (info->useForTest) {
			testFile->write(line.toUtf8());
			testFile->write("\n");
		}
	}

	return buf;
}

RawBuffer ClassificationPipeline::mergeFeatures(const RawBuffer &buf, int priv)
{
	Q_UNUSED(priv);
	static QHash<int, RawBuffer> buffers;
	int sno = buf.constPars()->streamBufferNo;
	if (buffers.contains(sno)) {
		const RawBuffer &buf2 = buffers[sno];
		CVBuffer *cbuf1 = (CVBuffer *)&buf;
		CVBuffer *cbuf2 = (CVBuffer *)&buf2;
#if 0
		int bowSize = histCount(pars.L) * pars.K;
		if (pars.homkermap)
			bowSize *= 3;
		return CVBuffer::createNewBuffer(mergeBuffersOrdered(cbuf1->getReferenceMat(), cbuf2->getReferenceMat(), bowSize), buf);
#else
		if (buf.constPars()->duration < buf.constPars()->duration)
			return CVBuffer::createNewBuffer(mergeBuffers(cbuf1->getReferenceMat(), cbuf2->getReferenceMat()), buf);
		return CVBuffer::createNewBuffer(mergeBuffers(cbuf2->getReferenceMat(), cbuf1->getReferenceMat()), buf);
#endif
	} else
		buffers.insert(sno, buf);

	return RawBuffer("application/empty", 1);
}

RawBuffer ClassificationPipeline::debugBuffer(const RawBuffer &buf, int priv)
{
	ffDebug() << buf.size() << priv << buf.constPars()->streamBufferNo;
	return CVBuffer::createNewBuffer(Mat(1, 1 + priv, CV_32F), buf);
}

void ClassificationPipeline::pipelineFinished()
{
	if (trainFile)
		trainFile->close();
	if (testFile)
		testFile->close();
	foreach (QFile *file, trainFilesMulti)
		file->close();
	foreach (QFile *file, testFilesMulti)
		file->close();
	stop();
	ffDebug() << "quitting";
	QApplication::exit();
}

void ClassificationPipeline::init()
{
	/* init thread branch data */
	createThreadData();

	/* init dataset */
	createDatasetInfo();

	/* create processing pipeline */
	createPipeline();

	/* adjust expected frame count for pipeline progress tracking */
	datasetIndex = 0;
	expectedFrameCount = 0;
	for (int i = 0; i < trainInfo.size(); i++)
		if (trainInfo[i]->useForTest || trainInfo[i]->useForTrain)
			expectedFrameCount++;

	if (pars.maxMemBytes) {
		int fsize = 128;
		if (pars.ft == FEAT_SURF)
			fsize = 64;
		else if (pars.ft == FEAT_SIFT)
			fsize = 128;
		else if (pars.ft == FEAT_CNN)
			fsize = 96;
		int dsize = fsize * 4;
		pars.maxFeaturesPerImage = (double)(pars.maxMemBytes) / trainInfo.size() / dsize;
		pars.dictSubSample = 0;
	}
}

void ClassificationPipeline::createDatasetInfo()
{
	dm = new DatasetManager;
	if (pars.datasetName == "ucf101")
		dm->addUCF101(pars.datasetPath, "/home/amenmd/myfs/tasks/video_analysis/dataset/ucf/ucfTrainTestlist");
	else
		dm->addDataset(pars.datasetName, pars.datasetPath);
	images = dm->dataSetImages(pars.datasetName);
	dm->exportImages(pars.datasetName, QString("%1/image_list.txt").arg(pars.dataPath));
	imageCount = images.size();
	finishedCount = 0;
	expectedFrameCount = imageCount;

	QDir d = QDir::current();
	d.mkpath(pars.dataPath);

	if (!pars.fileListTxt.isEmpty()) {
		QStringList lines = Common::importText(pars.fileListTxt);
		images.clear();
		foreach (const QString &line, lines) {
			QStringList flds = line.split(" ");
			if (flds.size() < 2)
				continue;
			images << flds.first();
		}
		assert(images.size() == imageCount);
	}
}

void ClassificationPipeline::initTrainTest()
{
	QString trainSetFileName = QString("%1/train_set.txt")
			.arg(pars.dataPath);
	if (pars.datasetName == "voc") {
		images.clear();
		QList<QPair<int, QString> > list = dm->voc2007GetImagesForCateogory(pars.datasetPath.remove("JPEGImages"), "trainval", "bus");
		for (int i = 0; i < list.size(); i++) {
			const QPair<int, QString> &p = list[i];
			TrainInfo *info = new TrainInfo;
			info->label = p.first;
			info->useForTrain = true;
			info->useForTest = false;
			info->imageFileName = p.second;
			trainInfo << info;
			images << info->imageFileName;
			augmentTrainData(trainInfo, info, pars.dataAug);
		}
		list = dm->voc2007GetImagesForCateogory(pars.datasetPath.remove("JPEGImages"), "test", "bus");
		for (int i = 0; i < list.size(); i++) {
			const QPair<int, QString> &p = list[i];
			TrainInfo *info = new TrainInfo;
			info->label = p.first;
			info->useForTrain = false;
			info->useForTest = true;
			info->imageFileName = p.second;
			images << info->imageFileName;
			trainInfo << info;
		}
		imageCount = images.size();
	} else if (QFile::exists(trainSetFileName) && pars.useExistingTrainSet) {
		QStringList lines = Common::importText(trainSetFileName);
		int ind = 0;
		foreach (const QString &line, lines) {
			QStringList vals = line.split(":");
			if (vals.size() != 3)
				continue;
			TrainInfo *info = new TrainInfo;
			info->label = vals[0].toInt();
			info->useForTrain = vals[1].toInt();
			info->useForTest = vals[2].toInt();
			info->imageFileName = images[ind++];
			trainInfo << info;

			if (info->useForTrain)
				augmentTrainData(trainInfo, info, pars.dataAug);
		}
	} else {
		/* split into train/test */
		if (pars.trainListTxt.isEmpty())
			createTrainTestSplit(trainSetFileName);
		else {
			QHash<QString, int> tthash;
			QHash<QString, int> cats;
			QStringList trainList = Common::importText(pars.trainListTxt);
			for (int j = 0; j < trainList.size(); j++) {
				QStringList flds = trainList[j].trimmed().split(" ");
				QString name = flds[0].remove(".avi");
				if (name.isEmpty())
					continue;
				QStringList vals = name.split("/");
				tthash.insert(vals.last(), 1);
				int cat = flds[1].trimmed().toInt();
				assert(cat);
				cats.insert(vals.first(), cat);
			}
			QStringList testList = Common::importText(pars.testListTxt);
			for (int j = 0; j < testList.size(); j++) {
				QStringList flds = testList[j].trimmed().split(" ");
				QString name = flds[0].remove(".avi");
				if (name.isEmpty())
					continue;
				tthash.insert(name.split("/").last(), 2);
			}
			QStringList lines;
			for (int i = 0; i < images.size(); i++) {
				TrainInfo *info = new TrainInfo;
				info->useForTrain = info->useForTest = false;
				QFileInfo fi(images[i]);
				QStringList flds = fi.baseName().split("_");
				flds.removeLast();
				QString key = flds.join("_");
				int val = tthash[key];
				if (val == 1)
					info->useForTrain = true;
				else if (val == 2)
					info->useForTest = true;
				info->label = cats[fi.dir().absolutePath().split("/").last()];
				info->imageFileName = images[i];
				assert(info->label);
				trainInfo << info;
				lines << QString("%1:%2:%3").arg(info->label).arg(info->useForTrain).arg(info->useForTest);

				if (info->useForTrain)
					augmentTrainData(trainInfo, info, pars.dataAug);
			}
			lines << "";
			Common::exportText(lines.join("\n"), trainSetFileName);
		}
	}
}

void ClassificationPipeline::createTrainTestSplit(const QString &trainSetFileName)
{
	QStringList cats;
	Mat labels(images.size(), 1, CV_32F);
	Mat classPos(images.size(), 1, CV_32F);
	int cp = 0;
	QHash<int, int> sampleCount;
	for (int i = 0; i < images.size(); i++) {
		QString iname = images[i];

		/* find label */
		QFileInfo fi(iname);
		QString cat = fi.dir().dirName();
		if (!cats.contains(cat)) {
			cp = 0;
			cats << cat;
		}
		int l = cats.indexOf(cat) + 1;
		labels.at<float>(i) = l;
		sampleCount[l]++;
		classPos.at<float>(i) = cp++;
	}

	int trcnt = pars.trainCount;
	int tscnt = pars.testCount;
	int total = trcnt + tscnt;
	vector<Mat> trainSet, testSet;
	for (int i = 0; i < cats.size(); i++) {
		int cnt = sampleCount[i + 1];
		Mat idx = OpenCV::createRandomized(0, cnt);
		/* special case: if trcnt = -1 and tscnt = 0, all will be used for training */
		if (trcnt > 0)
			trainSet.push_back(idx.rowRange(0, trcnt));
		else
			trainSet.push_back(idx);
		if (tscnt > 0)
			testSet.push_back(idx.rowRange(trcnt, idx.rows > total ? total : idx.rows));
	}

	QStringList lines;
	for (int i = 0; i < images.size(); i++) {
		TrainInfo *info = new TrainInfo;
		info->useForTrain = info->useForTest = false;

		int label = labels.at<float>(i);
		info->label = label;
		const Mat &mt = trainSet[label - 1];
		const Mat &me = testSet[label - 1];
		int cp = classPos.at<float>(i);
		if (OpenCV::matContains(mt, cp))
			info->useForTrain = true;
		else if (OpenCV::matContains(me, cp))
			info->useForTest = true;
		info->imageFileName = images[i];
		trainInfo << info;
		lines << QString("%1:%2:%3").arg(info->label).arg(info->useForTrain).arg(info->useForTest);

		if (info->useForTrain)
			augmentTrainData(trainInfo, info, pars.dataAug);
	}
	lines << "";
	Common::exportText(lines.join("\n"), trainSetFileName);
}

int ClassificationPipeline::pipelineOutput(BaseLmmPipeline *p, const RawBuffer &buf)
{
	Q_UNUSED(buf);
	if (++finishedCount == expectedFrameCount) {
		ffDebug() << "finished";
		emit pipelineFinished();
	}
	if (pars.debug) {
		static int cnt = 0;
		//if (++cnt % 100 == 0 || cnt > 9140)
			//ffDebug() << buf.constPars()->streamBufferNo << cnt;
		ffDebug() << buf.constPars()->streamBufferNo << cnt++ << p->getOutputQueue(0)->getFps() << expectedFrameCount;
	}

	return 0;
}

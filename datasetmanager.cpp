#include "datasetmanager.h"
#include "common.h"
#include "debug.h"

#include <QDir>

static QStringList listDir(QString path, QString suffix)
{
	QStringList list;
	QDir d(path);
	QStringList files = d.entryList(QStringList() << QString("*.%1").arg(suffix)
									, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	foreach (QString f, files)
		list << path + "/" + f;
	QStringList subdirs = d.entryList(QDir::NoDotAndDotDot | QDir::Dirs, QDir::Name);
	foreach (QString s, subdirs)
		list << listDir(d.filePath(s), suffix);
	return list;
}

DatasetManager::DatasetManager(QObject *parent) :
	QObject(parent)
{
	datasets.insert("sample", QStringList() << "testImage");
}

/**
 * @brief createTestingSubset
 * @param images
 * @param queryFileName
 * @return
 *
 * This function can be used to subsample some dataset images:
 *
 *	QStringList imagesTmp;
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/all_souls_1_query.txt");
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/all_souls_2_query.txt");
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/all_souls_3_query.txt");
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/ashmolean_1_query.txt");
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/christ_church_4_query.txt");
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/bodleian_3_query.txt");
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/cornmarket_1_query.txt");
	imagesTmp += createTestingSubset(images, "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/gt_files_170407/hertford_3_query.txt");
	images = imagesTmp;
	images.removeDuplicates();
 */
static QStringList createTestingSubset(const QStringList images, QString queryFileName)
{
	QStringList list;
	QFileInfo fi(images.first());

	QStringList lines = Common::importText(QString(queryFileName).split("_query.").first().append("_good.txt"));
	foreach (QString line, lines) {
		if (line.trimmed().isEmpty())
			continue;
		list << QString("%1/%2.jpg").arg(fi.absolutePath()).arg(line);
	}

	lines = Common::importText(QString(queryFileName).split("_query.").first().append("_ok.txt"));
	foreach (QString line, lines) {
		if (line.trimmed().isEmpty())
			continue;
		list << QString("%1/%2.jpg").arg(fi.absolutePath()).arg(line);
	}

	lines = Common::importText(QString(queryFileName).split("_query.").first().append("_junk.txt"));
	foreach (QString line, lines) {
		if (line.trimmed().isEmpty())
			continue;
		list << QString("%1/%2.jpg").arg(fi.absolutePath()).arg(line);
	}

	return list;
}

static void removeImageIff(QStringList *list, QString im)
{
	int ind =  list->indexOf(im);
	if (ind >= 0)
		list->removeAt(ind);
}

void DatasetManager::addDataset(const QString &name, const QString &path)
{
	QStringList images = listDir(path, "jpg");

	/* Oxford trick: remove unused file */
	int ind =  images.indexOf(path + "/ashmolean_000214.jpg");
	if (ind >= 0)
		images.removeAt(ind);
	/* Paris trick: remove corrupted files */
	removeImageIff(&images, path + "/paris_louvre_000136.jpg");
	removeImageIff(&images, path + "/paris_louvre_000146.jpg");
	removeImageIff(&images, path + "/paris_moulinrouge_000422.jpg");
	removeImageIff(&images, path + "/paris_museedorsay_001059.jpg");
	removeImageIff(&images, path + "/paris_notredame_000188.jpg");
	removeImageIff(&images, path + "/paris_pantheon_000284.jpg");
	removeImageIff(&images, path + "/paris_pantheon_000960.jpg");
	removeImageIff(&images, path + "/paris_pantheon_000974.jpg");
	removeImageIff(&images, path + "/paris_pompidou_000195.jpg");
	removeImageIff(&images, path + "/paris_pompidou_000196.jpg");
	removeImageIff(&images, path + "/paris_pompidou_000201.jpg");
	removeImageIff(&images, path + "/paris_pompidou_000467.jpg");
	removeImageIff(&images, path + "/paris_pompidou_000640.jpg");
	removeImageIff(&images, path + "/paris_sacrecoeur_000299.jpg");
	removeImageIff(&images, path + "/paris_sacrecoeur_000330.jpg");
	removeImageIff(&images, path + "/paris_sacrecoeur_000353.jpg");
	removeImageIff(&images, path + "/paris_triomphe_000662.jpg");
	removeImageIff(&images, path + "/paris_triomphe_000833.jpg");
	removeImageIff(&images, path + "/paris_triomphe_000863.jpg");
	removeImageIff(&images, path + "/paris_triomphe_000867.jpg");

	datasets.insert(name, images);
	currentDataset = name;
}

QStringList DatasetManager::availableDatasets()
{
	return datasets.keys();
}

QStringList DatasetManager::allImages()
{
	QStringList images;
	QHashIterator<QString, QStringList> i(datasets);
	while (i.hasNext()) {
		i.next();
		images << i.value();
	}
	return images;
}

QStringList DatasetManager::dataSetImages(const QString &dataset)
{
	if (!dataset.contains(dataset))
		return QStringList();
	return datasets[dataset];
}

QString DatasetManager::getImage(int pos)
{
	return datasets[currentDataset][pos];
}

QList<QPair<int, QString> > DatasetManager::voc2007GetImagesForCateogory(const QString &path, QString key, QString cat)
{
	QDir d(path + "/JPEGImages");
	QStringList files = d.entryList(QStringList() << QString("*.jpg")
									, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	QHash<int, QString> fhash;
	foreach (QString file, files) {
		int no = file.split(".jpg").first().toInt();
		fhash.insert(no, d.filePath(file));
	}

	d.setPath(path + "/ImageSets/Main");
	files = d.entryList(QStringList() << QString("*.txt")
									, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	QList<QPair<int, QString> > images;
	foreach (QString file, files) {
		if (file != QString("%1_%2.txt").arg(cat).arg(key))
			continue;
		QStringList lines = Common::importText(d.filePath(file));
		foreach (QString line, lines) {
			line = line.trimmed();
			if (!line.contains(" "))
				continue;
			QStringList vals = line.split(" ", QString::SkipEmptyParts);
			images << QPair<int, QString>(vals[1].toInt(), fhash[vals[0].toInt()]);
		}
		break;
	}
	return images;
}

void DatasetManager::parseOxfordFeatures(const QString &path, const QString &ftPath, vector<vector<KeyPoint> > &kpts, vector<Mat> &features, vector<Mat> &ids)
{
	QDir d(path);
	QStringList files = d.entryList(QStringList() << QString("*.txt")
									, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	foreach (QString file, files) {
		qDebug() << "parsing" << files.indexOf(file) << files.size();
		QStringList lines = Common::importText(d.filePath(file));
		vector<KeyPoint> fkpts;
		Mat fids(0, 1, CV_32S);
		/* first 2 lines can be ignored */
		for (int i = 2; i < lines.size(); i++) {
			QStringList vals = lines[i].split(" ", QString::SkipEmptyParts);
			if (vals.size() < 5)
				continue;
			KeyPoint kpt;
			kpt.pt.x = vals[1].toFloat();
			kpt.pt.y = vals[2].toFloat();
			fkpts.push_back(kpt);
			assert(vals[0].toInt() != 0);
			Mat m = Mat::ones(1, 1, CV_32S) * (vals[0].toInt() - 1); /* 1-based in dataset files */
			fids.push_back(m);
		}
		kpts.push_back(fkpts);
		ids.push_back(fids);
	}
	QByteArray ba = Common::importData(ftPath);
	const uchar *data = (const uchar *)ba.constData();
	for (uint i = 0; i < kpts.size(); i++) {
		qDebug() << "importing" << i << kpts.size();
		vector<KeyPoint> fkpts = kpts[i];
		Mat ffts(fkpts.size(), 128, CV_8U);
		for (uint j = 0; j < fkpts.size(); j++) {
			for (int k = 0; k < 128; k++) {
				ffts.at<uchar>(j, k) = data[k];
			}
			data += 128;
		}
		features.push_back(ffts);
	}
	data += 12;
	assert(data - (const uchar *)ba.constData() == ba.size());
	//return QPair<Mat, vector<KeyPoint> >(ids, kpts);
}

/**
 * @brief DatasetManager::checkOxfordMissing
 * @param images List of images to check.
 * @param featuresBase Base directory for Oxford images.
 *
 * This function checks files present to in Oxford database and prints
 * missing images in local database. Given folder, 'featuresBase', should
 * contain a README2.txt file and a subfolder 'oxbuild_images' containing
 * all images.
 */
void DatasetManager::checkOxfordMissing(const QStringList &images, const QString &featuresBase)
{
	QStringList lines = Common::importText(featuresBase + "/README2.txt");
	QStringList images2;
	for (int i = 20; i < lines.size(); i++) {
		if (!lines[i].contains("oxc1_"))
			continue;
		QString imname = lines[i].remove("oxc1_").append(".jpg");
		images2 << featuresBase + "/oxbuild_images//" + imname;
	}
	for (int i = 0; i < images.size(); i++)
		if (!images2.contains(images[i]))
			qDebug() << images[i];
}

/**
 * @brief DatasetManager::convertOxfordFeatures
 * @param featuresBase Base directory for Oxford images.
 *
 * This function converts Oxford features present into Oxford database
 * to our feature/keypoint format. There should be a 'feat_oxc1_hesaff_sift.bin'
 * file under 'featuresBase' directory. There should be 'word_oxc1_hesaff_sift_16M_1M'
 * folder under base folder as well.
 */
void DatasetManager::convertOxfordFeatures(const QString &featuresBase)
{
	QStringList images = dataSetImages("oxford");
	vector<vector<KeyPoint> > kpts;
	vector<Mat> features;
	vector<Mat> ids;
	QString base = featuresBase;
	parseOxfordFeatures(base + "/word_oxc1_hesaff_sift_16M_1M/", base + "/feat_oxc1_hesaff_sift.bin",
										kpts, features, ids);
	ffDebug() << dataSetImages("oxford").size() << kpts.size() << features.size() << ids.size();
	for (uint i = 0; i < kpts.size(); i++) {
		QString prefix = images[i].split(".").first().replace("oxbuild_images/", "oxford_features/");
		ffDebug() << "saving" << i << kpts.size() << prefix;
		OpenCV::exportKeyPoints(prefix + ".kpts", kpts[i]);
		OpenCV::exportMatrix(prefix + ".bin", features[i]);
		OpenCV::exportMatrix(prefix + ".ids", ids[i]);
	}
}

void DatasetManager::calculateOxfordIdfs(const QStringList &images, const QString ftPaths, int cols)
{
	/* calculate idf-s */
	Mat df = Mat::zeros(1, cols, CV_32S);
	for (int i = 0; i < images.size(); i++) {
		qDebug() << "idf" << i << images.size();
		QString iname = images[i].split("/", QString::SkipEmptyParts).last().split(".").first();
		Mat m = OpenCV::importMatrix(QString(iname).prepend(ftPaths).append("_pyr.bin"));
		Mat tcont = Mat::zeros(1, m.cols, CV_32S);
		for (int j = 0; j < m.cols; j++)
			if (m.at<float>(0, j) > 0)
				tcont.at<int>(0, j) = 1;
		for (int j = 0; j < m.cols; j++)
			if (tcont.at<int>(0, j))
				df.at<int>(0, j) += 1;
	}
	OpenCV::exportMatrix("data/ox_df.bin", df);
}

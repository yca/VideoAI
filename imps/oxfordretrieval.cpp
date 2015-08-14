#include "oxfordretrieval.h"
#include "datasetmanager.h"
#include "debug.h"
#include "common.h"
#include "vision/pyramids.h"

#include <QFile>
#include <QFileInfo>

#include <fstream>

enum distMetric {
	DISTM_L1,
	DISTM_L2,
	DISTM_HELLINGER,
	DISTM_HELLINGER2,
	DISTM_KL,
	DISTM_CHI2,
	DISTM_MAX,
	DISTM_MIN,
	DISTM_COS,
	DISTM_MAH,
};

typedef float precType;
#define normalizer(__m) OpenCV::getL1Norm(__m)
#define distcalc(__m1, __m2) OpenCV::getL2Norm(__m1, __m2)
#define DIST_SORT_ORDER CV_SORT_ASCENDING

static Mat applyIdf(const Mat &m, const Mat &df, int N)
{
	Mat m2(1, m.cols, CV_32F);
	static Mat idf;
	if (idf.rows == 0) {
		/* create idf index */
		idf = Mat(df.rows, df.cols, CV_32F);
		for (int i = 0; i < df.cols; i++) {
			if (df.at<int>(i) == 0)
				idf.at<float>(i) = 0;
			else
				idf.at<float>(i) = log((double)N / df.at<int>(i)) + 1;
		}
	}
	float *c = (float *)idf.data;
	float *src = (float *)m.data;
	float *dst = (float *)m2.data;
	#pragma omp parallel for
	for (int i = 0; i < m.cols; i++) {
		dst[i] = src[i] * c[i];
	}
	return m2;
}

static Mat postProcBow(const Mat &py, const Mat &df, int N)
{
	Mat bow = py / OpenCV::getL1Norm(py);
	bow = applyIdf(bow, df, N);
	return bow / normalizer(bow);
}

static Mat getBow(const Mat &ids, const Mat &df, int N, distMetric dm)
{
	Mat py = Mat::zeros(1, df.cols, CV_32F);
	for (int j = 0; j < ids.rows; j++)
		py.at<float>(ids.at<uint>(j)) += 1;
	Mat bow = py / OpenCV::getL1Norm(py);
	bow = applyIdf(bow, df, N);
	if (dm == DISTM_L1)
		return bow / OpenCV::getL1Norm(bow);
	if (dm == DISTM_L2)
		return bow / OpenCV::getL2Norm(bow);
	return bow;
}

static Mat getQueryIDs(QString ftPaths, QString queryFileName, vector<KeyPoint> &keypoints)
{
	QString query = Common::importText(queryFileName).first();
	QStringList vals = query.split(" ");
	QString bname = vals[0].remove("oxc1_");
	cv::Rect r(Point2f(vals[1].toFloat(), vals[2].toFloat()), Point2f(vals[3].toFloat(), vals[4].toFloat()));
	vector<KeyPoint> kpts = OpenCV::importKeyPoints(QString(bname).prepend(ftPaths).append(".kpts"));
	Mat ids = OpenCV::importMatrix(QString(bname).prepend(ftPaths).append(".ids"));

	/* filter query keypoints and ids */
	Mat idsF(0, ids.cols, ids.type());
	for (uint i = 0; i < kpts.size(); i++) {
		KeyPoint kpt = kpts[i];
		if (r.contains(kpt.pt)) {
			idsF.push_back(ids.row(i));
			keypoints.push_back(kpt);
		}
	}
	ids = idsF;

	return ids;
}

static double getKL(double p, double q)
{
	if (fabs(p) < DBL_EPSILON)
		return 0;
	if (fabs(q) < DBL_EPSILON)
		q = 1e-10;
	return p * log(p / q);
}

template<typename T1>
static Mat getQueryDists(const vector<Mat> &py, const QStringList &images, const QString ftPaths, const Mat &df, const vector<vector<int> > &iidx, const vector<vector<T1> > &iidx2, distMetric dist)
{
	Q_UNUSED(ftPaths);
	Q_UNUSED(df);
	Mat dists = Mat::zeros(images.size(), py.size(), sizeof(T1) == sizeof(float) ? CV_32F : CV_64F);
	Mat dists2 = Mat::zeros(images.size(), py.size(), sizeof(T1) == sizeof(float) ? CV_32F : CV_64F);
	Mat dists3 = Mat::zeros(images.size(), py.size(), sizeof(T1) == sizeof(float) ? CV_32F : CV_64F);
	for (uint j = 0; j < py.size(); j++) {
		ffDebug() << j << py.size();
		Mat pyq = py[j];
		for (int k = 0; k < pyq.cols; k++) {
			T1 val = pyq.at<float>(k);
			if (val) {
				const Mat c1;// = dcorr.row(k);
				const Mat cvar;// = dcorrCV.row(k);
				for (int i = 0; i < dists.rows; i++) {
					if (dist == DISTM_L1 || dist == DISTM_CHI2 || dist == DISTM_MAX)
						dists.at<T1>(i, j) += val;
					else if (dist == DISTM_L2)
						dists.at<T1>(i, j) += val * val;
					else if (dist == DISTM_KL)
						dists.at<T1>(i, j) += getKL(val, 0);
					else if (dist == DISTM_COS)
						dists3.at<T1>(i, j) += val * val;
					else if (dist == DISTM_MAH) {
						float sum = 0;
						for (int ii = 0; ii < c1.cols; ii++) {
							int kc = c1.at<int>(ii);
							if (kc == k)
								sum += val * cvar.at<float>(ii);
						}
						dists.at<T1>(i, j) += sum * val;
					} else if (dist == DISTM_HELLINGER2)
						dists.at<T1>(i, j) += val;

				}
			}
			//float c = log((double)images.size() / df.at<int>(k));
			const vector<int> &v = iidx[k];
			const vector<T1> &v2 = iidx2[k];
#if 0
			static int once = 0;
			if (!once) {
				/* This piece of code checks validity index data, i.e. it loads original pyramid and checks value of given index. It may also compare L2 distances */
				Mat impy = Mat::zeros(1, pyq.cols, CV_32F);
				int rest = 3893;
				for (int ii = 0; ii < pyq.cols; ii++) {
					const vector<int> &v = iidx[ii];
					for (uint i = 0; i < v.size(); i++) {
						int ind = v[i];
						if (ind == rest)
							impy.at<float>(ii) = iidx2[ii][i];
					}
				}
				QString iname = images[rest].split("/", QString::SkipEmptyParts).last().split(".").first();
				qDebug() << OpenCV::getL2Norm(impy, pyq) << iname;
				Mat m = OpenCV::importMatrix(QString(iname).prepend(ftPaths).append("_pyr.bin"));
				m = postProcBow(m, df, images.size());
				for (int ii = 0; ii < pyq.cols; ii++)
					if (m.at<float>(ii) != impy.at<float>(ii))
						qDebug() << m.at<float>(ii) << impy.at<float>(ii) << ii;
				once = 1;
			}
#endif
			for (uint i = 0; i < v.size(); i++) {
				int ind = v[i];
				T1 tmp = 0;
				T1 val2 = v2[i];

				if (dist == DISTM_L1)
					tmp = qAbs(val2 - val) - val;
				else if (dist == DISTM_HELLINGER)
					tmp = sqrt(val2 * val);
				else if (dist == DISTM_KL) {
					tmp = getKL(val, val2) + getKL(val2, val) - getKL(val, 0);
				} else if (dist == DISTM_CHI2) {
					tmp = val - val2;
					if (val < DBL_EPSILON)
						tmp = 0;
					else
						tmp = tmp * tmp / val - val;
				} else if (dist == DISTM_MAX)
					tmp = qMax(val2, val) - val;
				else if (dist == DISTM_MIN)
					tmp = -1 * qMin(val2, val);
				else if (dist == DISTM_L2) {
					//dists.at<T1>(ind, j) -= (val * val);
					//tmp = (val2 - val) * 2.0 * (val2 - val) * 2.0;// - (val * val);
					tmp = pow(val2, 2) - 2 * val * val2;
				} else if (dist == DISTM_COS) {
					tmp = val2 * val;
					dists2.at<T1>(ind, j) += val2 * val2;
				} else if (dist == DISTM_MAH) {
#if 0
					const Mat &c1 = dcorr.row(k);
					const Mat &cvar = dcorrCV.row(k);
					T1 tmp2 = 0;
					for (int ii = 0; ii < c1.cols; ii++) {
						int kc = c1.at<int>(ii);
						float ck = cvar.at<float>(ii);
						if (kc == k) {
							tmp += val2 * ck - val * ck;
							tmp2 += val2 * ck;
						}
					}
					tmp = tmp * val2 - tmp2 * val;
#endif
				} else if (dist == DISTM_HELLINGER2)
					tmp = val2 - 2 * sqrt(val) * sqrt(val2);
				dists.at<T1>(ind, j) += tmp;
			}
		}
		if (dist == DISTM_HELLINGER) {
			for (int i = 0; i < dists.rows; i++) {
				T1 r = dists.at<T1>(i, j);
				dists.at<T1>(i, j) = sqrt(qMax(1. - r, 0.));
			}
		} else if (dist == DISTM_COS) {
			for (int i = 0; i < dists.rows; i++) {
				T1 p = dists.at<T1>(i, j);
				T1 q = dists2.at<T1>(i, j);
				T1 r = dists3.at<T1>(i, j);
				dists.at<T1>(i, j) = -1 * p / (sqrt(q) * sqrt(r)); /* remember cosine is an inverse smilarity */
			}
		} else if (dist == DISTM_HELLINGER2) {
			for (int i = 0; i < dists.rows; i++) {
				T1 r = dists.at<T1>(i, j);
				dists.at<T1>(i, j) = sqrt(r) / sqrt(2);
			}
		}
	}

	return dists;
}

static Mat getQueryDists(const vector<Mat> &py, const QStringList &images, const QString ftPaths, const Mat &df)
{
	Mat dists(images.size(), py.size(), CV_32F);
	/* due to memory constraints we run query in chunks */
	for (int i = 0; i < images.size(); i++) {
		qDebug() << "checking" << i << images.size();
		QString iname = images[i].split("/", QString::SkipEmptyParts).last().split(".").first();
		Mat m = OpenCV::importMatrix(QString(iname).prepend(ftPaths).append("_pyr.bin"));
		m = postProcBow(m, df, images.size());

		for (uint j = 0; j < py.size(); j++) {
			float d = distcalc(m, py[j]);
			dists.at<float>(i, j) = d;
		}
	}
	return dists;
}

static vector<string>
load_list(const string& fname)
{
	vector<string> ret;
	ifstream fobj(fname.c_str());
	if (!fobj.good()) { cerr << "File " << fname << " not found!\n"; exit(-1); }
	string line;
	while (getline(fobj, line)) {
		ret.push_back(line);
	}
	return ret;
}

template<class T>
set<T> vector_to_set(const vector<T>& vec)
{ return set<T>(vec.begin(), vec.end()); }

static float compute_ap(const set<string>& pos, const set<string>& amb, const vector<string>& ranked_list, const Mat &dists)
{
	float old_recall = 0.0;
	float old_precision = 1.0;
	float ap = 0.0;

	size_t intersect_size = 0;
	size_t i = 0;
	size_t j = 0;
	for ( ; i<ranked_list.size(); ++i) {
		if (amb.count(ranked_list[i])) continue;
		int cnt = pos.count(ranked_list[i]);
		if (cnt)
			intersect_size++;
		if (dists.rows)
			cout << (cnt ? "good " : "bad ") << ranked_list[i] << " " << dists.row(i) << endl;

		float recall = intersect_size / (float)pos.size();
		float precision = intersect_size / (j + 1.0);

		ap += (recall - old_recall)*((old_precision + precision)/2.0);

		old_recall = recall;
		old_precision = precision;
		j++;
	}
	return ap;
}

static float compute_ap_main(const char *path, const char *query_file)
{
	string gtq = path;

	vector<string> ranked_list = load_list(query_file);
	set<string> good_set = vector_to_set( load_list(gtq + "_good.txt") );
	set<string> ok_set = vector_to_set( load_list(gtq + "_ok.txt") );
	set<string> junk_set = vector_to_set( load_list(gtq + "_junk.txt") );

	set<string> pos_set;
	pos_set.insert(good_set.begin(), good_set.end());
	pos_set.insert(ok_set.begin(), ok_set.end());

	return compute_ap(pos_set, junk_set, ranked_list, Mat());
}

static Mat computeQueryAP(const Mat &dists, const QStringList &images, const QStringList &queryFileNames, int maxResults = -1)
{
	Mat sorted;
	sortIdx(dists, sorted, CV_SORT_EVERY_COLUMN | DIST_SORT_ORDER);

	if (maxResults == -1)
		maxResults = images.size();
	Mat AP(sorted.cols, 1, CV_32F);
	for (int j = 0; j < sorted.cols; j++) {
		QFile f("/tmp/ranked_list.txt");
		f.open(QIODevice::WriteOnly);
		for (int i = 0; i < maxResults; i++)
			f.write(QString(images[sorted.at<int>(i, j)]).remove(".jpg").split("/", QString::SkipEmptyParts).last().append("\n").toUtf8());
		f.close();

		float ap = compute_ap_main(qPrintable(QString(queryFileNames[j]).split("_query.").first()),
						"/tmp/ranked_list.txt");
		AP.at<float>(j) = ap;
	}
	return AP;
}

OxfordRetrieval::OxfordRetrieval(QObject *parent) :
	QObject(parent)
{
	//int K = 1000000;
	//base: /home/amenmd/myfs/tasks/hilal_tez/dataset/oxford
	//dict file name: "data/myoxforddict2.bin"
}

/**
 * @brief OxfordRetrieval::convertFeatures
 *
 * Converts given features to our local representation.
 */
void OxfordRetrieval::convertFeatures()
{
	DatasetManager dm;
	dm.addDataset("oxford", "/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/oxbuild_images/");
	dm.convertOxfordFeatures("/home/amenmd/myfs/tasks/hilal_tez/dataset/oxford/");
}

/**
 * @brief OxfordRetrieval::createFeatures
 * @param base Base folder containing dataset files.
 * @param K Dictionary size.
 * @param useHessianAffine If false, function calculates features, otherwise uses hessian affine features present.
 * @param dictFileName Dictionary file name to save clustered dictionary.
 *
 * This function does following for all images in the dataset:
 *
 *		- Detects image keypoints and saves into file name with suffx *.kpts.
 *		- Calculates features and saves into file name with suffix *.bin.
 *		- Finds closest point to (created) visual dictionary and saves into a file with suffix *.ids.
 *
 * It also does the following:
 *		- Clusters a subset of all features into a visual dictionary.
 *
 * If you want to use Hessian affine features created with 'Perdoch'[1] set 'useHessianAffine' to 'true'.
 *
 * [1] https://github.com/perdoch/hesaff
 */
void OxfordRetrieval::createFeatures(const QString &base, int K, bool useHessianAffine, const QString &dictFileName)
{
	QString ftPaths = base + "/oxford_features/";
	DatasetManager dm;
	dm.addDataset("oxford", base + "/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	Mat clusterData(0, 128, CV_32F);
	//#pragma omp parallel for
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << i << images.size();
		QFileInfo fi(images[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		if (!QFile::exists(prefix + ".bin")) {
			vector<KeyPoint> kpts;
			Mat fts;
			if (!useHessianAffine) {
				Mat im = OpenCV::loadImage(images[i]);
				kpts = Pyramids::extractKeypoints(im);
				fts = Pyramids::computeFeatures(im, kpts);
			} else {
				QString ffile = QString(images[i]).append(".hesaff.sift");
				QStringList lines = Common::importText(ffile);
				for (int j = 2; j < lines.size(); j++) {
					if (lines[j].trimmed().isEmpty())
						continue;
					QStringList vals = lines[j].split(" ");
					KeyPoint kpt;
					kpt.pt.x = vals[0].toFloat();
					kpt.pt.y = vals[1].toFloat();
					kpts.push_back(kpt);
					Mat f(1, 128, CV_32F);
					for (int k = 0; k < 128; k++)
						f.at<float>(k) = vals[5 + k].toFloat();
					fts.push_back(f);
				}
			}

			OpenCV::exportKeyPoints(prefix + ".kpts", kpts);
			OpenCV::exportMatrix(prefix + ".bin", fts);
			/* for dictionary */
			clusterData.push_back(OpenCV::subSampleRandom(fts, 1000));
		}
	}


	ffDebug() << "clustering dictionary";
	Mat dict = clusterData;
	if (clusterData.rows > K)
		dict = Pyramids::clusterFeatures(clusterData, K);
	OpenCV::exportMatrix(dictFileName, dict);

	clusterData = Mat();
	/* now calculate id's */
	Pyramids pyr;
	pyr.setDict(dict);
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << "id calc" << i << images.size();
		QFileInfo fi(images[i]);
		QString prefix = QString("%1/%2").arg(ftPaths).arg(fi.baseName());
		//vector<KeyPoint> kpts = OpenCV::importKeyPoints(prefix + ".kpts");
		Mat fts = OpenCV::importMatrix(prefix + ".bin");
		std::vector<DMatch> matches = pyr.matchFeatures(fts);
		Mat ids(fts.rows, 1, CV_32S);
		for (uint i = 0; i < matches.size(); i++) {
			int idx = matches[i].trainIdx;
			ids.at<int>(i, 0) = idx;
		}
		OpenCV::exportMatrix(prefix + ".ids", ids);
	}
}

/**
 * @brief OxfordRetrieval::createPyramids
 * @param base Base folder containing dataset files.
 * @param dictFileName Dictionary file containing visual dictionary.
 * @param dfFileName Document frequencies file to export.
 *
 * This function creates image BoW descriptors from previously created
 * dataset files. In addition to BoW descriptors, this function also
 * creates document frequencies for each visual word.
 *
 * \sa createFeatures()
 */
void OxfordRetrieval::createPyramids(const QString &base, const QString &dictFileName, const QString &dfFileName)
{
	Mat dict = OpenCV::importMatrix(dictFileName);
	DatasetManager dm;
	dm.addDataset("oxford", base + "/oxbuild_images/");
	int K = dict.rows;
	QStringList images = dm.dataSetImages("oxford");
	Mat df = Mat::zeros(1, K, CV_32S);
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << "making pyramid" << i << images.size();
		QString prefix = images[i].split(".").first().replace("oxbuild_images/", "oxford_features/");
		Mat py = Mat::zeros(1, K, CV_32F);
		Mat ids = OpenCV::importMatrix(prefix + ".ids");
		for (int j = 0; j < ids.rows; j++)
			py.at<float>(ids.at<uint>(j)) += 1;
		OpenCV::exportMatrix(prefix + "_pyr.bin", py);
		//OpenCV::exportMatrixTxt(prefix + "_pyr.txt", py);

		/* df computation */
		Mat tcont = Mat::zeros(1, py.cols, CV_32S);
		for (int j = 0; j < py.cols; j++)
			if (py.at<float>(0, j) > 0)
				tcont.at<int>(0, j) = 1;
		for (int j = 0; j < py.cols; j++)
			if (tcont.at<int>(0, j))
				df.at<int>(0, j) += 1;
	}

	OpenCV::exportMatrix(dfFileName, df);
}

/**
 * @brief OxfordRetrieval::createInvertedIndex
 * @param base Base folder containing dataset files.
 * @param dfFileName Document frequencies file to use.
 * @param outputFolder Output folder to save index files.
 *
 * This function creates inverted index for the dataset. It uses previously created document frequencies as well
 * as image BoW descriptors and then applies idf weighting to each image descriptor. It creates 2 files reflecting
 * the used normalization scheme. This permits use of of L1 or L2 norming with the same dataset, though one should
 * adjust normalize() definition to use one of the normalizing functions.
 *
 * \sa createPyramids(), normalize()
 */
void OxfordRetrieval::createInvertedIndex(const QString &base, const QString &dfFileName, const QString &outputFolder)
{
	DatasetManager dm;
	dm.addDataset("oxford", base + "/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");
	vector<vector<int> > iidx;
	vector<vector<precType> > iidx2;
	Mat df = OpenCV::importMatrix(dfFileName);
	int n = 0;
	for (int i = 0; i < images.size(); i++) {
		ffDebug() << i << images.size();
		QString prefix = images[i].split(".").first().replace("oxbuild_images/", "oxford_features/");
		Mat py = OpenCV::importMatrix(prefix + "pyr.bin");
		Mat pyp = postProcBow(py, df, images.size());
		if (OpenCV::getL1Norm(pyp) == 1)
			n = 1;
		if (OpenCV::getL2Norm(pyp) == 1)
			n = 2;
		if (!iidx.size()) {
			for (int j = 0; j < py.cols; j++) {
				iidx.push_back(vector<int>());
				iidx2.push_back(vector<precType>());
			}
		}
		for (int j = 0; j < py.cols; j++) {
			if (py.at<float>(j) > 0) {
				iidx[j].push_back(i);
				iidx2[j].push_back(pyp.at<float>(j));
			}
		}
	}
	assert(n != 0);
	if (n == 1) {
		OpenCV::exportVector2(outputFolder + "/ox_iidx_l1.bin", iidx);
		OpenCV::exportVector2f(outputFolder + "/ox_iidx2_l1.bin", iidx2);
	} else if (n == 2) {
		OpenCV::exportVector2(outputFolder + "/ox_iidx_l2.bin", iidx);
		OpenCV::exportVector2f(outputFolder + "/ox_iidx2_l2.bin", iidx2);
	}
}

/**
 * @brief OxfordRetrieval::runAllQueries
 * @param base Base folder containing dataset files.
 * @param dfFileName Document frequencies file to use.
 * @param invertedIndexFolder Folder containing previously created inverted index files.
 *
 * This function applies all queries in the database and prints mAP of all queries.
 *
 * \sa createInvertedIndex()
 */
void OxfordRetrieval::runAllQueries(const QString &base, const QString &dfFileName, const QString &invertedIndexFolder)
{
	distMetric normMet = DISTM_L2;
	distMetric dmet = DISTM_HELLINGER2;
	DatasetManager dm;
	dm.addDataset("oxford", base + "/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");

	/* import idf vector */
	Mat df = OpenCV::importMatrix(dfFileName);
	vector<vector<int> > iidx;
	vector<vector<precType> > iidx2;
	if (normMet == DISTM_L1) {
		iidx = OpenCV::importVector2(invertedIndexFolder +"/ox_iidx_l1.bin");
		iidx2 = OpenCV::importVector2f(invertedIndexFolder + "/ox_iidx2_l1.bin");
	} else if (normMet == DISTM_L2) {
		iidx = OpenCV::importVector2(invertedIndexFolder + "/ox_iidx_l2.bin");
		iidx2 = OpenCV::importVector2f(invertedIndexFolder + "/ox_iidx2_l2.bin");
	}

	/* parse query info */
	QString ftPaths = base + "/oxford_features/";

	vector<Mat> qIds, qbes;
	QStringList queries = Common::listDir(base + "/gt_files_170407/", "txt");
	QStringList queryFileNames;
	vector<vector<KeyPoint> > keypoints;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		vector<KeyPoint> kpts;
		Mat ids = getQueryIDs(ftPaths, q, kpts);
		qIds.push_back(ids);
		/* calculate query pyramid, i.e. QBE */
		Mat py = getBow(ids, df, images.size(), normMet);
		qbes.push_back(py);
		queryFileNames << q;
		keypoints.push_back(kpts);
		ffDebug() << q << ids.rows << py.cols << kpts.size();
	}

	ffDebug() << "querying";
	/* real operation */
	Mat dists;
	if (iidx.size())
		dists = getQueryDists<precType>(qbes, images, ftPaths, df, iidx, iidx2, dmet);
	else
		dists = getQueryDists(qbes, images, ftPaths, df);

	Mat APs = computeQueryAP(dists, images, queryFileNames);
	ffDebug() << mean(APs)[0];
}

void OxfordRetrieval::runSingleQuery(const QString &base, const QString &dfFileName, const QString &invertedIndexFolder)
{
	distMetric normMet = DISTM_L2;
	distMetric dmet = DISTM_L2;
	DatasetManager dm;
	dm.addDataset("oxford", base + "/oxbuild_images/");
	QStringList images = dm.dataSetImages("oxford");

	QString ftPaths = base + "/oxford_features/";
	/* import idf vector */
	Mat df = OpenCV::importMatrix(dfFileName);
	vector<vector<int> > iidx;
	vector<vector<precType> > iidx2;
	if (normMet == DISTM_L1) {
		iidx = OpenCV::importVector2(invertedIndexFolder + "/ox_iidx_l1.bin");
		iidx2 = OpenCV::importVector2f(invertedIndexFolder +"/ox_iidx2_l1.bin");
	} else if (normMet == DISTM_L2) {
		iidx = OpenCV::importVector2(invertedIndexFolder + "/ox_iidx_l2.bin");
		iidx2 = OpenCV::importVector2f(invertedIndexFolder + "/ox_iidx2_l2.bin");
	}

	vector<Mat> qIds, qbes;
	QStringList queries = Common::listDir(base + "/gt_files_170407/", "txt");
	QStringList queryFileNames;
	vector<vector<KeyPoint> > keypoints;
	foreach (const QString &q, queries) {
		if (!q.endsWith("_query.txt"))
			continue;
		vector<KeyPoint> kpts;
		Mat ids = getQueryIDs(ftPaths, q, kpts);
		qIds.push_back(ids);
		/* calculate query pyramid, i.e. QBE */
		Mat py = getBow(ids, df, images.size(), normMet);
		qbes.push_back(py);
		queryFileNames << q;
		keypoints.push_back(kpts);
		ffDebug() << q << ids.rows << py.cols << kpts.size();
	}

	Mat dists = getQueryDists<precType>(qbes, images, ftPaths, df, iidx, iidx2, dmet);
}

#if 0 //temporary codes containing various trials

------------------------------------------------------------------------------------------------
Mat cdists1 = OpenCV::importMatrix("data/ox_dists1.bin");
Mat cdists2 = OpenCV::importMatrix("data/ox_dists2.bin");
for (int i = 0; i < cdists1.rows; i++)
	qDebug() << cdists1.col(0).at<float>(i) << cdists2.col(0).at<float>(i);distMetric
------------------------------------------------------------------------------------------------


return;

#endif

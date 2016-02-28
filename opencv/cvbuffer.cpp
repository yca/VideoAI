#include "cvbuffer.h"

using namespace cv;
using namespace std;

CVBuffer::CVBuffer(const Mat &m) :
	RawBuffer()
{
	CVBufferData *dd = new CVBufferData;
	dd->mat = m;
	d = dd;
	size_t size = m.total() * m.elemSize();
	setRefData("application/cv-mat", m.data, size);
}

CVBuffer::CVBuffer(const std::vector<KeyPoint> &vec) :
	RawBuffer()
{
	CVBufferData *dd = new CVBufferData;
	dd->vec = vec;
	d = dd;
	setRefData("application/cv-kpts", (void *)vec.data(), vec.size());
}

CVBuffer::CVBuffer(const std::vector<Mat> &vec)
{
	CVBufferData *dd = new CVBufferData;
	dd->vec2 = vec;
	d = dd;
	setRefData("application/cv-matv", (void *)vec.data(), vec.size());
}

const Mat CVBuffer::getReferenceMat() const
{
	CVBufferData *dd = (CVBufferData *)d.data();
	return dd->mat;
}

void CVBuffer::setReferenceMat(const Mat &m)
{
	CVBufferData *dd = (CVBufferData *)d.data();
	dd->mat = m;
}

std::vector<KeyPoint> &CVBuffer::getKeypoints() const
{
	CVBufferData *dd = (CVBufferData *)d.data();
	return dd->vec;
}

std::vector<Mat> &CVBuffer::getVector() const
{
	CVBufferData *dd = (CVBufferData *)d.data();
	return dd->vec2;
}

RawBuffer CVBuffer::createNewBuffer(const std::vector<cv::KeyPoint> &kpts, const Mat &m, const RawBuffer &buf)
{
	CVBuffer c2(kpts);
	c2.setReferenceMat(m);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	c2.pars()->videoWidth = buf.constPars()->videoWidth;
	c2.pars()->videoHeight = buf.constPars()->videoHeight;
	return c2;
}

RawBuffer CVBuffer::createNewBuffer(const std::vector<Mat> &fts, const RawBuffer &buf)
{
	CVBuffer c2(fts);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	c2.pars()->videoWidth = buf.constPars()->videoWidth;
	c2.pars()->videoHeight = buf.constPars()->videoHeight;
	return c2;
}

CVBuffer CVBuffer::createNewBuffer(const Mat &m, const RawBuffer &buf)
{
	CVBuffer c2(m);
	c2.pars()->metaData = buf.constPars()->metaData;
	c2.pars()->streamBufferNo = buf.constPars()->streamBufferNo;
	c2.pars()->videoWidth = buf.constPars()->videoWidth;
	c2.pars()->videoHeight = buf.constPars()->videoHeight;
	return c2;
}

CVBufferData::~CVBufferData()
{

}

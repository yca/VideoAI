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
	setRefData("applicaiton/cv-kpts", (void *)vec.data(), vec.size());
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

CVBufferData::~CVBufferData()
{

}

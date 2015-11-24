#ifndef CVBUFFER_H
#define CVBUFFER_H

#include <lmm/rawbuffer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

class CVBufferData : public RawBufferData
{
public:
	CVBufferData()
		: RawBufferData()
	{
		bufferOwner = true;
	}
	CVBufferData(const CVBufferData &other)
		: RawBufferData(other)
	{
		bufferOwner = other.bufferOwner;
		mat = other.mat;
	}
	~CVBufferData();
	cv::Mat mat;
	std::vector<cv::KeyPoint> vec;
	bool bufferOwner;
};

class CVBuffer : public RawBuffer
{
public:
	explicit CVBuffer(const cv::Mat &m);
	explicit CVBuffer(const std::vector<cv::KeyPoint> &vec);
	const cv::Mat getReferenceMat() const;
	void setReferenceMat(const cv::Mat &m);
	std::vector<cv::KeyPoint> & getKeypoints() const;
signals:

public slots:

};

#endif // CVBUFFER_H

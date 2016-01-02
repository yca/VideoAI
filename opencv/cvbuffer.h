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
	std::vector<cv::Mat> vec2;
	bool bufferOwner;
};

class CVBuffer : public RawBuffer
{
public:
	explicit CVBuffer(const cv::Mat &m);
	explicit CVBuffer(const std::vector<cv::KeyPoint> &vec);
	explicit CVBuffer(const std::vector<cv::Mat> &vec);
	const cv::Mat getReferenceMat() const;
	void setReferenceMat(const cv::Mat &m);
	std::vector<cv::KeyPoint> & getKeypoints() const;
	std::vector<cv::Mat> & getVector() const;
signals:

public slots:

};

#endif // CVBUFFER_H

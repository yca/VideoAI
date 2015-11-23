#ifndef CLASSIFICATIONPIPELINE_H
#define CLASSIFICATIONPIPELINE_H

#ifdef HAVE_LMM

#include <lmm/pipeline/pipelinemanager.h>

#include <errno.h>

class DatasetManager;

template <class T>
class OpElement : public BaseLmmElement
{
public:
	typedef int (T::*elementOp)(const RawBuffer &);
	OpElement(T *parent, elementOp op)
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
	}
	virtual int processBuffer(const RawBuffer &buf)
	{
		return (enc->*mfunc)(buf);
	}

private:
	T *enc;
	elementOp mfunc;
};

template <class T>
class OpSrcElement : public BaseLmmElement
{
public:
	typedef const RawBuffer (T::*elementOp)();
	OpSrcElement(T *parent, elementOp op)
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
	}
	virtual int processBuffer(const RawBuffer &) { return 0; }
	int processBlocking(int ch)
	{
		RawBuffer buf = (enc->*mfunc)();
		if (buf.isEOF())
			return -ENOENT;
		return newOutputBuffer(ch, buf);
	}

private:
	T *enc;
	elementOp mfunc;
};

class SourceLmmElement : public BaseLmmElement
{
	Q_OBJECT
public:
	SourceLmmElement(QObject *parent = NULL)
		: BaseLmmElement(parent)
	{}

	int processBlocking(int ch);
	virtual int processBuffer(const RawBuffer &) { return 0; }
};

class ClassificationPipeline : public PipelineManager
{
	Q_OBJECT
public:
	explicit ClassificationPipeline(QObject *parent = 0);

	virtual const RawBuffer readNextImage();
	virtual int detectKeypoints(const RawBuffer &buf);
signals:

public slots:
protected:
	DatasetManager *dm;

	QStringList images;
};

#endif

#endif // CLASSIFICATIONPIPELINE_H

#ifndef LMMELEMENTS_H
#define LMMELEMENTS_H

#include <lmm/baselmmelement.h>

template <class T>
class OpElement : public BaseLmmElement
{
public:
	typedef RawBuffer (T::*elementOp)(const RawBuffer &, int);
	OpElement(T *parent, elementOp op, int priv, QString objName = "BaseLmmElement")
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
		this->priv = priv;
		setObjectName(QString("%1%2").arg(objName).arg(priv));
	}
	virtual int processBuffer(const RawBuffer &buf)
	{
		RawBuffer buf2 = (enc->*mfunc)(buf, priv);
		if (buf.getMimeType() == "application/empty")
			return 0;
		while (getOutputQueue(0)->getBufferCount() > 500)
			usleep(1000 * 100);
		return newOutputBuffer(0, buf2);
	}

private:
	T *enc;
	elementOp mfunc;
	int priv;
};

template <class T>
class OpSrcElement : public BaseLmmElement
{
public:
	typedef const RawBuffer (T::*elementOp)();
	OpSrcElement(T *parent, elementOp op, QString objName = "BaseLmmElement")
		: BaseLmmElement(parent)
	{
		enc = parent;
		mfunc = op;
		setObjectName(objName);
	}
	virtual int processBuffer(const RawBuffer &) { return 0; }
	int processBlocking(int ch)
	{
		RawBuffer buf = (enc->*mfunc)();
		if (buf.getMimeType() == "application/empty")
			return 0;
		while (getOutputQueue(0)->getBufferCount() > 500)
			usleep(1000 * 100);
		return newOutputBuffer(ch, buf);
	}

private:
	T *enc;
	elementOp mfunc;
};

#endif // LMMELEMENTS_H

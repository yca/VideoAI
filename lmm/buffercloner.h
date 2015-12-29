#ifndef BUFFERCLONER_H
#define BUFFERCLONER_H

#include <QObject>

#include <lmm/baselmmelement.h>

class BufferCloner : public BaseLmmElement
{
	Q_OBJECT
public:
	BufferCloner();
protected:
	virtual int processBuffer(const RawBuffer &buf);
};

#endif // BUFFERCLONER_H

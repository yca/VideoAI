#include "buffercloner.h"

BufferCloner::BufferCloner()
{

}

int BufferCloner::processBuffer(const RawBuffer &buf)
{
	int err = newOutputBuffer(0, buf);
	if (err)
		return err;
	return newOutputBuffer(1, buf);
}


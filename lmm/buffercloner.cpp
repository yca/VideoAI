#include "buffercloner.h"

#include <unistd.h>

BufferCloner::BufferCloner()
{

}

int BufferCloner::processBuffer(const RawBuffer &buf)
{
	if (getOutputQueue(0)->getBufferCount() > 500
			|| getOutputQueue(1)->getBufferCount() > 500)
		usleep(1000 * 100);
	int err = newOutputBuffer(0, buf);
	if (err)
		return err;
	return newOutputBuffer(1, buf);
}


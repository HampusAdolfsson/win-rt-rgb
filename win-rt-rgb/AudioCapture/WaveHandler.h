#pragma once

#include "Windows.h"

/**
*	Receives and handles audio captured as float (i.e. WAVE_FORMAT_IEEE_FLOAT) buffers
*/
class WaveHandler
{
public:
	WaveHandler();
	virtual ~WaveHandler() = 0;

	/**
	*	Called when a buffer is ready, to let this handler do something with it
	*	@param samples The buffer
	*	@param nFrames The number of frames in the buffer (a frame consists of one sample per channel)
	*/
	virtual void receiveBuffer(float* samples, unsigned int nFrames) = 0;

	void setFormat(const unsigned int& samplesPerSec, const unsigned int& nChannels);

protected:
	int sampleRate;
	int nChannels;
};

#pragma once

#include "Windows.h"
#include <vector>

/**
*	Receives and handles audio captured as float (i.e. WAVE_FORMAT_IEEE_FLOAT) buffers
*/
class WaveHandler
{
public:
	WaveHandler(unsigned int buffersPerSecond);
	virtual ~WaveHandler() = 0;

	/**
	*	Called when a buffer of samples is ready, to let this handler do something with it
	*	@param samples The buffer
	*	@param nFrames The number of frames in the buffer (a frame consists of one sample per channel)
	*/
	void receiveSamples(float* samples, unsigned int nFrames);

	void setFormat(unsigned int samplesPerSec, unsigned int nChannels);

	virtual void handleWaveData(float* buffer, unsigned int nFrames) = 0;


protected:
	int sampleRate;
	int nChannels;

private:
	unsigned int buffersPerSecond;
	unsigned int bufferPosition;
	unsigned int activeBuffer;
	std::vector<float> buffers[2];
};
#pragma once
#include "AudioHandler.h"
#include <vector>

/**
*	Receives audio captured as float (i.e. WAVE_FORMAT_IEEE_FLOAT) arrays,
*	and handles the data in some way when enough samples have been buffered.
*/
class AudioSink
{
public:
	/**
	*	Creates a new audio sink
	*	@param audioHandler handler to call when an audio buffer is ready
	*	@param buffersPerSecond denotes how often the handler should be called. This, together with the sample rate,
	*		determines the number of samples to receive before the handler is called.
	*/
	AudioSink(std::unique_ptr<AudioHandler> audioHandler, unsigned int buffersPerSecond);

	/**
	*	Adds samples to the buffer.
	*	@param samples An array containing the samples
	*	@param nFrames The number of frames in the array (a frame consists of one sample per channel)
	*/
	void receiveSamples(float* samples, unsigned int nFrames);

	void setFormat(unsigned int samplesPerSec, unsigned int nChannels);

private:
	int sampleRate;
	int nChannels;

	unsigned int buffersPerSecond;
	unsigned int activeBuffer;
	unsigned int bufferPosition;
	std::vector<float> buffers[2];
	std::unique_ptr<AudioHandler> audioHandler;
};

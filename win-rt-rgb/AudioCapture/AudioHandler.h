#pragma once
#include <memory>

class AudioHandler
{
public:
	/**
	*	Called when a buffer is ready, to let this handler do something with it.
	*	Each buffer contains nFrames*nChannels samples.
	*	@param buffer The buffer containing the samples
	*	@param nFrames The number of frames in the buffer (a frame consists of one sample per channel)
	*/
	virtual void handleWaveData(float* buffer, unsigned int nFrames) = 0;
    virtual ~AudioHandler() = 0 {};

	void setFormat(unsigned int nChannels, unsigned int sampleRate);

protected:
	unsigned int nChannels;
	unsigned int sampleRate;
};

// class AudioHandlerFactory
// {
// public:
// 	virtual std::unique_ptr<AudioHandler> createAudioHandler(unsigned int nChannels, unsigned int sampleRate) = 0;
// };
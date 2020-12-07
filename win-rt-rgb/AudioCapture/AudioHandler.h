#pragma once
#include <memory>

class AudioHandler
{
public:
	/**
	*	Called when a buffer is ready, to let this handler do something with it.
	*	Each buffer contains nFrames*nChannels samples.
	*	@param buffer The buffer containing the samples
	*/
	virtual void handleWaveData(float* buffer) = 0;
    virtual ~AudioHandler() = 0 {};
};

class AudioHandlerFactory
{
public:
	virtual std::unique_ptr<AudioHandler> createAudioHandler(size_t bufferSize, unsigned int nChannels) = 0;
};
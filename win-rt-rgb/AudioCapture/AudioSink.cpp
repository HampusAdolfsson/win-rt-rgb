#include "AudioSink.h"
#include <cassert>

using namespace AudioCapture;

AudioSink::AudioSink(unsigned int buffersPerSecond, std::unique_ptr<AudioHandlerFactory> audioHandlerFactory)
: buffersPerSecond(buffersPerSecond),
sampleRate(0),
nChannels(0),
activeBuffer(0),
bufferPosition(0),
audioHandlerFactory(std::move(audioHandlerFactory)),
audioHandler(nullptr)
{
	assert(this->audioHandlerFactory);
}

void AudioSink::receiveSamples(float* samples, unsigned int nFrames)
{
	assert(nChannels && sampleRate);
	size_t nSamples = nFrames * nChannels;
	if (bufferPosition + nSamples >= buffers[0].size())
	{
		assert(this->audioHandler);
		size_t remaining = buffers[0].size() - bufferPosition;
		memcpy(&buffers[activeBuffer][bufferPosition], samples, remaining * sizeof(*samples));
		audioHandler->handleWaveData(buffers[activeBuffer].data());

		samples += remaining;
		nSamples -= remaining;
		bufferPosition = 0;
		activeBuffer = (activeBuffer + 1) % 2;
	}
	assert(bufferPosition + nSamples < sampleRate * nChannels / buffersPerSecond);
	memcpy(&buffers[activeBuffer][bufferPosition], samples, nSamples * sizeof(*samples));
	bufferPosition += nSamples;
}

void AudioSink::receiveEmptySamples(unsigned int nFrames)
{
	assert(nChannels && sampleRate);
	size_t nSamples = nFrames * nChannels;
	if (bufferPosition + nSamples >= buffers[0].size())
	{
		assert(this->audioHandler);
		size_t remaining = buffers[0].size() - bufferPosition;
		memset(&buffers[activeBuffer][bufferPosition], 0, remaining * sizeof(buffers[0][0]));
		audioHandler->handleWaveData(buffers[activeBuffer].data());

		nSamples -= remaining;
		bufferPosition = 0;
		activeBuffer = (activeBuffer + 1) % 2;
	}
	assert(bufferPosition + nSamples < sampleRate * nChannels / buffersPerSecond);
	memset(&buffers[activeBuffer][bufferPosition], 0, nSamples * sizeof(buffers[0][0]));
	bufferPosition += nSamples;
}

void AudioSink::setFormat(unsigned int samplesPerSec, unsigned int nChannels)
{
	this->sampleRate = samplesPerSec;
	this->nChannels = nChannels;
	buffers[0] = std::vector<float>(sampleRate * nChannels / buffersPerSecond);
	buffers[1] = std::vector<float>(sampleRate * nChannels / buffersPerSecond);
	audioHandler = audioHandlerFactory->createAudioHandler(sampleRate * nChannels / buffersPerSecond, nChannels);
}
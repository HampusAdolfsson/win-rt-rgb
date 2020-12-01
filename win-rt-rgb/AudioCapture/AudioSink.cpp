#include "AudioSink.h"
#include <cassert>

AudioSink::AudioSink(std::unique_ptr<AudioHandler> audioHandler, unsigned int buffersPerSecond)
: buffersPerSecond(buffersPerSecond),
sampleRate(0),
nChannels(0),
activeBuffer(0),
bufferPosition(0),
audioHandler(std::move(audioHandler))
{
	assert(this->audioHandler);
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
		audioHandler->handleWaveData(buffers[activeBuffer].data(), buffers[activeBuffer].size() / nChannels);

		samples += remaining;
		nSamples -= remaining;
		bufferPosition = 0;
		activeBuffer = (activeBuffer + 1) % 2;
	}
	assert(bufferPosition + nSamples < sampleRate * nChannels / buffersPerSecond);
	memcpy(&buffers[activeBuffer][bufferPosition], samples, nSamples * sizeof(*samples));
	bufferPosition += nSamples;
}

void AudioSink::setFormat(unsigned int samplesPerSec, unsigned int nChannels)
{
	audioHandler->setFormat(nChannels, samplesPerSec);
	this->sampleRate = samplesPerSec;
	this->nChannels = nChannels;
	buffers[0] = std::vector<float>(sampleRate * nChannels / buffersPerSecond);
	buffers[1] = std::vector<float>(sampleRate * nChannels / buffersPerSecond);
}
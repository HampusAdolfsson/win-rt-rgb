#include "WaveHandler.h"

WaveHandler::WaveHandler(unsigned int buffersPerSecond)
: buffersPerSecond(buffersPerSecond),
sampleRate(0),
nChannels(0),
activeBuffer(0),
bufferPosition(0)
{
}

void WaveHandler::receiveSamples(float* samples, unsigned int nFrames)
{
	size_t nSamples = nFrames * nChannels;
	if (bufferPosition + nSamples >= buffers[activeBuffer].size())
	{
		size_t remaining = buffers[activeBuffer].size() - bufferPosition;
		memcpy(&buffers[activeBuffer][bufferPosition], samples, remaining * sizeof(*samples));
		handleWaveData(buffers[activeBuffer].data(), buffers[activeBuffer].size() / nChannels);

		samples += remaining;
		nSamples -= remaining;
		bufferPosition = 0;
		activeBuffer = (activeBuffer + 1) % 2;
	}
	memcpy(&buffers[activeBuffer][bufferPosition], samples, nSamples * sizeof(*samples));
	bufferPosition += nSamples;
}

void WaveHandler::setFormat(unsigned int samplesPerSec, unsigned int nChannels)
{
	this->sampleRate = samplesPerSec;
	this->nChannels = nChannels;
	buffers[0] = std::vector<float>(sampleRate * nChannels / buffersPerSecond);
	buffers[1] = std::vector<float>(sampleRate * nChannels / buffersPerSecond);
}

WaveHandler::~WaveHandler()
{
}
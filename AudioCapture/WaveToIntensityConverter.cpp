#include "WaveToIntensityConverter.h"
#include "Logger.h"
#include <cassert>
#include <algorithm>
#include <mmeapi.h>
#include <mmreg.h>

// Size of softening filter. Higher value means more smooth output, but lower responsiveness (more delay).
#define MEAN_ORDER 8
// Determines how fast the max value decays (as the time in seconds it takes for it to decay from max to 0).
// Higher values mean more stable normalization, but it will be slower to adjust to decreased volumes.
#define MAX_VAL_DECAY_TIME 330

WaveToIntensityConverter::WaveToIntensityConverter(std::function<void(const float&)> callback)
	: sum(0),
	maxSum(0),
	meanPrevVals(MEAN_ORDER, 0),
	callback(callback)
{
}
WaveToIntensityConverter::~WaveToIntensityConverter()
{
}

void WaveToIntensityConverter::receiveBuffer(float* samples, unsigned int nFrames) {
	// This code could be improved a lot.
	// Ideally it should have the same output regardless of sampling frequency,
	// and should not be dependent on buffers always having the same size.
	// There may also be better way of handling multi-channel audio.
	unsigned int nSamples = nFrames * nChannels;
	float sampleSum = 0;
	for (size_t i = 0; i < nSamples; i++)
	{
		float sample = samples[i];
		sample = sample < 0 ? -sample : sample;
		sampleSum += sample * sample;
	}
	float mean = sampleSum / nSamples;
	//This is the root mean square of the buffer
	mean = sqrt(mean);

	{
		float newVal = mean / MEAN_ORDER;
		sum += newVal;
		sum -= meanPrevVals.putAndGet(newVal);
	}
	if (sum > maxSum)
	{
		maxSum = sum;
	}
	else
	{
		if (maxSum > 0.001f)
		{
			maxSum -= (1 / float(MAX_VAL_DECAY_TIME)) / (sampleRate / nSamples);
		}
	}
	//outputFilter.put(normalizedRms);
	//roofFilter.put(normalizedRms);

	//float roof = std::max(0.1f, roofFilter.getOutput());
	//return std::min(1.0f, outputFilter.getOutput() / roof);
	LOGINFO("%f, %f", max(0, sum / maxSum), maxSum);
	callback(std::clamp(sum / maxSum, 0.0f, 1.0f));
}
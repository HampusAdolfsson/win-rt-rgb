#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "WaveToIntensityStrategy.h"
#include "Logger.h"

WaveToIntensityStrategy::WaveToIntensityStrategy(const unsigned int& sampleRate)
	: sum(0),
	maxSum(0),
	meanPrevVals(MEAN_ORDER, 0),
	sampleRate(sampleRate)
{
}

float WaveToIntensityStrategy::getIntensity(const char* buffer, const size_t& bufSiz, const size_t& sampleSize)
{
	unsigned int sampleSum = 0;
	for (size_t i = 0; i < bufSiz; i += sampleSize)
	{
		long sample = *((int16_t*)(buffer + i));
		sample = sample < 0 ? -sample : sample;
		sampleSum += sample * sample;
	}
	unsigned int nSamples = bufSiz / sampleSize;
	float mean = float(sampleSum) / nSamples;
	mean = sqrt(mean);

	{
		float newVal = float(mean) / MEAN_ORDER;
		sum += newVal;
		sum -= meanPrevVals.putAndGet(newVal);
	}
	if (sum > maxSum)
	{
		maxSum = sum;
	}
	else
	{
		if (maxSum > 50)
		{
			maxSum -= MAX_VAL_DECAY_SPEED * INT16_MAX / (sampleRate / nSamples);
		}
	}

	return  sum / maxSum;
}
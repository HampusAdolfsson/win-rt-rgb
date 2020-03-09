#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "WavetoColorStrategy.h"

WaveToIntensityStrategy::WaveToIntensityStrategy()
	: sum(0),
	prevValsIndex(0)
{
	memset(prevVals, 0, MEAN_ORDER * sizeof(*prevVals));
}

uint8_t WaveToIntensityStrategy::getIntensity(const char *buffer, const size_t& bufSiz, const size_t& sampleSize)
{
	unsigned int mean = 0;
	for (size_t i = 0; i < bufSiz; i += sampleSize)
	{
		long sample = *((int16_t*)(buffer + i));
		sample = sample < 0 ? -sample : sample;
		mean += sample* sample;
	}
	mean /= (bufSiz / sampleSize);
	mean = sqrt(mean);
	sum -= prevVals[prevValsIndex];
	prevVals[prevValsIndex] = mean / MEAN_ORDER;
	sum += prevVals[prevValsIndex];
	prevValsIndex++;
	if (prevValsIndex == MEAN_ORDER) prevValsIndex = 0;
	int normalizedSum = sum * SCALING;
	normalizedSum /= 128; // fit to (0-255)*SCALING range
	return normalizedSum > UINT8_MAX ? UINT8_MAX : normalizedSum;
}
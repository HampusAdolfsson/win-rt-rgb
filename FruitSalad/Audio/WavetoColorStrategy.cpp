#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "WavetoColorStrategy.h"

WavetoColorStrategy::WavetoColorStrategy()
	: rollingAvg(0),
	prevValsIndex(0),
	baseColor({ 0xFF, 0x00, 0x00 })
{
	memset(prevMean, 0, MEAN_ORDER * sizeof(*prevMean));
}

RgbColor WavetoColorStrategy::getColor(const char *buffer, const size_t& bufSiz, const size_t& sampleSize)
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
	mean *= SCALING;
	rollingAvg -= prevMean[prevValsIndex];
	prevMean[prevValsIndex] = mean / MEAN_ORDER;
	rollingAvg += prevMean[prevValsIndex];
	prevValsIndex++;
	if (prevValsIndex == MEAN_ORDER) prevValsIndex = 0;
	return baseColor * (rollingAvg > UINT16_MAX ? 1 : (float)rollingAvg / UINT16_MAX);
}
#include <cstring>
#include <cmath>
#include "WavetoColorStrategy.h"

	WavetoColorStrategy::WavetoColorStrategy()
	: rollingAvg(0),
	prevValsIndex(0),
	baseColor({ 0xff, 0xff, 0xff })
{
	memset(prevVals, 0, FILTER_SIZE * sizeof(*prevVals));
}

Color WavetoColorStrategy::getColor(const char *buffer, const size_t& bufSiz, const size_t& sampleSize)
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
	rollingAvg -= prevVals[prevValsIndex];
	prevVals[prevValsIndex] = mean / FILTER_SIZE;
	rollingAvg += prevVals[prevValsIndex];
	prevValsIndex++;
	if (prevValsIndex == FILTER_SIZE) prevValsIndex = 0;
	return baseColor * (rollingAvg > UINT16_MAX ? 1 : (float)rollingAvg / UINT16_MAX);
}
#pragma once
#include <time.h>
#include <fstream>

// Size of softening filter. Higher value means more smooth output, but lower responsiveness.
#define MEAN_ORDER 16

#define SCALING 10

/**
*	Gets the intensity of a wave (currently implemented as RMS with an extra rolling average filter).
*/
class WaveToIntensityStrategy
{
	unsigned int sum;
	unsigned int prevVals[MEAN_ORDER];
	size_t		 prevValsIndex;

public:
	WaveToIntensityStrategy();
	uint8_t getIntensity(const char *buffer, const size_t& bufSiz, const size_t& sampleSize);
};

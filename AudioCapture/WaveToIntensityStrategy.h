#pragma once
#include <time.h>
#include <fstream>
#include "RingBuffer.h"

// Size of softening filter. Higher value means more smooth output, but lower responsiveness (more delay).
#define MEAN_ORDER 16
// How fast the max value decays (as a fraction of the total range per second).
// Higher values mean more stable normalization, but it will be slower to adjust to decreased volumes.
#define MAX_VAL_DECAY_SPEED 0.0036


/**
*	Gets the intensity of a wave (currently implemented as RMS with an extra rolling average filter).
*	The intensity is normalized to a 0-1 range based on the highest (absolute) intensity observed within a recent period of time.
*/
class WaveToIntensityStrategy
{
	float sum;
	float maxSum;
	RingBuffer<float> meanPrevVals;

	const unsigned int sampleRate;

public:
	WaveToIntensityStrategy(const unsigned int& sampleRate);
	float getIntensity(const char* buffer, const size_t& bufSiz, const size_t& sampleSize);
};
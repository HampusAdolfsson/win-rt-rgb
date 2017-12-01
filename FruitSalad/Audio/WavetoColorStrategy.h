#pragma once
#include "Color.h"

// Size of softening filter. Higher value means more smooth output, but lower responsiveness.
#define FILTER_SIZE 8

#define SCALING 50

/**
*	Defines a strategy to convert waveform samples into a representative color
*/
class WavetoColorStrategy
{
	unsigned int rollingAvg;
	unsigned int prevVals[FILTER_SIZE];
	size_t		 prevValsIndex;
	Color		 baseColor;

public:
	WavetoColorStrategy();
	Color getColor(char *buffer, size_t bufSiz, size_t sampleSize);
};

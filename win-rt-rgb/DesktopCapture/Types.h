#pragma once
#include "Color.h"
#include <functional>

typedef struct
{
	unsigned int left, top, width, height;
} Rect;

struct SamplingSpecification
{
	/**
	*	The number of regions to sample from. This is the number of colors the sampler
	*	will generate from each frame.
	*/
	unsigned int numberOfRegions;

	/**
	*	A saturation amount to add to the calculated color for each region.
	*	1.0f would mean a 100% increase (always fully saturated), and -1.0f a 100% decrease (e.g. black and white).
	*/
	float saturationAdjustment;

	/**
	*	The amount of blur to apply to the output. 0 means no blurring. Note that using too high values (e.g. close to half
	*	the number of regions) will also remove the blurring (the blurring algorithm needs to be improved).
	*/
	unsigned int blurRadius;

	/**
	*	By default the leftmost region corresponds is output first in the resulting colors array, and so on.
	*	Setting this bool flips it so that the leftmost part corresponds to the last LED (and so on).
	*/
	bool flipHorizontally;
};

/**
 *	Callback for when colors have been sampled from a desktop frame. The first parameter is the index of
 *	the SamplingSpecification used to generate the colors (as passed into the constructor of DesktopColorSampler).
 *	The second parameter is an array with the colors itself (the size of the array matches the number
 *	of groups in the specification).
 */
typedef std::function<void(unsigned int, RgbColor*)> DesktopSamplingCallback;


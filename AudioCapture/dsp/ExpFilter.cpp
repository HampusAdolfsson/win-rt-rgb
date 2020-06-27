#include "ExpFilter.h"

ExpFilter::ExpFilter(const float &smoothing) : smoothing(smoothing), output(0)
{
}

void ExpFilter::put(const float &value)
{
	output = smoothing * value + (1 - smoothing) * output;
}
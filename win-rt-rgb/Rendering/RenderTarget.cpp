#include <cassert>
#include <cstdlib>
#include "RenderTarget.h"

RenderTarget::RenderTarget(const unsigned int& size)
	: size(size),
	colors(size, {0,0,0})
{
}

void RenderTarget::drawRange(const unsigned int& startIndex, const unsigned int& length, const RgbColor* toDraw)
{
	memcpy(colors.data() + startIndex, toDraw, length * sizeof(*toDraw));
}

void RenderTarget::beginFrame()
{
	memset(colors.data(), 0, size);
}

void RenderTarget::setIntensity(const float& intensity)
{
	for (size_t i = 0; i < size; i++) // TODO: consider parallelism
	{
		colors[i] = colors[i] * intensity;
	}
}

void RenderTarget::cloneFrom(const RenderTarget& other)
{
	assert(size == other.size);
	memcpy(colors.data(), other.colors.data(), sizeof(RgbColor) * size);
}
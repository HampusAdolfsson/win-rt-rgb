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
	memset(colors.data(), false, size);
}

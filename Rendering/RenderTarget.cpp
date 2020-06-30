#include "RenderTarget.h"

RenderTarget::RenderTarget(unsigned int& size)
	: size(size),
	colors(size, {0,0,0})
{
}

void RenderTarget::drawRange(unsigned int& startIndex, unsigned int& length, const RgbColor* toDraw)
{
	memcpy(colors.data(), toDraw, length * sizeof(*toDraw));
}

void RenderTarget::beginFrame()
{
	memset(colors.data(), false, size);
}

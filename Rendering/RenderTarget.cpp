#include "RenderTarget.h"

RenderTarget::RenderTarget(unsigned int& size)
	: size(size),
	colors(size, {0,0,0}),
	dirty(size, true)
{
}

void RenderTarget::drawRange(unsigned int& startIndex, unsigned int& length, const RgbColor* toDraw)
{
	for (unsigned int i = startIndex; i < startIndex + length; i++)
	{
		dirty[i] = dirty[i] || colors[i] != toDraw[i - startIndex];
	}
	memcpy(colors.data(), toDraw, length * sizeof(*toDraw));
}

void RenderTarget::beginFrame()
{
	memset(colors.data(), false, size);
}

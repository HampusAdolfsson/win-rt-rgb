#include "RenderTarget.h"
#include <cassert>
#include <cstdlib>
#ifdef USE_SSE
#include <immintrin.h>
#endif
#include "Logger.h"

using namespace Rendering;

RenderTarget::RenderTarget(const unsigned int& size, std::unique_ptr<MaskingBehaviour> maskBehaviour)
	: size(size),
	maskBehaviour(std::move(maskBehaviour)),
#ifdef USE_SSE
	colors((RgbColor*)(_aligned_malloc((size * sizeof(RgbColor) / sizeof(__m256) + 1) * sizeof(__m256), sizeof(__m256))), &_aligned_free)
#else
	colors(new RgbColor[size])
#endif
{
}

void RenderTarget::drawRange(const unsigned int& startIndex, const unsigned int& length, const RgbColor* toDraw)
{
	if (startIndex + length > size)
	{
		LOGSEVERE("Drawing outside of RenderTarget buffer");
	}
	else
	{
		memcpy(colors.get() + startIndex, toDraw, length * sizeof(*toDraw));
	}
}

void RenderTarget::beginFrame()
{
	memset(colors.get(), 0, size);
}

void RenderTarget::applyAdjustments(float hue, float saturation, float value)
{
	if (hue == .0f && saturation == .0f && value == .0f) return;
	#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		HsvColor hsv = rgbToHsv(colors[i]);
		hsv.hue += hue;
		if (hsv.hue >= 360.0f) hsv.hue -= 360.0f;
		if (hsv.hue < 0.0f) hsv.hue += 360.0f;
		if (hsv.saturation > 0.001f)  // don't change the saturation for greyscale colors, they'll just turn blue
		{
			hsv.saturation = std::min(std::max(hsv.saturation + saturation, 0.0f), 1.0f);
		}
		hsv.value = std::min(std::max(hsv.value + value, 0.0f), 1.0f);
		colors[i] = hsvToRgb(hsv);
	}
}

void RenderTarget::setIntensity(const float& intensity)
{
	maskBehaviour->applyMask(colors.get(), size, intensity);
}

void RenderTarget::cloneFrom(const RenderTarget& other)
{
	assert(size == other.size);
	memcpy(colors.get(), other.colors.get(), sizeof(RgbColor) * size);
}
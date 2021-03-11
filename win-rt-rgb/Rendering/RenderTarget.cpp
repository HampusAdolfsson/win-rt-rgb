#include "RenderTarget.h"
#include <cassert>
#include <cstdlib>
#ifdef USE_SSE
#include <immintrin.h>
#include <new>
#endif
#include "Logger.h"

using namespace Rendering;

RenderTarget::RenderTarget(const unsigned int& size)
	: size(size),
#ifdef USE_SSE
#define PADDING (32/sizeof(float))
	// colors((size / ALIGNMENT + 1) * ALIGNMENT, {0,0,0})
	colors((RgbColor*)(_aligned_malloc((size * sizeof(RgbColor) / sizeof(__m256) + 1) * sizeof(__m256), sizeof(__m256))))
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

void RenderTarget::setIntensity(const float& intensity)
{
#ifdef USE_SSE
	float* colorsVector = (float*) colors.get();
	__m256 intensityVector = _mm256_set1_ps(intensity);
	for (size_t i = 0; i * sizeof(float) < sizeof(RgbColor) * size; i += sizeof(__m256) / sizeof(float))
	{
		__m256 color = _mm256_load_ps(colorsVector + i);
		_mm256_store_ps(colorsVector + i, _mm256_mul_ps(color, intensityVector));
	}
#else
	for (size_t i = 0; i < size; i++)
	{
		colors[i] = colors[i] * intensity;
	}
#endif
}

void RenderTarget::cloneFrom(const RenderTarget& other)
{
	assert(size == other.size);
	memcpy(colors.get(), other.colors.get(), sizeof(RgbColor) * size);
}
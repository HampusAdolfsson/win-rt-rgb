#include "MaskingBehaviour.h"
#ifdef USE_SSE
#include <immintrin.h>
#endif

using namespace Rendering;

void UniformMaskingBehaviour::applyMask(RgbColor* colors, unsigned int size, float opacity)
{
#ifdef USE_SSE
	float* colorsVector = (float*) colors;
	__m256 intensityVector = _mm256_set1_ps(opacity);
	for (size_t i = 0; i * sizeof(float) < sizeof(RgbColor) * size; i += sizeof(__m256) / sizeof(float))
	{
		__m256 color = _mm256_load_ps(colorsVector + i);
		_mm256_store_ps(colorsVector + i, _mm256_mul_ps(color, intensityVector));
	}
#else
	for (size_t i = 0; i < size; i++)
	{
		colors[i] = colors[i] * opacity;
	}
#endif
}

void GradientMaskingBehaviour::applyMask(RgbColor* colors, unsigned int size, float opacity)
{
	float breakingPoint = opacity * size;
	int low = (int) breakingPoint;
	float decimals = breakingPoint - low;
	int gradientEnd = breakingPoint + 1;
	for (size_t i = gradientEnd + 1; i < size; i++)
	{
		colors[i] = RgbColor { 0.0f, 0.0f, 0.0f };
	}
	if (low + 1 < size) {
		colors[low+1] = colors[low + 1] * decimals;
	}
	// if (gradientEnd < size) {
	// 	colors[gradientEnd] = colors[gradientEnd] * 0.1f;
	// }
	// if (gradientStart >= 0) {
	// 	colors[gradientStart] = colors[gradientStart] * 0.2f;
	// }
	// colors[breakingPoint] = colors[breakingPoint] * 0.15f;
}
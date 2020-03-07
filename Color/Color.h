#pragma once
#include <cinttypes>

typedef struct
{
	uint8_t red, green, blue;
} Color;

Color operator*(const Color& c, const float &factor);

/**
*	Blends two colors together.
*	@param c1 the first color
*	@param c2 the second color
*	@param 0-1. 0 means the result == c1, 1 means result == c2.
*/
Color blendColors(Color c1, Color c2, float progress);

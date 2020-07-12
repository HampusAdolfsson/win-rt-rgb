#pragma once
#include <cinttypes>

typedef float ColorChannel;

typedef struct
{
	ColorChannel blue, green, red;
} RgbColor;

typedef struct
{
	float hue, saturation, value;
} HsvColor;

RgbColor operator*(const RgbColor& c, const float &factor);
bool operator==(const RgbColor& c1, const RgbColor &c2);
bool operator!=(const RgbColor& c1, const RgbColor &c2);

/**
*	Blends two colors together.
*	@param c1 the first color
*	@param c2 the second color
*	@param progress must be 0-1. 0 means the result == c1, 1 means result == c2.
*/
RgbColor blendColors(RgbColor c1, RgbColor c2, const float& progress);

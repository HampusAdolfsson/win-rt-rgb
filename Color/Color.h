#pragma once
#include <cinttypes>

typedef struct
{
	uint8_t red, green, blue;
} RgbColor;

typedef struct
{
	uint8_t hue, saturation, value;
} HsvColor;

RgbColor hsvToRgb(HsvColor hsv);
HsvColor rgbToHsv(RgbColor rgb);


RgbColor operator*(const RgbColor& c, const float &factor);

/**
*	Blends two colors together.
*	@param c1 the first color
*	@param c2 the second color
*	@param progress must be 0-1. 0 means the result == c1, 1 means result == c2.
*/
RgbColor blendColors(RgbColor c1, RgbColor c2, const float& progress);

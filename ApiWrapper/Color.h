#pragma once
#include <cinttypes>

typedef struct
{
	uint8_t red, green, blue;
} Color;

Color operator*(const Color& c, const float &factor);

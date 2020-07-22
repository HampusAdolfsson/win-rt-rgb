#include "Color.h"
#include <cassert>
#include <algorithm>

RgbColor operator*(const RgbColor& c, const float& factor)
{
	assert(factor >= 0 && factor <= 1);
	return {
		static_cast<ColorChannel>(c.red * factor),
		static_cast<ColorChannel>(c.green * factor),
		static_cast<ColorChannel>(c.blue * factor)
	};
}

bool operator==(const RgbColor& c1, const RgbColor& c2)
{
    return c1.red == c2.red && c1.green == c2.green && c1.blue == c2.blue;
}
bool operator!=(const RgbColor& c1, const RgbColor& c2)
{
    return c1.red != c2.red || c1.green != c2.green || c1.blue != c2.blue;
}

RgbColor blendColors(RgbColor c1, RgbColor c2, const float& progress)
{
	assert(progress >= 0 && progress <= 1);
	c1 = c1 * (1 - progress);
	c2 = c2 * progress;
	return {
		static_cast<ColorChannel>(c1.red + c2.red),
		static_cast<ColorChannel>(c1.green + c2.green),
		static_cast<ColorChannel>(c1.blue + c2.blue)
	};
}

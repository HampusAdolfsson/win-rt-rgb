#include <cassert>
#include "Color.h"

Color operator*(const Color& c, const float &factor) {
	assert(factor >= 0);
	return {
		static_cast<uint8_t>(c.red * factor),
		static_cast<uint8_t>(c.green * factor),
		static_cast<uint8_t>(c.blue * factor)
	};
}

Color blendColors(Color c1, Color c2, float progress)
{
	assert(progress >= 0 && progress <= 1);
	c1 = c1 * (1 - progress);
	c2 = c2 * progress;
	return {
		static_cast<uint8_t>(c1.red * c2.red),
		static_cast<uint8_t>(c1.green * c2.green),
		static_cast<uint8_t>(c1.blue * c2.blue)
	};
}

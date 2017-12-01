#include "Color.h"

Color operator*(const Color& c, const float &factor) {
	return {
		static_cast<uint8_t>(c.red * factor),
		static_cast<uint8_t>(c.green * factor),
		static_cast<uint8_t>(c.blue * factor)
	};
}

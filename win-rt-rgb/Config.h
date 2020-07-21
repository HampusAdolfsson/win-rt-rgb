#pragma once
#include "Rect.h"

#define WLED_ADDRESS "192.168.1.14"
#define WLED_UDP_PORT 21324

#define NUMBER_OF_LEDS 89

namespace Config {
	constexpr Rect defaultCaptureRegion = {0, 540, 1920, 540};
};

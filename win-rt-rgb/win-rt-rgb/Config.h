#pragma once
#include "Rect.h"
#include "SamplingSpecification.h"
#include <vector>

#define WLED_UDP_PORT 21324


#define NUMBER_OF_LEDS 89

namespace Config {
	// Enumerates the WLED devices to output to
	std::vector<std::pair<const char*, SamplingSpecification>> outputs = {
		std::pair<const char*, SamplingSpecification>("192.168.1.2", { 89, 0.2f, 3, true }),
		std::pair<const char*, SamplingSpecification>("192.168.1.13", { 10, 0.2f, 1, false }),
	};

	constexpr Rect defaultCaptureRegion = {0, 540, 1920, 540};
};

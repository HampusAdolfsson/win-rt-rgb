#pragma once
#include "Rect.h"
#include "SamplingSpecification.h"
#include <vector>

#define WLED_UDP_PORT 21324


namespace Config {
	// Enumerates the WLED devices to output to
	std::vector<std::pair<const char*, SamplingSpecification>> outputs = {
		std::pair<const char*, SamplingSpecification>("192.168.1.8", { 47, 0.15f, 1, true }),
		// std::pair<const char*, SamplingSpecification>("192.168.1.13", { 10, 0.2f, 0, false }),
	};

	constexpr Rect defaultCaptureRegion = {0, 810, 1920, 270};

	constexpr unsigned int websocketPort = 9901;
};

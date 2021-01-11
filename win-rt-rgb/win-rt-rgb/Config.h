#pragma once
#include "Types.h"
#include <vector>

#define WLED_UDP_PORT 21324


namespace Config {
	struct DeviceConfig {
		const char* ipAddress;
		DesktopCapture::SamplingSpecification samplingSpec;
		bool useAudio;
	};
	// Enumerates the WLED devices to output to
	std::vector<DeviceConfig> outputs {
		DeviceConfig{ "192.168.1.8", { 50, 0.15f, 1, false }, false },
		DeviceConfig{ "192.168.1.13", { 10, 0.5f, 2, false }, true },
	};

	constexpr DesktopCapture::Rect defaultCaptureRegion = {0, 810, 1920, 270};

	constexpr unsigned int websocketPort = 9901;

	constexpr unsigned int audioFps = 30;
};

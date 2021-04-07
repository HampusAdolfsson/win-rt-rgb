#pragma once
#include "Types.h"
#include <vector>

#define WLED_UDP_PORT 21324


namespace Config {
	struct DeviceConfig {
		const char* ipAddress;
		DesktopCapture::SamplingSpecification samplingSpec;
		unsigned int colorTemperature;
		float gamma;
		bool useAudio;
		unsigned int preferredMonitor;
	};
	// Enumerates the WLED devices to output to
	std::vector<DeviceConfig> outputs {
		DeviceConfig{ "192.168.1.8", { 50, 0.0f, 0.1f, 1, false }, 5000, 2.0f, false, 0 },
		DeviceConfig{ "192.168.1.13", { 10, 0.5f, 0.6f, 2, false }, 5000, 2.0f, true, 1 },
	};

	constexpr DesktopCapture::Rect defaultCaptureRegion = {0, 810, 1920, 270};

	constexpr unsigned int websocketPort = 9901;

	constexpr unsigned int audioFps = 30;
};

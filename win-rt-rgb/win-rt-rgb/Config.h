#pragma once
#include "Types.h"
#include <vector>

#define WLED_UDP_PORT 21324


namespace Config {

	constexpr DesktopCapture::Rect defaultCaptureRegion = {0, 810, 1920, 270};

	constexpr unsigned int websocketPort = 9901;

	constexpr unsigned int audioFps = 30;

	constexpr DesktopCapture::Rect monitors[3] = {
		{0, 0, 2560, 1440},
		{2560, 200, 1920, 1080},
		{365, -1080, 1920, 1081}
	};
};

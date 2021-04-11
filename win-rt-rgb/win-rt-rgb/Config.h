#pragma once
#include "Types.h"
#include <vector>

#define WLED_UDP_PORT 21324


namespace Config {

	constexpr DesktopCapture::Rect defaultCaptureRegion = {0, 810, 1920, 270};

	constexpr unsigned int websocketPort = 9901;

	constexpr unsigned int audioFps = 30;
};

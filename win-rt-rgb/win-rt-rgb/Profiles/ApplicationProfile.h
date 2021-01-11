#pragma once
#include "Types.h"
#include <vector>
#include <regex>

namespace WinRtRgb
{
	/**
	*	An application profile specifices a screen region to use to capture a specific
	*	application or window. The regex specifies the title of windows to use the profile for,
	*	and the rect gives the area to capture for those windows.
	*/
	struct ApplicationProfile
	{
		ApplicationProfile(std::string regexSpecifier, DesktopCapture::Rect region);
		std::regex windowTitle;
		std::string regexSpecifier;
		DesktopCapture::Rect captureRegion;
	};
}
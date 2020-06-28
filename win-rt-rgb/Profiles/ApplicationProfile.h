#pragma once
#include "Rect.h"
#include <vector>
#include <regex>

struct ApplicationProfile
{
	std::regex windowTitle;
	Rect captureRegion;
};

namespace Profiles
{
	extern std::vector<ApplicationProfile> dynamicProfiles;

	extern ApplicationProfile defaultProfiles[2];
}

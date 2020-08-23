#pragma once
#include "Rect.h"
#include <vector>
#include <regex>

struct ApplicationProfile
{
	ApplicationProfile(std::string regexSpecifier, Rect region);
	std::regex windowTitle;
	std::string regexSpecifier;
	Rect captureRegion;
};

namespace Profiles
{
	extern std::vector<ApplicationProfile> dynamicProfiles;

	extern ApplicationProfile defaultProfiles[2];
}

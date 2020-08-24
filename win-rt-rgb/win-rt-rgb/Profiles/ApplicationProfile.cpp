#include "ApplicationProfile.h"

ApplicationProfile::ApplicationProfile(std::string regexSpecifier, Rect region)
 : windowTitle(std::regex(regexSpecifier)),
 regexSpecifier(regexSpecifier),
 captureRegion(region)
{
}
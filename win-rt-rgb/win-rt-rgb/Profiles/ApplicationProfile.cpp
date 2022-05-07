#include "ApplicationProfile.h"

using namespace WinRtRgb;

ApplicationProfile::ApplicationProfile(unsigned int id, std::string regexSpecifier, std::vector<AreaSpecification> areas, int priority)
 : id(id),
 windowTitle(std::regex(regexSpecifier)),
 regexSpecifier(regexSpecifier),
 areas(areas),
 priority(priority)
{
}
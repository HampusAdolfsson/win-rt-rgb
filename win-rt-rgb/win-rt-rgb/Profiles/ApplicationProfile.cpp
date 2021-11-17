#include "ApplicationProfile.h"

using namespace WinRtRgb;

ApplicationProfile::ApplicationProfile(unsigned int id, std::string regexSpecifier, DesktopCapture::Rect region, int priority)
 : id(id),
 windowTitle(std::regex(regexSpecifier)),
 regexSpecifier(regexSpecifier),
 captureRegion(region),
 priority(priority)
{
}
#include "ApplicationProfile.h"

using namespace WinRtRgb;

ApplicationProfile::ApplicationProfile(std::string regexSpecifier, DesktopCapture::Rect region)
 : windowTitle(std::regex(regexSpecifier)),
 regexSpecifier(regexSpecifier),
 captureRegion(region)
{
}
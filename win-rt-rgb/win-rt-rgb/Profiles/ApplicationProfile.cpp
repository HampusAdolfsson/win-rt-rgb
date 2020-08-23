#include "ApplicationProfile.h"

ApplicationProfile::ApplicationProfile(std::string regexSpecifier, Rect region)
 : windowTitle(std::regex(regexSpecifier)),
 regexSpecifier(regexSpecifier),
 captureRegion(region)
{
}

std::vector<ApplicationProfile> Profiles::dynamicProfiles = {
	ApplicationProfile("^Spotify Premium$",
						{ 0, 590, 400, 400 }),
	ApplicationProfile(" - Microsoft Visual Studio",
						{ 0, 1057, 1920, 23 }),
	ApplicationProfile(" - Unity",
						{ 420, 200, 980, 500 }),
	ApplicationProfile("Substance Painter",
						{ 445, 145, 775, 620 }),
	ApplicationProfile(" - Visual Studio Code",
						{ 0, 1059, 1920, 21 }),
	ApplicationProfile("League of Legends \\(TM\\) Client",
						{ 0, 0, 1920, 1080 }),
	ApplicationProfile("^.+YouTube - Mozilla Firefox",
						{ 235, 162, 1430, 800 }),
	ApplicationProfile("^.+Twitch - Mozilla Firefox",
						{ 340, 134, 1278, 718 }),
	ApplicationProfile("^.+YouTube - Google Chrome",
						{ 72, 156, 1270, 710 })
};

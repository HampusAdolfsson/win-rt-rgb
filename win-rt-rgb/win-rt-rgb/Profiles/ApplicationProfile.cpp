#include "ApplicationProfile.h"

std::vector<ApplicationProfile> Profiles::dynamicProfiles = {
	{ std::regex("^Spotify Premium$"),
		{ 0, 590, 400, 400 } },
	{ std::regex(" - Microsoft Visual Studio"),
		{ 0, 1057, 1920, 23 } },
	{ std::regex(" - Unity"),
		{ 420, 200, 980, 500 } },
	{ std::regex("Substance Painter"),
		{ 445, 145, 775, 620 } },
	{ std::regex(" - Visual Studio Code"),
		{ 0, 1059, 1920, 21 } },
	{ std::regex("League of Legends \\(TM\\) Client"),
		{ 0, 0, 1920, 1080 } },
	{ std::regex("^.+YouTube - Mozilla Firefox"),
		{ 235, 162, 1430, 800 } },
	{ std::regex("^.+Twitch - Mozilla Firefox"),
		{ 340, 134, 1278, 718 } },
	{ std::regex("^.+YouTube - Google Chrome"),
		{ 72, 156, 1270, 710 } }
};

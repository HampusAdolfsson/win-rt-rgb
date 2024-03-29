#include <Windows.h>
#include "Config.h"
#include "RenderService.h"
#include "Logger.h"
#include "Profiles/ProfileManager.h"
#include "Profiles/ApplicationProfile.h"
#include "RenderTarget.h"
#include "WledRenderOutput.h"
#include "WebsocketServer.h"

using namespace WinRtRgb;


int main(int argc, char** argv)
{
	Logger::Instance().setLogFile("log");
	LOGINFO("Starting application");

	CoInitializeEx(nullptr, COINIT_MULTITHREADED);

	WSAData wsa;
	WSAStartup(MAKEWORD(2, 2), &wsa);

	RenderService renderer(Config::defaultCaptureRegion);


	int capturedOutput = 0;

	ProfileManager::start({});
	ProfileManager::addCallback([&](ProfileManager::ActiveProfileData profileData) {
		renderer.setActiveProfile(profileData);
	});


	bool audioVisualizerRunning = true;
	bool desktopVisualizerRunning = true;


	WebsocketServer server([](std::vector<ApplicationProfile> newProfiles)
	{ /* Profiles callback */
		ProfileManager::setProfiles(newProfiles);
	},
	[&](std::vector<RenderDeviceConfig> newDevices)
	{ /* Devices callback */
		renderer.stop();
		renderer.setRenderOutputs(std::move(newDevices));
		renderer.start();
	});
	ProfileManager::addCallback([&](ProfileManager::ActiveProfileData profileData) {
		if (profileData.profile.has_value())
		{
			server.notifyActiveProfileChanged(profileData.monitorIndex, profileData.profile->id);
		}
		else
		{
			server.notifyActiveProfileChanged(profileData.monitorIndex, std::nullopt);
		}
	});
	std::thread wsThread(&WebsocketServer::start, &server, Config::websocketPort);

	MSG msg;
	while (GetMessage(&msg, 0, WM_HOTKEY, 0) == 1)
	{
		switch (msg.message)
		{
		}
	}

	renderer.stop();
	LOGINFO("Exiting application ----------------------------------------------");
	ExitProcess(0);
	return 0;
}

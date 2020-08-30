#include <Windows.h>
#include "Config.h"
#include "AudioDesktopRenderer.h"
#include "Logger.h"
#include "WledHttpClient.h"
#include "Profiles/ProfileManager.h"
#include "Profiles/ApplicationProfile.h"
#include "HotkeyManager.h"
#include "RenderTarget.h"
#include "WledRenderOutput.h"
#include "WebsocketServer.h"

int main(int argc, char** argv)
{
	Logger::Instance().setLogFile("log");
	LOGINFO("Starting application");

	CoInitializeEx(nullptr, COINIT_MULTITHREADED);

	WSAData wsa;
	WSAStartup(MAKEWORD(2, 2), &wsa);

	AudioDesktopRenderer renderer;
	for (const auto& output : Config::outputs)
	{
		renderer.addRenderOutput(std::unique_ptr<RenderOutput>(new WledRenderOutput(output.second.numberOfRegions == 50 ? 89 : output.second.numberOfRegions, output.first, WLED_UDP_PORT)),
			output.second, false);
	}
	renderer.start();


	int capturedOutput = 0;

	ProfileManager::start({});
	ProfileManager::addCallback([&](std::optional<ProfileManager::ActiveProfileData> profileData) {
		if (profileData.has_value())
		{
			renderer.setDesktopRegion(profileData->monitorIndex, profileData->profile.captureRegion);
		}
		else
		{
			renderer.setDesktopRegion(capturedOutput, Config::defaultCaptureRegion);
		}
	});


	bool audioVisualizerRunning = true;
	bool desktopVisualizerRunning = true;


	WebsocketServer server([](std::vector<ApplicationProfile> newProfiles)
	{ /* Profiles callback */
		ProfileManager::setProfiles(newProfiles);
	}, [](std::optional<std::pair<unsigned int, unsigned int>> profileAndMonitorIdx)
	{ /* LockProfile callback */
		if (profileAndMonitorIdx.has_value())
		{
			ProfileManager::lockProfile(profileAndMonitorIdx->first, profileAndMonitorIdx->second);
		}
		else
		{
			ProfileManager::unlock();
		}

	});
	ProfileManager::addCallback([&](std::optional<ProfileManager::ActiveProfileData> profileData) {
		if (profileData.has_value())
		{
			server.notifyActiveProfileChanged(profileData->profileIndex);
		}
		else
		{
			server.notifyActiveProfileChanged(std::nullopt);
		}
	});
	std::thread wsThread(&WebsocketServer::start, &server, Config::websocketPort);


	HotkeyManager hotkeys;
	hotkeys.addHotkey(0x4b, [&]() { // k key
		LOGINFO("Hotkey pressed, switching captured monitor");
		capturedOutput = capturedOutput == 0 ? 1 : 0;
		renderer.setDesktopRegion(capturedOutput, Config::defaultCaptureRegion);
		return false;
	});
	hotkeys.addHotkey(0x4e, [&]() { // n key
		LOGINFO("Hotkey pressed, exiting application");
		return true;
	});
	hotkeys.runHandlerLoop();

	renderer.stop();
	LOGINFO("Exiting application ----------------------------------------------");
	ExitProcess(0);
	return 0;
}

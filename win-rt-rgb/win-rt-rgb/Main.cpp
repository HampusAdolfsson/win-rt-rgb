#include <Windows.h>
#include "Config.h"
#include "AudioDesktopRenderer.h"
#include "Logger.h"
#include "Profiles/ProfileManager.h"
#include "Profiles/ApplicationProfile.h"
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
		renderer.addRenderOutput(std::make_unique<WledRenderOutput>(output.samplingSpec.numberOfRegions, output.ipAddress, WLED_UDP_PORT),
			output.samplingSpec, output.useAudio);
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

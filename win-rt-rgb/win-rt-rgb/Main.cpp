#include <Windows.h>
#include "Config.h"
#include "App.h"
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

	std::vector<std::unique_ptr<RenderOutput>> outputs;
	std::vector<RenderTarget> targets;
	std::vector<SamplingSpecification> specifications;
	for (const auto& output : Config::outputs)
	{
		outputs.emplace_back(new WledRenderOutput(output.second.numberOfRegions, output.first, WLED_UDP_PORT));
		targets.push_back(RenderTarget(output.second.numberOfRegions));
		specifications.push_back(output.second);
	}

	App app(targets, std::move(outputs), specifications);
	app.startAudioVisualizer();
	app.startDesktopVisualizer();

	int capturedOutput = 0;

	ProfileManager::start([&](std::optional<std::pair<ApplicationProfile, unsigned int>> profileData) {
		if (profileData.has_value())
		{
			app.setDesktopRegion(profileData->second, profileData->first.captureRegion);
		}
		else
		{
			app.setDesktopRegion(capturedOutput, Config::defaultCaptureRegion);
		}
	}, { });


	bool audioVisualizerRunning = true;
	bool desktopVisualizerRunning = true;

	HotkeyManager hotkeys;
	hotkeys.addHotkey(0x56, [&]() { // v key
		LOGINFO("Hotkey pressed, toggling audio visualizer");
		if (audioVisualizerRunning) app.stopAudioVisualizer();
		else						app.startAudioVisualizer();
		audioVisualizerRunning = !audioVisualizerRunning;
		return false;
	});
	hotkeys.addHotkey(0x42, [&]() { // b key
		LOGINFO("Hotkey pressed, toggling desktop visualizer");
		if (desktopVisualizerRunning) app.stopDesktopVisualizer();
		else						  app.startDesktopVisualizer();
		desktopVisualizerRunning = !desktopVisualizerRunning;
		return false;
	});
	hotkeys.addHotkey(0x4b, [&]() { // k key
		LOGINFO("Hotkey pressed, switching captured monitor");
		capturedOutput = capturedOutput == 0 ? 1 : 0;
		app.setDesktopRegion(capturedOutput, Config::defaultCaptureRegion);
		return false;
	});
	hotkeys.addHotkey(0x4e, [&]() { // n key
		LOGINFO("Hotkey pressed, exiting application");
		return true;
	});

	WebsocketServer server([](std::vector<ApplicationProfile> newProfiles)
	{
		ProfileManager::setProfiles(newProfiles);
	}, [](std::optional<std::pair<unsigned int, unsigned int>> profileAndMonitorIdx)
	{
		if (profileAndMonitorIdx.has_value())
		{
			ProfileManager::lockProfile(profileAndMonitorIdx->first, profileAndMonitorIdx->second);
		}
		else
		{
			ProfileManager::unlock();
		}

	});
	std::thread wsThread(&WebsocketServer::start, &server, Config::websocketPort);

	hotkeys.runHandlerLoop();

	if (audioVisualizerRunning) app.stopAudioVisualizer();
	if (desktopVisualizerRunning) app.stopDesktopVisualizer();
	LOGINFO("Exiting application ----------------------------------------------");
	return 0;
}

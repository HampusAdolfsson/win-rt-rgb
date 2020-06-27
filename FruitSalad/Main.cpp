#include <Windows.h>
#include "App.h"
#include "Logger.h"
#include "Profiles/ProfileManager.h"
#include "Profiles/ApplicationProfile.h"
#include "HotkeyManager.h"

#define ADDR "192.168.1.6"
#define TCP_PORT "8844"
#define UDP_PORT 8845

// Played when launched
const LightEffect startEffect(1500000000, Breathing, { {0,0,60}, {0,0,150}, {0,0,255}, {0,0,150}, {0,0,60} });
// Played when exiting
const LightEffect exitEffect(1500000000, Breathing, { {60,0,0}, {150,0,0}, {255,0,0}, {150,0,0}, {60,0,0} });

int main(int argc, char** argv)
{
	Logger::Instance().setLogFile("log");
	LOGINFO("Starting application");

	CoInitializeEx(nullptr, COINIT_MULTITHREADED);

	WSAData wsa;
	WSAStartup(MAKEWORD(2, 2), &wsa);

	App app(ADDR, TCP_PORT, UDP_PORT);
	app.setServerOn();
	app.playLightEffect(LightEffect(startEffect));
	app.startAudioVisualizer();
	app.startDesktopVisualizer();

	int capturedOutput = 0;
	const Rect defaultCaptureRegion = { 0, 0, 1920, 1080 };
	bool locked = false;

	ProfileManager::start([&](std::optional<std::pair<ApplicationProfile, unsigned int>> profileData) {
		if (profileData.has_value())
		{
			if (!locked)
			{
				app.setDesktopRegion(profileData->second, profileData->first.captureRegion);
			}
		}
		else
		{
			if (!locked)
			{
				app.setDesktopRegion(capturedOutput, defaultCaptureRegion);
			}
		}
	}, Profiles::dynamicProfiles);


	bool audioVisualizerRunning = true;
	bool desktopVisualizerRunning = true;

	HotkeyManager hotkeys;
	hotkeys.addHotkey(0x43, [&]() { app.toggleServerOn(); return false; }); // c key
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
		app.setDesktopRegion(capturedOutput, defaultCaptureRegion);
		return false;
	});
	hotkeys.addHotkey(0x4c, [&]() { // l key
		LOGINFO("Hotkey pressed, toggling capture profile lock");
		locked = !locked;
		return false;
	});
	hotkeys.addHotkey(0x4e, [&]() { // n key
		LOGINFO("Hotkey pressed, exiting application");
		return true;
	});

	hotkeys.runHandlerLoop();

	app.playLightEffect(exitEffect);

	if (audioVisualizerRunning) app.stopAudioVisualizer();
	if (desktopVisualizerRunning) app.stopDesktopVisualizer();
	LOGINFO("Exiting application ----------------------------------------------");
	return 0;
}

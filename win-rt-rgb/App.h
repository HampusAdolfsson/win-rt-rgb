#pragma once
#include "RequestClient.h"
#include "OverrideColorClient.h"
#include "AudioMonitor.h"
#include "DesktopCaptureController.h"
#include "Profiles/ApplicationProfile.h"

class App
{
	RequestClient requestClient;
    OverrideColorClient realtimeClient;

	AudioMonitor audioMonitor;
	bool audioActive;
	DesktopCaptureController desktopCapturer;
	bool desktopActive;
	RgbColor desktopColor;

	void audioCallback(const float& intensity);
	void desktopCallback(RgbColor* colors);

public:
	App(const std::string& serverAddr, const std::string& tcpPort, const int& udpPort);

	void startAudioVisualizer();
	void stopAudioVisualizer();

	void startDesktopVisualizer();
	void stopDesktopVisualizer();

	void playLightEffect(const LightEffect& effect);

	void setServerOn();
	void toggleServerOn();

	void setDesktopRegion(const unsigned int& outputIdx, const Rect& region);
};

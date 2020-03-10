#pragma once#include "RequestClient.h"
#include "Audio\AudioMonitor.h"
#include "Gw2\Gw2BossNotifier.h"
#include "DesktopCaptureController.h"

class App
{
	RequestClient requestClient;
    OverrideColorClient realtimeClient;

	AudioMonitor audioMonitor;
	bool audioActive;
	DesktopCaptureController desktopCapturer;
	bool desktopActive;
	RgbColor desktopColor;

	Gw2BossNotifier gw2Notif;

	void audioCallback(const uint8_t& intensity);
	void desktopCallback(const RgbColor& color);

public:
	App(const WAVEFORMATEX& pwfx, const std::string& serverAddr, const std::string& tcpPort, const int& udpPort);

	void startAudioVisualizer();
	void stopAudioVisualizer();

	void startDesktopVisualizer();
	void stopDesktopVisualizer();

	void playLightEffect(const LightEffect& effect);

	void setServerOn();
	void toggleServerOn();
};

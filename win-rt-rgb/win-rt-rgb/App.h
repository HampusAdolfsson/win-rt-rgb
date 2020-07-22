#pragma once
#include "AudioMonitor.h"
#include "DesktopCaptureController.h"
#include "Profiles/ApplicationProfile.h"
#include "RenderTarget.h"
#include "RenderOutput.h"
#include "WledHttpClient.h"
#include <chrono>

class App
{
	RenderTarget renderTarget;
	std::unique_ptr<RenderOutput> renderOutput;
	WledHttpClient wledHttpClient;

	AudioMonitor audioMonitor;
	bool audioActive;
	DesktopCaptureController desktopCapturer;
	bool desktopActive;
	RgbColor desktopColor;

	std::chrono::time_point<std::chrono::system_clock> lastFpsTime;
	unsigned int frames;

	void audioCallback(const float& intensity);
	void desktopCallback(RgbColor* colors);

public:
	App(RenderTarget renderTarget, std::unique_ptr<RenderOutput> renderOutput, WledHttpClient httpClient);

	void startAudioVisualizer();
	void stopAudioVisualizer();

	void startDesktopVisualizer();
	void stopDesktopVisualizer();

	void setServerOn();
	void toggleServerOn();

	void setDesktopRegion(const unsigned int& outputIdx, const Rect& region);
};

#pragma once
#include "AudioMonitor.h"
#include "DesktopCaptureController.h"
#include "Profiles/ApplicationProfile.h"
#include "RenderTarget.h"
#include "RenderOutput.h"
#include "WledHttpClient.h"
#include "SamplingSpecification.h"
#include <chrono>

class App
{
	std::vector<RenderTarget> renderTargets;
	std::vector<std::unique_ptr<RenderOutput>> renderOutputs;

	AudioMonitor audioMonitor;
	bool audioActive;
	DesktopCaptureController desktopCapturer;
	bool desktopActive;
	RgbColor desktopColor;

	std::chrono::time_point<std::chrono::system_clock> lastFpsTime;
	unsigned int frames;

	void audioCallback(const float& intensity);
	void desktopCallback(const unsigned int& renderTargetIdx, RgbColor* colors);

public:
	App(std::vector<RenderTarget> renderTargets,
		std::vector<std::unique_ptr<RenderOutput>> renderOutputs,
		std::vector<SamplingSpecification> specifications);

	void startAudioVisualizer();
	void stopAudioVisualizer();

	void startDesktopVisualizer();
	void stopDesktopVisualizer();

	void setDesktopRegion(const unsigned int& outputIdx, const Rect& region);
};

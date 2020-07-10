#pragma once
#include "AudioMonitor.h"
#include "DesktopCaptureController.h"
#include "Profiles/ApplicationProfile.h"
#include "RenderTarget.h"
#include "RenderOutput.h"

class App
{
	RenderTarget renderTarget;
	std::unique_ptr<RenderOutput> renderOutput;

	AudioMonitor audioMonitor;
	bool audioActive;
	DesktopCaptureController desktopCapturer;
	bool desktopActive;
	RgbColor desktopColor;

	void audioCallback(const float& intensity);
	void desktopCallback(RgbColor* colors);

public:
	App(RenderTarget renderTarget, std::unique_ptr<RenderOutput> renderOutput);

	void startAudioVisualizer();
	void stopAudioVisualizer();

	void startDesktopVisualizer();
	void stopDesktopVisualizer();

	void setServerOn();
	void toggleServerOn();

	void setDesktopRegion(const unsigned int& outputIdx, const Rect& region);
};

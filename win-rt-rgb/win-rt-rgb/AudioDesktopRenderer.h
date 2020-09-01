#pragma once
#include "DesktopCaptureController.h"
#include "SamplingSpecification.h"
#include "AudioMonitor.h"
#include "RenderOutput.h"
#include <memory>
#include <optional>
#include <chrono>

struct RenderDevice;

/**
*   Renders colors captured from the desktop and (optionally) captured audio to a
*   set of render outputs.
*/
class AudioDesktopRenderer
{
public:
	void addRenderOutput(std::unique_ptr<RenderOutput> renderOutput, SamplingSpecification desktopCaptureParams, bool useAudio);

	void start();
	void stop();

	void setDesktopRegion(const unsigned int& outputIdx, const Rect& region);

private:
	bool started = false;

	std::vector<RenderDevice> devices;
	std::unique_ptr<DesktopCaptureController> desktopCaptureController = nullptr;
	std::unique_ptr<AudioMonitor> audioMonitor = nullptr;

	// measuring fps
	unsigned int frames = 0;
	std::chrono::time_point<std::chrono::system_clock> lastFpsTime;

	void audioCallback(const float& intensity);
	void desktopCallback(const unsigned int& deviceIdx, RgbColor* colors);
};

struct RenderDevice {
	std::unique_ptr<RenderOutput> renderOutput;
	SamplingSpecification desktopCaptureParams;
	RenderTarget desktopRenderTarget;
	std::optional<RenderTarget> audioRenderTarget;
};
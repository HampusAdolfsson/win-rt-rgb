#pragma once
#include "DesktopCaptureController.h"
#include "Types.h"
#include "AudioMonitor.h"
#include "RenderOutput.h"
#include <vector>
#include <memory>
#include <optional>
#include <chrono>

namespace WinRtRgb
{
	struct RenderDevice;

	/**
	*   Renders colors captured from the desktop and (optionally) captured audio to a
	*   set of render outputs.
	*/
	class AudioDesktopRenderer
	{
	public:
		void addRenderOutput(std::unique_ptr<Rendering::RenderOutput> renderOutput, DesktopCapture::SamplingSpecification desktopCaptureParams, bool useAudio);

		void start();
		void stop();

		void setDesktopRegion(const unsigned int& outputIdx, const DesktopCapture::Rect& region);

	private:
		bool started = false;

		std::vector<RenderDevice> devices;
		std::unique_ptr<DesktopCapture::DesktopCaptureController> desktopCaptureController = nullptr;
		std::unique_ptr<AudioCapture::AudioMonitor> audioMonitor = nullptr;

		// measuring fps
		unsigned int frames = 0;
		std::chrono::time_point<std::chrono::system_clock> lastFpsTime;

		void audioCallback(float intensity);
		void desktopCallback(unsigned int deviceIdx, RgbColor* colors);
	};

	struct RenderDevice {
		std::unique_ptr<Rendering::RenderOutput> renderOutput;
		DesktopCapture::SamplingSpecification desktopCaptureParams;
		Rendering::RenderTarget desktopRenderTarget;
		std::optional<Rendering::RenderTarget> audioRenderTarget;
	};
}
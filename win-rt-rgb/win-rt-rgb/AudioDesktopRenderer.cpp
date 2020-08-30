#include "AudioDesktopRenderer.h"
#include "Logger.h"
#include <stdexcept>

void AudioDesktopRenderer::addRenderOutput(std::unique_ptr<RenderOutput> renderOutput, SamplingSpecification desktopCaptureParams, bool useAudio)
{
	if (started) { throw std::runtime_error("Already started"); }
	RenderDevice device = {
		std::move(renderOutput),
		desktopCaptureParams,
		RenderTarget(desktopCaptureParams.numberOfRegions),
		useAudio ? std::optional(RenderTarget(desktopCaptureParams.numberOfRegions)) : std::nullopt
	};
	devices.push_back(std::move(device));
}

void AudioDesktopRenderer::start()
{
	if (started) { throw std::runtime_error("Already started"); }
	started = true;

	if (!desktopCaptureController.get())
	{
		std::vector<SamplingSpecification> specs;
		for (const auto& device : devices)
		{
			specs.push_back(device.desktopCaptureParams);
		}
		desktopCaptureController = std::unique_ptr<DesktopCaptureController>(new DesktopCaptureController(0, specs,
			std::bind(&AudioDesktopRenderer::desktopCallback, this, std::placeholders::_1, std::placeholders::_2)));
	}
	lastFpsTime = std::chrono::system_clock::now();
	desktopCaptureController->start();
}

void AudioDesktopRenderer::stop()
{
	if (!started) { throw std::runtime_error("Not running"); }
	started = false;
	desktopCaptureController->stop();
}

void AudioDesktopRenderer::setDesktopRegion(const unsigned int& outputIdx, const Rect& region)
{
	if (desktopCaptureController.get())
	{
		desktopCaptureController->setOutput(outputIdx, region);
	}
}

void AudioDesktopRenderer::audioCallback(const float& intensity)
{
	if (!started) { return; }
		for (RenderDevice& device: devices)
		{
			if (!device.audioRenderTarget.has_value()) { return; }
			device.audioRenderTarget->cloneFrom(device.desktopRenderTarget);
			device.audioRenderTarget->setIntensity(intensity);
			device.renderOutput->draw(*device.audioRenderTarget);
		}
}
void AudioDesktopRenderer::desktopCallback(const unsigned int& deviceIdx, RgbColor* colors)
{
	if (!started) { return; }
	RenderDevice& device = devices[deviceIdx];
	device.desktopRenderTarget.beginFrame();
	device.desktopRenderTarget.drawRange(0, device.desktopRenderTarget.getSize(), colors);
	if (!device.audioRenderTarget.has_value())
	{
		// This device shouldn't use audio, so just render desktop colors
		device.renderOutput->draw(device.desktopRenderTarget);
		frames++;
		auto timeSinceLastFps = std::chrono::system_clock::now() - lastFpsTime;
		if (timeSinceLastFps > std::chrono::seconds(1)) {
			LOGINFO("Desktop FPS: %d", frames);
			frames = 0;
			lastFpsTime = lastFpsTime + std::chrono::seconds(1); // TODO: fix accuracy
		}
	}
}
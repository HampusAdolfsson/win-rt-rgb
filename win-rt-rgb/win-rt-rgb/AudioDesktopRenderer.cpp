#include "AudioDesktopRenderer.h"
#include "EnergyAudioHandler.h"
#include "Logger.h"
#include <stdexcept>

using namespace WinRtRgb;

void AudioDesktopRenderer::addRenderOutput(std::unique_ptr<Rendering::RenderOutput> renderOutput, DesktopCapture::SamplingSpecification desktopCaptureParams, bool useAudio)
{
	if (started) { throw std::runtime_error("Already started"); }
	RenderDevice device = {
		std::move(renderOutput),
		desktopCaptureParams,
		Rendering::RenderTarget(desktopCaptureParams.numberOfRegions),
		useAudio ? std::optional(Rendering::RenderTarget(desktopCaptureParams.numberOfRegions)) : std::nullopt
	};
	devices.push_back(std::move(device));
}

void AudioDesktopRenderer::start()
{
	if (started) { throw std::runtime_error("Already started"); }
	started = true;

	if (!desktopCaptureController.get())
	{
		std::vector<std::pair<DesktopCapture::SamplingSpecification, DesktopCapture::DesktopSamplingCallback>> specs;
		for (int i = 0; i < devices.size(); i++)
		{
			DesktopCapture::DesktopSamplingCallback callback = std::bind(&AudioDesktopRenderer::desktopCallback, this, i, std::placeholders::_1);
			specs.push_back({devices[i].desktopCaptureParams, callback});
		}
		desktopCaptureController = std::make_unique<DesktopCapture::DesktopCaptureController>(0, specs);
	}
	if (!audioMonitor.get())
	{
		audioMonitor = std::make_unique<AudioCapture::AudioMonitor>(AudioCapture::AudioSink(30, std::make_unique<AudioCapture::EnergyAudioHandlerFactory>(std::bind(&AudioDesktopRenderer::audioCallback, this, std::placeholders::_1))));
		audioMonitor->initialize();
	}

	lastFpsTime = std::chrono::system_clock::now();
	desktopCaptureController->start();
	audioMonitor->start();
}

void AudioDesktopRenderer::stop()
{
	if (!started) { throw std::runtime_error("Not running"); }
	started = false;
	desktopCaptureController->stop();
	audioMonitor->stop();
}

void AudioDesktopRenderer::setDesktopRegion(const unsigned int& outputIdx, const DesktopCapture::Rect& region)
{
	if (desktopCaptureController.get())
	{
		desktopCaptureController->setCaptureRegion(outputIdx, region);
	}
}

void AudioDesktopRenderer::audioCallback(float intensity)
{
	if (!started) { return; }
	for (RenderDevice& device : devices)
	{
		if (!device.audioRenderTarget.has_value()) { continue; }
		device.audioRenderTarget->cloneFrom(device.desktopRenderTarget);
		device.audioRenderTarget->setIntensity(intensity);
		device.renderOutput->draw(*device.audioRenderTarget);
	}
}
void AudioDesktopRenderer::desktopCallback(unsigned int deviceIdx, RgbColor* colors)
{
	if (!started) { return; }
	RenderDevice& device = devices[deviceIdx];
	device.desktopRenderTarget.beginFrame();
	device.desktopRenderTarget.drawRange(deviceIdx == 0 ? 12 : 0, device.desktopCaptureParams.numberOfRegions, colors);
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
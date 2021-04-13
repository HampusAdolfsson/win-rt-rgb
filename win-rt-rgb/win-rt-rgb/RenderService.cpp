#include "RenderService.h"
#include "EnergyAudioHandler.h"
#include "MaskingBehaviour.h"
#include "Logger.h"
#include <stdexcept>
#include <numeric>

using namespace WinRtRgb;


RenderService::RenderService(DesktopCapture::Rect defaultCaptureRegion)
 : defaultCaptureRegion(defaultCaptureRegion)
{
}

void RenderService::setRenderOutputs(std::vector<RenderDeviceConfig> devices)
{
	if (started) { throw std::runtime_error("Already started"); }
	this->devices.clear();
	for (auto& devConf : devices)
	{
		devConf.output->initialize();
		size_t ledCount = devConf.output->getLedCount();
		RenderDevice device = {
			std::move(devConf.output),
			Rendering::RenderTarget(ledCount, std::unique_ptr<Rendering::MaskingBehaviour>(new Rendering::UniformMaskingBehaviour())),
			devConf.useAudio ? std::optional(Rendering::RenderTarget(ledCount, std::unique_ptr<Rendering::MaskingBehaviour>(new Rendering::UniformMaskingBehaviour()))) : std::nullopt,
			devConf.preferredMonitor,
			devConf.saturationAdjustment,
			devConf.valueAdjustment
		};
		this->devices.push_back(std::move(device));
	}
}

void RenderService::start()
{
	if (started) { return; }

	if (!desktopCaptureController.get())
	{
		desktopCaptureController = std::make_unique<DesktopCapture::DesktopCaptureController>();
	}
	if (!audioMonitor.get())
	{
		for (const auto& dev : devices)
		{
			if (dev.audioRenderTarget.has_value())
			{
				audioMonitor = std::make_unique<AudioCapture::AudioMonitor>(AudioCapture::AudioSink(30, std::make_unique<AudioCapture::EnergyAudioHandlerFactory>(std::bind(&RenderService::audioCallback, this, std::placeholders::_1))));
				audioMonitor->initialize();
				break;
			}
		}

	}

	lastFpsTime = std::chrono::system_clock::now();

	std::vector<std::pair<size_t, DesktopCapture::DesktopSamplingCallback>> specs;
	for (int i = 0; i < devices.size(); i++)
	{
		DesktopCapture::DesktopSamplingCallback callback = std::bind(&RenderService::desktopCallback, this, i, std::placeholders::_1);
		specs.push_back({devices[i].renderOutput->getLedCount(), callback});
	}
	desktopCaptureController->setOutputSpecifications(specs);
	desktopCaptureController->start();
	if (audioMonitor.get()) audioMonitor->start();

	started = true;
}

void RenderService::stop()
{
	if (!started) { return; }
	started = false;
	if (desktopCaptureController.get()) { desktopCaptureController->stop(); }
	if (audioMonitor.get()) { audioMonitor->stop(); }
}

void RenderService::setActiveProfile(ProfileManager::ActiveProfileData profileData)
{
	if (profileData.profile.has_value()) activeProfiles.insert(std::make_pair(profileData.monitorIndex, profileData));
	else activeProfiles.erase(profileData.monitorIndex);
	if (desktopCaptureController.get())
	{
		desktopCaptureController->setCaptureRegionForMonitor(profileData.monitorIndex,
			profileData.profile.has_value() ? profileData.profile->captureRegion : defaultCaptureRegion);
		for (int i = 0; i < devices.size(); i++)
		{
			if (activeProfiles.count(devices[i].preferredMonitor) != 0)
			{
				desktopCaptureController->setCaptureMonitorForOutput(i, devices[i].preferredMonitor);
			}
			else
			{
				desktopCaptureController->setCaptureMonitorForOutput(i, activeProfiles.empty() ? 0 : activeProfiles.begin()->first);
			}

		}
	}
}

void RenderService::audioCallback(float intensity)
{
	if (!started) { return; }
	for (RenderDevice& device : devices)
	{
		if (!device.audioRenderTarget.has_value()) { continue; }
		device.audioRenderTarget->cloneFrom(device.desktopRenderTarget);
		device.audioRenderTarget->applyAdjustments(0, device.audioRenderTarget->getSize(),
													.0f,
													device.saturationAdjustment,
													device.valueAdjustment);
		device.audioRenderTarget->setIntensity(intensity);
		device.renderOutput->draw(*device.audioRenderTarget);
	}
}
void RenderService::desktopCallback(unsigned int deviceIdx, const RgbColor* colors)
{
	if (!started) { return; }
	RenderDevice& device = devices[deviceIdx];
	device.desktopRenderTarget.beginFrame();
	device.desktopRenderTarget.drawRange(0, device.desktopRenderTarget.getSize(), colors);
	if (!device.audioRenderTarget.has_value())
	{
		// This device shouldn't use audio, so just render desktop colors
		device.desktopRenderTarget.applyAdjustments(0, device.desktopRenderTarget.getSize(),
													.0f,
													device.saturationAdjustment,
													device.valueAdjustment);
		device.renderOutput->draw(device.desktopRenderTarget);
		frames++;
		auto timeSinceLastFps = std::chrono::system_clock::now() - lastFpsTime;
		if (timeSinceLastFps > std::chrono::seconds(1)) {
			// LOGINFO("Desktop FPS: %d", frames);
			frames = 0;
			lastFpsTime = lastFpsTime + std::chrono::seconds(1); // TODO: fix accuracy
		}
	}
}
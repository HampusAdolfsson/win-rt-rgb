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
			devConf.audioAmount > .0f ? std::optional(Rendering::RenderTarget(ledCount, std::unique_ptr<Rendering::MaskingBehaviour>(new Rendering::UniformMaskingBehaviour()))) : std::nullopt,
			devConf.saturationAdjustment,
			devConf.valueAdjustment,
			devConf.audioAmount
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
				audioMonitor = std::make_unique<AudioCapture::AudioMonitor>();
				audioMonitor->addAudioSink(AudioCapture::AudioSink(30, std::make_unique<AudioCapture::EnergyAudioHandlerFactory>(std::bind(&RenderService::audioCallback, this, std::placeholders::_1))));
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
	if (profileData.profile.has_value()) activeProfiles.insert(std::make_pair(profileData.monitorIndex, profileData.profile.value()));
	else activeProfiles.erase(profileData.monitorIndex);
	if (desktopCaptureController.get())
	{
		std::optional<std::pair<unsigned int, int>> maxPriority;
		for (const auto& prof : activeProfiles)
		{
			if (maxPriority.has_value())
			{
				if (prof.second.priority > maxPriority->second)
				{
					maxPriority = std::make_pair(prof.first, prof.second.priority);
				}
			}
			else
			{
				maxPriority = std::make_pair(prof.first, prof.second.priority);
			}

		}
		unsigned int monitorIndex = maxPriority.has_value() ? maxPriority->first : 0;
		setActiveMonitor(monitorIndex);
	}
}

void RenderService::setActiveMonitor(unsigned  int monitorIdx)
{
	LOGINFO("Activating monitor %d.", monitorIdx);
	if (activeProfiles.count(monitorIdx) == 1)
	{
		std::optional<AreaSpecification> matchingArea;
		auto monitorDimensions = desktopCaptureController->getMonitorDimensions(monitorIdx);
		for (const AreaSpecification area : activeProfiles.at(monitorIdx).areas)
		{
			if (!area.resolution.has_value() && !matchingArea.has_value()) matchingArea = area;
			else if (area.resolution.has_value() && area.resolution.value() == monitorDimensions) matchingArea = area;
		}

		if (matchingArea.has_value())
		{
			DesktopCapture::Rect resolvedArea;
			resolvedArea.left = resolveMonitorDistance(matchingArea->x, monitorDimensions.first);
			resolvedArea.width = resolveMonitorDistance(matchingArea->width, monitorDimensions.first);
			resolvedArea.top = resolveMonitorDistance(matchingArea->y, monitorDimensions.second);
			resolvedArea.height = resolveMonitorDistance(matchingArea->height, monitorDimensions.second);
			desktopCaptureController->setCaptureRegionForMonitor(monitorIdx, resolvedArea);
		}
		else
		{
			LOGWARNING("Profile %s has no matching area for %dx%d.", activeProfiles.at(monitorIdx).regexSpecifier, monitorDimensions.first, monitorDimensions.second);
			desktopCaptureController->setCaptureRegionForMonitor(monitorIdx, defaultCaptureRegion);
		}
	}
	else
	{
		desktopCaptureController->setCaptureRegionForMonitor(monitorIdx, defaultCaptureRegion);
	}
	for (int i = 0; i < devices.size(); i++)
	{
		desktopCaptureController->setCaptureMonitorForOutput(i, monitorIdx);
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
		float actualIntensity = intensity * device.audioAmount + (1.f - device.audioAmount);
		device.audioRenderTarget->setIntensity(actualIntensity);
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
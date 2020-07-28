#include "App.h"
#include "Logger.h"
#include "WaveToIntensityConverter.h"
#include <algorithm>

App::App(std::vector<RenderTarget> renderTargets,
		std::vector<std::unique_ptr<RenderOutput>> renderOutputs,
		std::vector<SamplingSpecification> specifications)
	: renderTargets(renderTargets),
	renderOutputs(std::move(renderOutputs)),
	audioMonitor(std::make_unique<WaveToIntensityConverter>(std::bind(&App::audioCallback, this, std::placeholders::_1))),
	audioActive(false),
	desktopCapturer(0, specifications, std::bind(&App::desktopCallback, this, std::placeholders::_1, std::placeholders::_2)),
	desktopActive(false),
	lastFpsTime(std::chrono::system_clock::now()),
	frames(0)
{
	audioMonitor.initialize();
}

void App::startAudioVisualizer()
{
	// audioActive = true;
	// audioMonitor.start();
}
void App::stopAudioVisualizer()
{
	audioActive = false;
	audioMonitor.stop();
}

void App::startDesktopVisualizer()
{
	lastFpsTime = std::chrono::system_clock::now();
	desktopActive = true;
	desktopCapturer.start();
}
void App::stopDesktopVisualizer()
{
	desktopActive = false;
	desktopCapturer.stop();
}

void App::setDesktopRegion(const unsigned int& outputIdx, const Rect& region)
{
	desktopCapturer.setOutput(outputIdx, region);
}

void App::audioCallback(const float& intensity)
{
	if (!audioActive) return;
	if (desktopActive)
	{
		//realtimeClient.sendColor(hsvToRgb(hsv) * intensity);
	}
	else
	{
		RgbColor base = { 255, 0, 0 };
		//realtimeClient.sendColor(base * intensity);
	}
}
void App::desktopCallback(const unsigned int& renderTargetIdx, RgbColor* colors)
{
	RenderTarget& target = renderTargets[renderTargetIdx];
	RenderOutput* output = renderOutputs[renderTargetIdx].get();
	if (!desktopActive) return;
	target.beginFrame();
	target.drawRange(0, target.getSize(), colors);
	if (audioActive)
	{
		desktopColor = colors[0];
	}
	else
	{
		output->draw(target);
		frames++;
		auto timeSinceLastFps = std::chrono::system_clock::now() - lastFpsTime;
		if (timeSinceLastFps > std::chrono::seconds(1)) {
			LOGINFO("Desktop FPS: %d", frames);
			frames = 0;
			lastFpsTime = lastFpsTime + std::chrono::seconds(1); // TODO: fix accuracy
		}
	}
}
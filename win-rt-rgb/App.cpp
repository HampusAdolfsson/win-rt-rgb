#include "App.h"
#include "Logger.h"
#include "WaveToIntensityConverter.h"
#include <algorithm>

App::App(RenderTarget renderTarget, std::unique_ptr<RenderOutput> renderOutput, WledHttpClient httpClient)
	: renderTarget(renderTarget),
	renderOutput(std::move(renderOutput)),
	wledHttpClient(httpClient),
	audioMonitor(std::make_unique<WaveToIntensityConverter>(std::bind(&App::audioCallback, this, std::placeholders::_1))),
	audioActive(false),
	desktopCapturer(0, renderTarget.getColors().size(), std::bind(&App::desktopCallback, this, std::placeholders::_1)),
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

void App::setServerOn()
{
	wledHttpClient.setPowerStatus(WledPowerStatus::On);
}
void App::toggleServerOn()
{
	wledHttpClient.setPowerStatus(WledPowerStatus::Toggle);
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
void App::desktopCallback(RgbColor* colors)
{
	if (!desktopActive) return;
	renderTarget.beginFrame();
	renderTarget.drawRange(0, renderTarget.getSize(), colors);
	if (audioActive)
	{
		desktopColor = colors[0];
	}
	else
	{
		renderOutput->draw(renderTarget);
		frames++;
		auto timeSinceLastFps = std::chrono::system_clock::now() - lastFpsTime;
		if (timeSinceLastFps > std::chrono::seconds(1)) {
			LOGINFO("Desktop FPS: %d", frames);
			frames = 0;
			lastFpsTime = lastFpsTime + std::chrono::seconds(1); // TODO: fix accuracy
		}
	}
}
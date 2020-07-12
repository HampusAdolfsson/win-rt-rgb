#include "App.h"
#include "Logger.h"
#include "WaveToIntensityConverter.h"
#include <algorithm>

App::App(RenderTarget renderTarget, std::unique_ptr<RenderOutput> renderOutput)
	: renderTarget(renderTarget),
	renderOutput(std::move(renderOutput)),
	audioMonitor(std::make_unique<WaveToIntensityConverter>(std::bind(&App::audioCallback, this, std::placeholders::_1))),
	audioActive(false),
	desktopCapturer(0, renderTarget.getColors().size(), std::bind(&App::desktopCallback, this, std::placeholders::_1)),
	desktopActive(false)
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

}
void App::toggleServerOn()
{

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
		HsvColor hsv = rgbToHsv(desktopColor);
		if (hsv.saturation > 6)
		{
			hsv.saturation = min(hsv.saturation + 100, 255);
		}
		hsv.value = 255;
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
		HsvColor hsv = rgbToHsv(colors[0]);
		hsv.saturation = min(hsv.saturation + 100, 255);
		hsv.value = 255;
		renderOutput->draw(renderTarget);
		// realtimeClient.sendColor(hsvToRgb(hsv));
	}
}
#include "App.h"
#include "ResponseCodes.h"
#include "Logger.h"
#include "WaveToIntensityConverter.h"
#include <algorithm>

App::App(const std::string& serverAddr, const std::string& tcpPort, const int& udpPort)
	: requestClient(serverAddr, tcpPort),
	audioMonitor(std::make_unique<WaveToIntensityConverter>(std::bind(&App::audioCallback, this, std::placeholders::_1))),
	audioActive(false),
	desktopCapturer(0, 1, std::bind(&App::desktopCallback, this, std::placeholders::_1)),
	desktopActive(false),
	realtimeClient(serverAddr, udpPort)
{
	audioMonitor.initialize();
	OverrideColorClient colorClient(serverAddr, udpPort);
}

void App::startAudioVisualizer()
{
	audioActive = true;
	audioMonitor.start();
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

void App::playLightEffect(const LightEffect& effect)
{
	unsigned char res = requestClient.sendLightEffect(effect, false);
	if (res != SUCCESS)
	{
		LOGWARNING("Couldn't play lighteffect, server returned: %d", res);
	}
}

void App::setServerOn()
{
	unsigned char res = requestClient.sendOnOffRequest(ON);
	if (res != SUCCESS)
	{
		LOGWARNING("Couldn't set server on, server returned: %d", res);
	}
}
void App::toggleServerOn()
{
	unsigned char res = requestClient.sendOnOffRequest(TOGGLE);
	if (res != SUCCESS)
	{
		LOGWARNING("Couldn't toggle server, server returned: %d", res);
	}
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
		realtimeClient.sendColor(hsvToRgb(hsv) * intensity);
	}
	else
	{
		RgbColor base = { 255, 0, 0 };
		realtimeClient.sendColor(base * intensity);
	}
}
void App::desktopCallback(RgbColor* color)
{
	if (!desktopActive) return;
	if (audioActive)
	{
		desktopColor = color[0];
	}
	else
	{
		HsvColor hsv = rgbToHsv(color[0]);
		hsv.saturation = min(hsv.saturation + 100, 255);
		hsv.value = 255;
		realtimeClient.sendColor(hsvToRgb(hsv));
	}
}
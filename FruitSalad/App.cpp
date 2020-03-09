#include "App.h"
#include "ResponseCodes.h"
#include "Logger.h"
#include <algorithm>

App::App(const WAVEFORMATEX &pwfx, const std::string &serverAddr, const std::string &tcpPort, const int &udpPort)
    : requestClient(serverAddr, tcpPort),
      audioMonitor(2, pwfx, std::bind(&App::audioCallback, this, std::placeholders::_1)),
      audioActive(false),
      desktopCapturer(0),
      desktopActive(false),
      realtimeClient(serverAddr, udpPort),
      gw2Notif(requestClient)
{
    audioMonitor.initialize();
    OverrideColorClient colorClient(serverAddr, udpPort);
    //while (true)
    //{
    //    RgbColor color = desktopCapturer.getColor();
    //    HsvColor hsv = rgbToHsv(color);
    //    hsv.saturation = 255;
    //    //hsv.value = visualizer.getIntensity();
    //    //colorClient.sendColor(hsvToRgb(hsv));
    //}
}

void App::startVisualizer()
{
    audioActive = true;
    audioMonitor.start();
}
void App::stopVisualizer()
{
    audioActive = false;
    audioMonitor.stop();
}

void App::playLightEffect(const LightEffect &effect)
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

void App::audioCallback(uint8_t intensity)
{
    if (!audioActive) return;
    if (desktopActive)
    {
        HsvColor hsv = rgbToHsv(desktopColor);
        hsv.saturation = min(hsv.saturation + 100, 255);
        hsv.value = intensity;
        realtimeClient.sendColor(hsvToRgb(hsv));
    }
    else
    {
        RgbColor base = {255, 0, 0};
        realtimeClient.sendColor(base * (intensity / 255.0));
    }
}
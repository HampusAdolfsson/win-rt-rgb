#include <Windows.h>
#include "App.h"
#include "Logger.h"


#define ADDR "192.168.1.6"
#define TCP_PORT "8844"
#define UDP_PORT 8845
#define TOGGLE_SERVER_KEY 0x43 // c key
#define TOGGLE_SERVER_ID 0x1
#define TOGGLE_AUDIO_VISUALIZER_KEY 0x56 // v key
#define TOGGLE_AUDIO_VISUALIZER_ID 0x2
#define TOGGLE_DESKTOP_VISUALIZER_KEY 0x42 // b key
#define TOGGLE_DESKTOP_VISUALIZER_ID 0x3
#define EXIT_APPLICATION_KEY 0x4e // n key
#define EXIT_APPLICATION_ID 0x4



// Played when launched
const LightEffect startEffect(1500000000, Breathing, { {0,0,60}, {0,0,150}, {0,0,255}, {0,0,150}, {0,0,60} });
// Played when exiting
const LightEffect exitEffect(1500000000, Breathing, { {60,0,0}, {150,0,0}, {255,0,0}, {150,0,0}, {60,0,0} });

int main(int argc, char **argv) {
	Logger::Instance().setLogFile("log");
	LOGINFO("Starting application");

	WSAData wsa;
	WSAStartup(MAKEWORD(2,2), &wsa);

	WAVEFORMATEX pwfx;
	pwfx.wFormatTag = WAVE_FORMAT_PCM;
	pwfx.nChannels = 1;
	pwfx.nSamplesPerSec = 48000;
	pwfx.wBitsPerSample = 16;
	pwfx.nBlockAlign = pwfx.nChannels * (pwfx.wBitsPerSample / 8);
	pwfx.nAvgBytesPerSec = pwfx.nBlockAlign * pwfx.nSamplesPerSec;
	pwfx.cbSize = 0;
	
	App app(pwfx, ADDR, TCP_PORT, UDP_PORT);
	app.setServerOn();
	app.playLightEffect(LightEffect(startEffect));
	app.startAudioVisualizer();
	app.startDesktopVisualizer();

	RegisterHotKey(NULL, TOGGLE_SERVER_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, TOGGLE_SERVER_KEY);
	RegisterHotKey(NULL, TOGGLE_AUDIO_VISUALIZER_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, TOGGLE_AUDIO_VISUALIZER_KEY);
	RegisterHotKey(NULL, TOGGLE_DESKTOP_VISUALIZER_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, TOGGLE_DESKTOP_VISUALIZER_KEY);
	RegisterHotKey(NULL, EXIT_APPLICATION_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, EXIT_APPLICATION_KEY);

	MSG msg;
	bool audioVisualizerRunning = true;
	bool desktopVisualizerRunning = true;
	while (GetMessage(&msg, 0, 0, 0) == 1)
	{
		switch (msg.message)
		{
		case WM_HOTKEY:
			switch (msg.wParam)
			{
			case TOGGLE_SERVER_ID:
				LOGINFO("Hotkey pressed, toggling server");
				app.toggleServerOn();
				break;
			case TOGGLE_AUDIO_VISUALIZER_ID:
				LOGINFO("Hotkey pressed, toggling audio visualizer");
				if (audioVisualizerRunning) app.stopAudioVisualizer();
				else						app.startAudioVisualizer();
				audioVisualizerRunning = !audioVisualizerRunning;
				break;
			case TOGGLE_DESKTOP_VISUALIZER_ID:
				LOGINFO("Hotkey pressed, toggling desktop visualizer");
				if (desktopVisualizerRunning) app.stopDesktopVisualizer();
				else						  app.startDesktopVisualizer();
				desktopVisualizerRunning = !desktopVisualizerRunning;
				break;
			case EXIT_APPLICATION_ID:
				LOGINFO("Hotkey pressed, exiting application");
				goto Exit;
			}
		}
	}

	Exit:
	app.playLightEffect(exitEffect);
	UnregisterHotKey(NULL, TOGGLE_SERVER_ID);
	UnregisterHotKey(NULL, TOGGLE_AUDIO_VISUALIZER_ID);
	UnregisterHotKey(NULL, TOGGLE_DESKTOP_VISUALIZER_ID);
	UnregisterHotKey(NULL, EXIT_APPLICATION_ID);

	if (audioVisualizerRunning) app.stopAudioVisualizer();
	LOGINFO("Exiting application ----------------------------------------------");
	return 0;
}

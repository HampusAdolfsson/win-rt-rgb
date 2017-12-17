#include <Windows.h>
#include "RequestClient.h"
#include "Audio\AudioVisualizer.h"

#define ADDR "192.168.1.25"
#define TCP_PORT "8844"
#define UDP_PORT 8845
#define TOGGLE_VISUALIZER_KEY 0x56 // v key
#define TOGGLE_VISUALIZER_ID 0x1
#define EXIT_APPLICATION_KEY 0x42 // b key
#define EXIT_APPLICATION_ID 0x2

class Main
{
	RequestClient requestClient;
	AudioVisualizer visualizer;

public:
	Main(const WAVEFORMATEX& pwfx, const std::string& serverAddr, const std::string& tcpPort, const int& udpPort)
		: requestClient(serverAddr, tcpPort),
		visualizer(1, pwfx, serverAddr, udpPort)
	{
		visualizer.initialize();
	}

	void startVisualizer()
	{
		visualizer.start();
	}
	void stopVisualizer()
	{
		visualizer.stop();
	}

	void playLightEffect(const LightEffect& effect)
	{
		requestClient.sendLightEffect(effect, false);
	}

};

// Played when launched
LightEffect startEffect(1500000000, Breathing, { {0,0,60}, {0,0,150}, {0,0,255}, {0,0,150}, {0,0,60} });
// Played when exiting
LightEffect exitEffect(1500000000, Breathing, { {60,0,0}, {150,0,0}, {255,0,0}, {150,0,0}, {60,0,0} });

int main(int argc, char **argv) {
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
	
	Main main(pwfx, ADDR, TCP_PORT, UDP_PORT);
	main.playLightEffect(LightEffect(startEffect));
	main.startVisualizer();

	RegisterHotKey(NULL, TOGGLE_VISUALIZER_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, TOGGLE_VISUALIZER_KEY);
	RegisterHotKey(NULL, EXIT_APPLICATION_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, EXIT_APPLICATION_KEY);

	MSG msg;
	bool visualizerRunning = true;
	while (GetMessage(&msg, 0, 0, 0) == 1)
	{
		switch (msg.message)
		{
		case WM_HOTKEY:
			switch (msg.wParam)
			{
			case TOGGLE_VISUALIZER_ID:
				if (visualizerRunning) main.stopVisualizer();
				else				   main.startVisualizer();
				visualizerRunning = !visualizerRunning;
				break;
			case EXIT_APPLICATION_ID:
				goto Exit;
			}
		}
	}

	Exit:
	main.playLightEffect(exitEffect);
	UnregisterHotKey(NULL, TOGGLE_VISUALIZER_ID);

	if (visualizerRunning) main.stopVisualizer();
	return 0;
}

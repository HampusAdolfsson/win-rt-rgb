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


int main(int argc, char **argv) {
	WSAData wsa;
	WSAStartup(MAKEWORD(2,2), &wsa);

	RequestClient cl(ADDR, TCP_PORT);
	cl.sendLightEffect(LightEffect(1500000000, Breathing, { {0,0,60}, {0,0,150}, {0,0,255}, {0,0,150}, {0,0,60} }), true);

	WAVEFORMATEX pwfx;
	pwfx.wFormatTag = WAVE_FORMAT_PCM;
	pwfx.nChannels = 1;
	pwfx.nSamplesPerSec = 48000;
	pwfx.wBitsPerSample = 16;
	pwfx.nBlockAlign = pwfx.nChannels * (pwfx.wBitsPerSample / 8);
	pwfx.nAvgBytesPerSec = pwfx.nBlockAlign * pwfx.nSamplesPerSec;
	pwfx.cbSize = 0;
	
	AudioVisualizer cap(1, pwfx, ADDR, UDP_PORT);
	cap.initialize();
	cap.start();

	RegisterHotKey(NULL, TOGGLE_VISUALIZER_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, TOGGLE_VISUALIZER_KEY);
	RegisterHotKey(NULL, EXIT_APPLICATION_ID, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, EXIT_APPLICATION_KEY);

	MSG msg;
	bool capRunning = true;
	while (GetMessage(&msg, 0, 0, 0) == 1)
	{
		switch (msg.message)
		{
		case WM_HOTKEY:
			switch (msg.wParam)
			{
			case TOGGLE_VISUALIZER_ID:
				if (capRunning) cap.stop();
				else			cap.start();
				capRunning = !capRunning;
				break;
			case EXIT_APPLICATION_ID:
				goto Exit;
			}
		}
	}

	Exit:
	UnregisterHotKey(NULL, TOGGLE_VISUALIZER_ID);

	if (capRunning) cap.stop();
	return 0;
}

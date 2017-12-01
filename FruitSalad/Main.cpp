#include <Windows.h>
#include <Winuser.h>
#include "Audio\AudioVisualizer.h"

#define ADDR "192.168.1.25"
#define PORT 8845
#define TOGGLE_VISUALIZER_KEY 0x56 // X key
#define TOGGLE_VISUALIZER_ID 0x1

int main(int argc, char **argv) {
	WAVEFORMATEX pwfx;
	pwfx.wFormatTag = WAVE_FORMAT_PCM;
	pwfx.nChannels = 1;
	pwfx.nSamplesPerSec = 48000;
	pwfx.wBitsPerSample = 16;
	pwfx.nBlockAlign = pwfx.nChannels * (pwfx.wBitsPerSample / 8);
	pwfx.nAvgBytesPerSec = pwfx.nBlockAlign * pwfx.nSamplesPerSec;
	pwfx.cbSize = 0;
	
	AudioVisualizer cap(1, pwfx, ADDR, PORT);
	cap.initialize();
	cap.start();

	RegisterHotKey(NULL, TOGGLE_VISUALIZER_ID, MOD_CONTROL | MOD_SHIFT, TOGGLE_VISUALIZER_KEY);

	MSG msg;
	bool capRunning = true;
	while (GetMessage(&msg, 0, 0, 0) == 1)
	{
		switch (msg.message)
		{
		case WM_HOTKEY:
			printf("Hotkey pressed...\n");
			if (capRunning) cap.stop();
			else			cap.start();
			capRunning = !capRunning;
			break;
		}
	}

	UnregisterHotKey(NULL, TOGGLE_VISUALIZER_ID);

	if (capRunning) cap.stop();
	return 0;
}

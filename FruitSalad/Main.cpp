#include "Audio\AudioVisualizer.h"

namespace FruitSalad {
}
int main(int argc, char **argv) {
	//FruitSalad::doStuff();
	WAVEFORMATEX pwfx;
	pwfx.wFormatTag = WAVE_FORMAT_PCM;
	pwfx.nChannels = 1;
	pwfx.nSamplesPerSec = 48000;
	pwfx.wBitsPerSample = 16;
	pwfx.nBlockAlign = pwfx.nChannels * (pwfx.wBitsPerSample / 8);
	pwfx.nAvgBytesPerSec = pwfx.nBlockAlign * pwfx.nSamplesPerSec;
	pwfx.cbSize = 0;
	AudioVisualizer cap(1, pwfx);
	cap.initialize();
	cap.start();
	Sleep(2000);
	cap.stop();
	Sleep(4000);
	cap.start();
	Sleep(2000);
	return 0;
}

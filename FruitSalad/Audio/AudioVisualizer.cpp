#include <cstdio>
#include <cmath>
#include <bitset>
#include <iostream>
#include "AudioVisualizer.h"

#define BUFFERS_PER_S 100

#define RETURN_ON_ERROR(mres) do {\
						if (mres) {\
							fprintf(stderr, "ERROR: %d %s %d", mres, __FILE__, __LINE__);\
							return false;\
						}\
						} while(0)

AudioVisualizer::AudioVisualizer(const DWORD &devId, const WAVEFORMATEX &format)
	: deviceId(devId),
	pwfx(format),
	isRunning(false) {}

bool AudioVisualizer::initialize()
{
	bool opened = openDevice();
	if (!opened) return false;

	int buffer_len = pwfx.nBlockAlign * (pwfx.nSamplesPerSec / BUFFERS_PER_S);
	buffer.reserve(buffer_len * NUM_BUFFERS);
	MMRESULT mres;
	for (int i = 0; i < NUM_BUFFERS; i++)
	{
		waveHeaders[i].dwBufferLength = buffer_len;
		waveHeaders[i].lpData = buffer.data() + buffer_len * i;
		mres = waveInPrepareHeader(waveInHandle, waveHeaders + i, sizeof(waveHeaders[i]));
		RETURN_ON_ERROR(mres);
	}
	for (int i = 0; i < NUM_BUFFERS; i++)
	{
		mres = waveInAddBuffer(waveInHandle, waveHeaders + i, sizeof(waveHeaders[i]));
		RETURN_ON_ERROR(mres);
	}
	return true;
}

bool AudioVisualizer::start()
{
	MMRESULT mres = waveInStart(waveInHandle);
	RETURN_ON_ERROR(mres);
	isRunning = true;
	return true;
}

bool AudioVisualizer::stop()
{
	MMRESULT mres = waveInStop(waveInHandle);
	RETURN_ON_ERROR(mres);

	isRunning = false;
	return true;
}

bool AudioVisualizer::openDevice()
{

	DWORD threadId;
	handlerThread = std::thread(&AudioVisualizer::handleWaveMessages, this);
	threadId = GetThreadId(handlerThread.native_handle());

	MMRESULT mres = waveInOpen(&waveInHandle, deviceId, &pwfx, threadId, 0, CALLBACK_THREAD);
	RETURN_ON_ERROR(mres);
	return true;
}

AudioVisualizer::~AudioVisualizer()
{

	if (isRunning) waveInStop(waveInHandle);
	MMRESULT mres;
	for (int i = 0; i < NUM_BUFFERS; i++)
	{
		mres = waveInUnprepareHeader(waveInHandle, waveHeaders + i, sizeof(waveHeaders[i]));
		if (mres) break;
	}
	MMRESULT mres2 = waveInClose(waveInHandle);
	if (!mres && !mres2 && handlerThread.joinable())
	{
		handlerThread.join();
	}
	else
	{
		handlerThread.detach();
	}
}

void AudioVisualizer::handleWaveMessages()
{
	MSG msg;
	printf("Thread started\n");
	while (GetMessage(&msg, 0, 0, 0))
	{
		switch (msg.message)
		{
		case MM_WIM_DATA: {
			if (pwfx.wBitsPerSample != 16) printf("%s %d : WARNING! This code only supports 16-bit samples.\nPlease change the code.\n", __FILE__, __LINE__); // For performance reasons
			WAVEHDR *hdr = (WAVEHDR*)msg.lParam;
			if (hdr->dwBytesRecorded > 0)
			{
				int sampleSize = pwfx.wBitsPerSample / 8;
				long sqAvg = 0;
				for (int i = 0; i < hdr->dwBytesRecorded; i += sampleSize)
				{
					short sample = *((short*)(hdr->lpData + i));
					sqAvg += sample*sample; // We square the samples to get an absolute amplitude, roughly correlating to volume.
				}
				sqAvg /= (hdr->dwBytesRecorded / sampleSize);
			}
			waveInAddBuffer(waveInHandle, waveHeaders, sizeof(WAVEHDR));
			continue;
		}
		case MM_WIM_OPEN:
			printf("Opening device!\n");
			continue;
		case MM_WIM_CLOSE:
			printf("CLOSING? BYE.\n");
			return;
		}
	}
}

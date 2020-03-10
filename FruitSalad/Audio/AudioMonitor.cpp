#include <cstdio>
#include <iostream>
#include <bitset>
#include "Logger.h"
#include "AudioMonitor.h"

#pragma comment(lib, "winmm.lib")

#define BUFFERS_PER_S 120

#define RETURN_ON_ERROR(mres) do {\
						if (mres) {\
							LOGSEVERE("Audiovisualizer got error: %d, line %d", mres, __LINE__);\
							return false;\
						}\
						} while(0)

AudioMonitor::AudioMonitor(const DWORD &devId, const WAVEFORMATEX &format, std::function<void(const uint8_t&)> callback)
	: deviceId(devId),
	pwfx(format),
    callback(callback),
	isRunning(false) {}

bool AudioMonitor::initialize()
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

bool AudioMonitor::start()
{
	MMRESULT mres = waveInStart(waveInHandle);
	RETURN_ON_ERROR(mres);
	isRunning = true;
	return true;
}

bool AudioMonitor::stop()
{
	MMRESULT mres = waveInStop(waveInHandle);
	RETURN_ON_ERROR(mres);

	isRunning = false;
	return true;
}

bool AudioMonitor::openDevice()
{

	DWORD threadId;
	handlerThread = std::thread(&AudioMonitor::handleWaveMessages, this);
	threadId = GetThreadId(handlerThread.native_handle());

	MMRESULT mres = waveInOpen(&waveInHandle, deviceId, &pwfx, threadId, 0, CALLBACK_THREAD);
	RETURN_ON_ERROR(mres);
	return true;
}

AudioMonitor::~AudioMonitor()
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

void AudioMonitor::handleWaveMessages()
{
	MSG msg;
	while (GetMessage(&msg, 0, 0, 0))
	{
		switch (msg.message)
		{
		case MM_WIM_DATA: {
			if (pwfx.wBitsPerSample != 16) printf("%s %d : WARNING! This code only supports 16-bit samples.\nPlease change the code.\n", __FILE__, __LINE__); // For performance reasons
			WAVEHDR *hdr = (WAVEHDR*)msg.lParam;
			if (hdr->dwBytesRecorded > 0)
			{
				size_t sampleSize = pwfx.wBitsPerSample / 8; // TODO: benchmark doing stuff here instead
				uint8_t intensity = waveStrategy.getIntensity(hdr->lpData, hdr->dwBytesRecorded, sampleSize);
                callback(intensity);
			}
			for (int i = 0; i < NUM_BUFFERS; i++)
			{
				if (hdr == waveHeaders + i) {
					int mres = waveInAddBuffer(waveInHandle, waveHeaders + i, sizeof(waveHeaders[i]));
					if (mres) {
						LOGSEVERE("Audiovisualizer got error: %d, line %d", mres, __LINE__);
						return;
					}
				}
			}
			continue;
		}
		case MM_WIM_OPEN:
			continue;
		case MM_WIM_CLOSE:
			return;
		}
	}
}

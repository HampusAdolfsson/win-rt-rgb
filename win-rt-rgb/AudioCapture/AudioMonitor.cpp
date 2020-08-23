#include <cassert>
#include <iostream>
#include <bitset>
#include <mmdeviceapi.h>
#include "Logger.h"
#include "AudioMonitor.h"

// #pragma comment(lib, "winmm.lib")

#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MILLISEC  10000

static const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
static const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
static const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);
static const IID IID_IAudioClient = __uuidof(IAudioClient);

#define EXIT_ON_ERROR(hr) do {\
						if (FAILED(hr)) {\
							LOGSEVERE("Audiomonitor got error: 0x%08lx, line %d", hr, __LINE__);\
							goto Exit;\
						}\
						} while(0)

AudioMonitor::AudioMonitor(std::unique_ptr<WaveHandler> handler)
	: audioClient(nullptr),
	captureClient(nullptr),
	hnsRequestedDuration(REFTIMES_PER_SEC / 30),
	handler(std::move(handler)),
	isRunning(false) {}

bool AudioMonitor::initialize()
{
	bool opened = openDevice();
	if (!opened) return false;

	WAVEFORMATEX* pwfx = nullptr;
	HRESULT hr = audioClient->GetMixFormat(&pwfx);
	EXIT_ON_ERROR(hr);

	WAVEFORMATEXTENSIBLE* ex;
	ex = reinterpret_cast<WAVEFORMATEXTENSIBLE*>(pwfx);
	if (pwfx->wFormatTag != WAVE_FORMAT_EXTENSIBLE || ex->SubFormat != KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)
	{
		LOGSEVERE("The audio format used by the audio mixer is not supported by this application!");
		return false;
	}

	hr = audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED,
									AUDCLNT_STREAMFLAGS_LOOPBACK,
									hnsRequestedDuration,
									0,
									pwfx,
									nullptr);
	EXIT_ON_ERROR(hr);

	UINT32 bufferSize;
    hr = audioClient->GetBufferSize(&bufferSize);
    EXIT_ON_ERROR(hr);

	hr = audioClient->GetService(IID_IAudioCaptureClient, (void**) &captureClient);
	EXIT_ON_ERROR(hr);

	handler->setFormat(pwfx->nSamplesPerSec, pwfx->nChannels);

	hnsActualDuration = (double)REFTIMES_PER_SEC * bufferSize / pwfx->nSamplesPerSec;

Exit:
	if (pwfx)
	{
		CoTaskMemFree(pwfx);
	}
	return true;
}

bool AudioMonitor::start()
{
	HRESULT hr = audioClient->Start();
	EXIT_ON_ERROR(hr);
	isRunning = true;

	handlerThread = std::thread(&AudioMonitor::handleWaveMessages, this);

	return true;
Exit:
	return false;
}

bool AudioMonitor::stop()
{
	isRunning = false;
	return true;
}

bool AudioMonitor::openDevice()
{
	bool success = false;

	IMMDeviceEnumerator* enumerator = nullptr;
	IMMDevice *device = nullptr;

	HRESULT hr = CoCreateInstance(
		CLSID_MMDeviceEnumerator, NULL,
		CLSCTX_ALL, IID_IMMDeviceEnumerator,
		(void **)&enumerator);
	EXIT_ON_ERROR(hr);

	hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
	EXIT_ON_ERROR(hr);

	hr = device->Activate(IID_IAudioClient, CLSCTX_ALL, NULL, (void**) &audioClient);
	EXIT_ON_ERROR(hr);

	success = true;

Exit:
	if (enumerator != nullptr) enumerator->Release();
	if (device != nullptr) device->Release();

	return success;
}

AudioMonitor::~AudioMonitor()
{
	if (audioClient != nullptr) audioClient->Release();
	if (captureClient != nullptr) audioClient->Release();

	if (handlerThread.joinable())
	{
		handlerThread.join();
	}
}

void AudioMonitor::handleWaveMessages()
{
	while (isRunning)
	{
		// Sleep for half the buffer duration.
        Sleep(hnsActualDuration / 2 / REFTIMES_PER_MILLISEC);

		UINT32 packetSize;
        HRESULT hr = captureClient->GetNextPacketSize(&packetSize);
		EXIT_ON_ERROR(hr);

		while (packetSize > 0)
		{
			BYTE* packetData;
			UINT nFrames;
			DWORD flags;
			hr = captureClient->GetBuffer(&packetData, &nFrames, &flags, nullptr, nullptr);
			EXIT_ON_ERROR(hr);

			if (flags & AUDCLNT_BUFFERFLAGS_SILENT)
			{
				break;
			}

			handler->receiveBuffer(reinterpret_cast<float*>(packetData), nFrames);

			hr = captureClient->ReleaseBuffer(nFrames);
			EXIT_ON_ERROR(hr);
			hr = captureClient->GetNextPacketSize(&packetSize);
			EXIT_ON_ERROR(hr);
		}
	}

Exit:
	audioClient->Stop();
}

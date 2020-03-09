#include "DesktopColorSampler.h"
#include "Logger.h"
#include <d3d11.h>

#define RETURN_ON_ERROR(hr) do {\
						if (hr != S_OK) {\
							LOGSEVERE("DesktopColorSampler got error: %d, line %d", hr, __LINE__);\
							return;\
						}\
						} while(0)

DesktopColorSampler::DesktopColorSampler(const UINT& outputIdx)
	: desktopDuplicator(),
	frameSampler(),
	isRunning(false)
{
	HRESULT hr;

	hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL,
#ifdef _DEBUG
		D3D11_CREATE_DEVICE_DEBUG,
#else
		0,
#endif
		NULL, 0, D3D11_SDK_VERSION, &device, NULL, NULL);
	if (hr != S_OK)
	{
		LOGSEVERE("Failed to create d3d device");
		return;
	}
	device->GetImmediateContext(&deviceContext);

	desktopDuplicator.initialize(device, outputIdx);
	frameSampler.initialize(device, desktopDuplicator.getFrameWidth(), desktopDuplicator.getFrameHeight());

	sampleAvailSemaphore = CreateSemaphore(NULL, 0, 1, NULL);
	sampleRequestSemaphore = CreateSemaphore(NULL, 0, 1, NULL);

	// allocate our buffer
	D3D11_TEXTURE2D_DESC texDesc;
	RtlZeroMemory(&texDesc, sizeof(texDesc));
	texDesc.Width = desktopDuplicator.getFrameWidth();
	texDesc.Height = desktopDuplicator.getFrameHeight();
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	texDesc.SampleDesc.Count = 1;
	texDesc.Usage = D3D11_USAGE_STAGING;
	texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	hr = device->CreateTexture2D(&texDesc, NULL, &frameBuffer);
	if (hr != S_OK)
	{
		LOGSEVERE("Failed to create texture");
		return;
	}
}

DesktopColorSampler::~DesktopColorSampler()
{
	device->Release();
	deviceContext->Release();
}

void DesktopColorSampler::start()
{
	isRunning = true;
	samplerThread = std::thread(&DesktopColorSampler::sampleLoop, this);
	ReleaseSemaphore(sampleRequestSemaphore, 1, NULL);
}

void DesktopColorSampler::stop()
{
	isRunning = false;
	// TODO: should maybe claim the semaphore
	samplerThread.join();
}

Color DesktopColorSampler::getSample()
{
	WaitForSingleObject(sampleAvailSemaphore, INFINITE);
	ReleaseSemaphore(sampleRequestSemaphore, 1, NULL);
	return currentSample;
}

// run by worker thread
void DesktopColorSampler::sampleLoop()
{
	bool WaitToProcessCurrentFrame = false;
	while (isRunning)
	{
		WaitForSingleObject(sampleRequestSemaphore, INFINITE); // TODO: wait finite time
		ID3D11Texture2D* frame = desktopDuplicator.captureFrame();
		if (frame)
		{
			deviceContext->CopyResource(frameBuffer, frame);
			currentSample = frameSampler.sample(frameBuffer);
			desktopDuplicator.releaseFrame();
		}
		ReleaseSemaphore(sampleAvailSemaphore, 1, NULL);
	}
}

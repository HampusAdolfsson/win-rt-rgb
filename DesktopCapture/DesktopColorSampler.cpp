#include "DesktopColorSampler.h"
#include "Logger.h"
#include <d3d11.h>

#define RETURN_ON_ERROR(hr) do {\
						if (hr != S_OK) {\
							LOGSEVERE("DesktopColorSampler got error: %d, line %d", hr, __LINE__);\
							return;\
						}\
						} while(0)

DesktopColorSampler::DesktopColorSampler(const UINT& outputIdx, std::function<void(const RgbColor&)> callback)
	: desktopDuplicator(),
	frameSampler(),
	isRunning(false),
	callback(callback)
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
}

void DesktopColorSampler::stop()
{
	isRunning = false;
	samplerThread.join();
}

// run by worker thread
void DesktopColorSampler::sampleLoop()
{
	bool WaitToProcessCurrentFrame = false;
	while (isRunning)
	{
		ID3D11Texture2D* frame = desktopDuplicator.captureFrame();
		if (frame)
		{
			deviceContext->CopyResource(frameBuffer, frame);
			desktopDuplicator.releaseFrame();
			callback(frameSampler.sample(frameBuffer));
		}
	}
}

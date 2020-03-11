#include "DesktopColorSampler.h"
#include "Logger.h"
#include <d3d11.h>

DesktopColorSampler::DesktopColorSampler(const UINT& outputIdx, std::function<void(const RgbColor&)> callback)
	: desktopDuplicator(),
	frameSampler(),
	isRunning(false),
	callback(callback)
{
	ID3D11Device* device;

	HRESULT hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL,
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

	desktopDuplicator.initialize(device, outputIdx);
	frameSampler.initialize(device, desktopDuplicator.getFrameWidth(), desktopDuplicator.getFrameHeight());

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
	while (isRunning)
	{
		ID3D11Texture2D* frame = desktopDuplicator.captureFrame();
		if (frame)
		{
			frameSampler.setFrameData(frame);
			desktopDuplicator.releaseFrame();
			callback(frameSampler.sample());
		}
	}
}

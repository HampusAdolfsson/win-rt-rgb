#include "DesktopColorSampler.h"
#include "Logger.h"
#include <d3d11.h>
#include <optional>

DesktopColorSampler::DesktopColorSampler(const UINT& outputIdx,
											const std::vector<SamplingSpecification>& specifications,
											DesktopSamplingCallback callback)
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
	frameSampler.initialize(device, desktopDuplicator.getFrameWidth(), desktopDuplicator.getFrameHeight(), specifications);
	// by default, capture entire screen
	captureRegion = { 0, 0, desktopDuplicator.getFrameWidth(), desktopDuplicator.getFrameHeight() };

}

void DesktopColorSampler::setCaptureRegion(Rect captureRegion)
{
	this->captureRegion = captureRegion;
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
	std::vector<RgbColor*> lastResult;
	while (isRunning)
	{
		ID3D11Texture2D* frame = desktopDuplicator.captureFrame();
		if (frame)
		{
			frameSampler.setFrameData(frame);
			desktopDuplicator.releaseFrame();
			lastResult = frameSampler.sample(captureRegion);
		}
		for (size_t i = 0; i < lastResult.size(); i++)
		{
			callback(i, lastResult[i]);
		}
	}
}

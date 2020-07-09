#include "DesktopCaptureController.h"
#include "Logger.h"

DesktopCaptureController::DesktopCaptureController(const UINT& initialOutputIdx, const UINT& nSamples, std::function<void(RgbColor*)> callback)
	: isActive(false)
{
	activeOutput = initialOutputIdx;

	UINT nOutputs = getNumberOfOutputs();
	samplers = std::vector<DesktopColorSampler*>(nOutputs, nullptr);
	for (UINT i = 0; i < nOutputs; i++)
	{
		samplers[i] = new DesktopColorSampler(i, nSamples, callback);
	}
}

DesktopCaptureController::~DesktopCaptureController()
{
	samplers[activeOutput]->stop();
	for (int i = 0; i < samplers.size(); i++)
	{
		delete samplers[i];
	}
}

void DesktopCaptureController::setOutput(const UINT& outputIdx, Rect captureRegion)
{
	if (outputIdx != activeOutput)
	{
		if (outputIdx >= samplers.size())
		{
			LOGSEVERE("Tried to sample output that does not exist");
			return;
		}
		if (isActive)
		{
			samplers[activeOutput]->stop();
			samplers[outputIdx]->start();
		}
		activeOutput = outputIdx;
	}
	samplers[activeOutput]->setCaptureRegion(captureRegion);
}
void DesktopCaptureController::start()
{
	if (isActive) return;
	isActive = true;
	samplers[activeOutput]->start();
}
void DesktopCaptureController::stop()
{
	if (!isActive) return;
	isActive = false;
	samplers[activeOutput]->stop();
}

const UINT DesktopCaptureController::getNumberOfOutputs() const
{
	ID3D11Device* device;
	HRESULT hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &device, NULL, NULL);
	if (hr != S_OK)
	{
		LOGSEVERE("Unable to enumerate outputs");
		return 0;
	}

	IDXGIDevice* dxgiDevice;
	hr = device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
	device->Release();
	if (FAILED(hr))
	{
		LOGSEVERE("Unable to enumerate outputs");
		return 0;
	}
	IDXGIAdapter1* adapter;
	hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&adapter));
	dxgiDevice->Release();
	if (FAILED(hr))
	{
		LOGSEVERE("Unable to enumerate outputs");
		return 0;
	}

	UINT nOutputs = 0;
	IDXGIOutput* output;
	while (adapter->EnumOutputs(nOutputs, &output) != DXGI_ERROR_NOT_FOUND)
	{
		output->Release();
		++nOutputs;
	}
	adapter->Release();
	return nOutputs;
}


#include "DesktopDuplicator.h"
#include "Logger.h"
#include <dxgi1_2.h>
#include <d3d11.h>

#define FRAME_TIMEOUT 5000

DesktopDuplicator::DesktopDuplicator()
: device(nullptr),
outputDuplication(nullptr),
currentFrame(nullptr)
{
	RtlZeroMemory(&outputDesc, sizeof(outputDesc));
}

DuplReturn_t DesktopDuplicator::initialize(ID3D11Device *device, UINT outputIdx) {

	this->device = device;
	this->device->AddRef();

	HRESULT hr;

	// get adapter from device
	IDXGIDevice *dxgiDevice;
	hr = device->QueryInterface(__uuidof(IDXGIDevice), (void**) &dxgiDevice);
	if (FAILED(hr)) {
		return ProcessError(device, hr, nullptr);
	}
	IDXGIAdapter1* adapter;
	hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&adapter));
	dxgiDevice->Release();
	if (FAILED(hr)) {
		return ProcessError(device, hr, SystemTransitionsExpectedErrors);
	}

	// get output from adapter
	IDXGIOutput* output;
	hr = adapter->EnumOutputs(outputIdx, &output);
	adapter->Release();
	if (FAILED(hr))
	{
		return ProcessError(device, hr, EnumOutputsExpectedErrors);
	}
	output->GetDesc(&outputDesc);

	IDXGIOutput1* output1;
	hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**) &output1);
	output->Release();
	if (FAILED(hr))
	{
		return ProcessError(device, hr, nullptr);
	}

	// create duplicator for output
	hr = output1->DuplicateOutput(device, &outputDuplication);
	output1->Release();
	if (FAILED(hr))
	{
		if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE)
		{
			LOGSEVERE("API is currently inavailable");
			return DUPL_RETURN_ERROR_UNEXPECTED;
		}
		return ProcessError(device, hr, CreateDuplicationExpectedErrors);
	}

	return DUPL_RETURN_SUCCESS;
}

DuplReturn_t DesktopDuplicator::captureFrame(_Out_ ID3D11Texture2D** frame, _Out_ bool *timedOut) {
    IDXGIResource* desktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;

    // Get new frame
    HRESULT hr = outputDuplication->AcquireNextFrame(500, &frameInfo, &desktopResource);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        *timedOut = true;
		LOGWARNING("Frame duplication timed out");
        return DUPL_RETURN_SUCCESS;
    }
    *timedOut = false;

    if (FAILED(hr))
    {
        return ProcessError(device, hr, FrameInfoExpectedErrors);
    }

    // If still holding old frame, destroy it
    if (currentFrame)
    {
        currentFrame->Release();
        currentFrame = nullptr;
    }

    // QI for IDXGIResource
    hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&currentFrame));
    desktopResource->Release();
    desktopResource = nullptr;
    if (FAILED(hr))
    {
        return ProcessError(nullptr, hr, nullptr);
    }

	*frame = currentFrame;

    return DUPL_RETURN_SUCCESS;
}

DuplReturn_t DesktopDuplicator::releaseFrame() {
    HRESULT hr = outputDuplication->ReleaseFrame();
    if (FAILED(hr))
    {
        return ProcessError(device, hr, FrameInfoExpectedErrors);
    }

    if (currentFrame)
    {
        currentFrame->Release();
        currentFrame = nullptr;
    }

    return DUPL_RETURN_SUCCESS;
}

UINT DesktopDuplicator::getFrameWidth() {
	return outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
}
UINT DesktopDuplicator::getFrameHeight() {
	return outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
}

DesktopDuplicator::~DesktopDuplicator() {
	if (device) {
		device->Release();
	}
	if (outputDuplication) {
		outputDuplication->Release();
	}
    if (currentFrame) {
        currentFrame->Release();
    }
}

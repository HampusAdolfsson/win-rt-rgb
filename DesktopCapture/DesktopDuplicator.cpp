#include "DesktopDuplicator.h"
#include "Logger.h"
#include <dxgi1_2.h>
#include <d3d11.h>

#define FRAME_TIMEOUT 500

constexpr HRESULT duplicationExpectedErrors[] = {
	DXGI_ERROR_INVALID_CALL,
	DXGI_ERROR_DEVICE_REMOVED,
	DXGI_ERROR_ACCESS_LOST,
	S_OK
};

DesktopDuplicator::DesktopDuplicator()
: device(nullptr),
outputDuplication(nullptr),
currentFrame(nullptr)
{
	RtlZeroMemory(&outputDesc, sizeof(outputDesc));
}

void DesktopDuplicator::initialize(ID3D11Device *device, UINT outputIdx) {

	this->device = device;
	this->device->AddRef();
	this->outputIdx = outputIdx;
	reInitialize();
}

// TODO: use mutex
ID3D11Texture2D* DesktopDuplicator::captureFrame() {
    IDXGIResource* desktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
	HRESULT hr;

    // Get new frame
	while (true) {
		hr = outputDuplication->AcquireNextFrame(FRAME_TIMEOUT, &frameInfo, &desktopResource);
		if (SUCCEEDED(hr) && (frameInfo.TotalMetadataBufferSize > 0 || frameInfo.LastPresentTime.QuadPart > 0)) {
			break;
		}
		else if (FAILED(hr)) {
			if (isExpectedError(hr)) {
				LOGINFO("Reinitializing duplication");
				reInitialize();
				continue;
			}
			else if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
				LOGWARNING("Frame duplication timed out");
			}
			else {
				LOGSEVERE("Unexpected error occured for desktop duplication");
				throw(hr);
			}
		}
		outputDuplication->ReleaseFrame();
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
		LOGSEVERE("Failed to query for frame");
		return nullptr;
    }

	return currentFrame;
}

void DesktopDuplicator::releaseFrame() {
    HRESULT hr = outputDuplication->ReleaseFrame();
    if (FAILED(hr))
    {
		LOGSEVERE("Failed to release frame");
		return;
    }

    if (currentFrame)
    {
        currentFrame->Release();
        currentFrame = nullptr;
    }
}

UINT DesktopDuplicator::getFrameWidth() {
	return outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
}
UINT DesktopDuplicator::getFrameHeight() {
	return outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
}

void DesktopDuplicator::reInitialize() {
	if (outputDuplication) {
		outputDuplication->Release();
	}
	if (currentFrame) {
		currentFrame->Release();

	}

	HRESULT hr;

	// get adapter from device
	IDXGIDevice *dxgiDevice;
	hr = device->QueryInterface(__uuidof(IDXGIDevice), (void**) &dxgiDevice);
	if (FAILED(hr)) {
		LOGSEVERE("Failed to get dxgiDevice");
		return;
	}
	IDXGIAdapter1* adapter;
	hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&adapter));
	dxgiDevice->Release();
	if (FAILED(hr)) {
		LOGSEVERE("Failed to get adapter");
		return;
	}

	// get output from adapter
	IDXGIOutput* output;
	hr = adapter->EnumOutputs(outputIdx, &output);
	adapter->Release();
	if (FAILED(hr))
	{
		LOGSEVERE("Failed to get output");
		return;
	}
	output->GetDesc(&outputDesc);

	IDXGIOutput1* output1;
	hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**) &output1);
	output->Release();
	if (FAILED(hr))
	{
		LOGSEVERE("Failed to get output1");
		return;
	}

	// create duplicator for output
	hr = output1->DuplicateOutput(device, &outputDuplication);
	output1->Release();
	if (FAILED(hr))
	{
		LOGSEVERE("Failed to duplicate output");
		return;
	}
}

bool DesktopDuplicator::isExpectedError(HRESULT hr) {
	const HRESULT* error = duplicationExpectedErrors;
	while (*error != S_OK)
	{
		if (*(error++) == hr)
		{
			return true;
		}
	}
	return false;
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
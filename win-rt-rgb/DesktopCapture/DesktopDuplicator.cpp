#include "DesktopDuplicator.h"
#include "Logger.h"
#include <dxgi1_2.h>
#include <d3d11.h>

using namespace DesktopCapture;

#define FRAME_TIMEOUT 250

constexpr HRESULT duplicationExpectedErrors[] = {
	DXGI_ERROR_INVALID_CALL,
	DXGI_ERROR_DEVICE_REMOVED,
	DXGI_ERROR_ACCESS_LOST,
	S_OK
};

DesktopDuplicator::DesktopDuplicator(ID3D11Device* device, UINT outputIdx)
	: device(nullptr),
	outputDuplication(nullptr),
	currentFrame(nullptr)
{
	RtlZeroMemory(&outputDesc, sizeof(outputDesc));
	this->device = device;
	this->device->AddRef();
	this->outputIdx = outputIdx;
	reInitialize();
}
DesktopDuplicator::DesktopDuplicator(DesktopDuplicator &&other)
{
	device = other.device;
	outputDesc = other.outputDesc;
	outputDuplication = other.outputDuplication;
	outputIdx = other.outputIdx;
	currentFrame = other.currentFrame;

	other.device = nullptr;
	other.outputDuplication = nullptr;
	other.currentFrame = nullptr;
}

// TODO: use mutex
ID3D11Texture2D* DesktopDuplicator::captureFrame()
{
	IDXGIResource* desktopResource = nullptr;
	DXGI_OUTDUPL_FRAME_INFO frameInfo;
	HRESULT hr;

	// Get new frame
	while (true)
	{
		hr = outputDuplication->AcquireNextFrame(FRAME_TIMEOUT, &frameInfo, &desktopResource);
		if (SUCCEEDED(hr) && (frameInfo.TotalMetadataBufferSize > 0 || frameInfo.LastPresentTime.QuadPart > 0))
		{
			break;
		}
		else if (FAILED(hr))
		{
			if (isExpectedError(hr))
			{
				LOGINFO("Reinitializing duplication");
				reInitialize();
				continue;
			}
			else if (hr == DXGI_ERROR_WAIT_TIMEOUT)
			{
				outputDuplication->ReleaseFrame();
				return nullptr;
			}
			else
			{
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

	if (desktopResource)
	{
		hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&currentFrame));
		desktopResource->Release();
		desktopResource = nullptr;
		if (FAILED(hr))
		{
			LOGSEVERE("Failed to query for frame");
			return nullptr;
		}
	}

	return currentFrame;
}

void DesktopDuplicator::releaseFrame()
{
	HRESULT hr = outputDuplication->ReleaseFrame();
	if (FAILED(hr))
	{
		LOGSEVERE("Failed to release frame: %x", hr);
		return;
	}

	if (currentFrame)
	{
		currentFrame->Release();
		currentFrame = nullptr;
	}
}

const UINT DesktopDuplicator::getFrameWidth() const
{
	return outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
}
const UINT DesktopDuplicator::getFrameHeight() const
{
	return outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
}

void DesktopDuplicator::reInitialize()
{
	if (outputDuplication)
	{
		outputDuplication->Release();
	}
	if (currentFrame)
	{
		currentFrame->Release();

	}
	outputDuplication = nullptr;

	HRESULT hr;

	// get adapter from device
	IDXGIDevice* dxgiDevice;
	hr = device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
	if (FAILED(hr))
	{
		LOGSEVERE("Failed to get dxgiDevice");
		return;
	}
	IDXGIAdapter1* adapter;
	hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&adapter));
	dxgiDevice->Release();
	if (FAILED(hr))
	{
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
	hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&output1);
	output->Release();
	if (FAILED(hr))
	{
		LOGSEVERE("Failed to get output1");
		return;
	}

	// create duplicator for output
	hr = output1->DuplicateOutput(device, &outputDuplication);
	output1->Release();
	if (hr == E_ACCESSDENIED)
	{
		Sleep(100);
		reInitialize();
	}
	else if (FAILED(hr))
	{
		LOGSEVERE("Failed to duplicate output");
		return;
	}
}

bool DesktopDuplicator::isExpectedError(const HRESULT& hr)
{
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

DesktopDuplicator::~DesktopDuplicator()
{
	if (device)
	{
		device->Release();
	}
	if (outputDuplication)
	{
		outputDuplication->Release();
	}
	if (currentFrame)
	{
		currentFrame->Release();
	}
}

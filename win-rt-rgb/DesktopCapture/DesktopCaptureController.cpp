#include "DesktopCaptureController.h"
#include "Logger.h"
#include <algorithm>

using namespace DesktopCapture;

DesktopCaptureController::DesktopCaptureController(const std::vector<std::pair<SamplingSpecification, DesktopSamplingCallback>>& outputSpecifications)
	: isActive(false),
	assignedCallbacks(outputSpecifications.size()),
	assignedHandles(outputSpecifications.size())
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

	UINT nOutputs = getNumberOfMonitors();
	for (UINT i = 0; i < nOutputs; i++)
	{
		duplicators.emplace_back(device, i, std::bind(&DesktopCaptureController::frameCallback, this, i, std::placeholders::_1));
		colorSamplers.emplace_back(device, duplicators[i].getFrameWidth(), duplicators[i].getFrameHeight());
		captureRegions.push_back({0, 0, duplicators[i].getFrameWidth(), duplicators[i].getFrameHeight()});
	}
	this->outputSpecifications.reserve(outputSpecifications.size());
	for (UINT i = 0; i < outputSpecifications.size(); i++)
	{
		auto& spec = outputSpecifications[i];
		this->outputSpecifications.push_back({D3DMeanColorSpecificationHandle(spec.first), spec.second});
		// Assign all outputs to monitor 0 by default
		assignedHandles[0].push_back(&(this->outputSpecifications[i].first));
		assignedCallbacks[0].push_back(spec.second);
		outputAssignments.push_back({0, i});
	}
}

DesktopCaptureController::~DesktopCaptureController()
{
	for (int i = 0; i < duplicators.size(); i++)
	{
		duplicators[i].stop();
	}
}

void DesktopCaptureController::setCaptureRegionForMonitor(UINT monitorIdx, Rect captureRegion)
{
	if (monitorIdx >= duplicators.size())
	{
		LOGSEVERE("Tried to sample monitor that does not exist");
		return;
	}
	// TODO: use mutex
	captureRegions[monitorIdx] = captureRegion;
}
void DesktopCaptureController::setCaptureMonitorForOutput(UINT outputIdx, UINT monitorIdx)
{
	if (outputIdx >= outputAssignments.size())
	{
		LOGSEVERE("Tried to set monitor for output spec that does not exist");
		return;
	}
	// TODO: use mutex
	const auto& prevAssignment = outputAssignments[outputIdx];
	auto handleIt = assignedHandles[prevAssignment.first].begin() + prevAssignment.second;
	auto callbackIt = assignedCallbacks[prevAssignment.first].begin() + prevAssignment.second;
	D3DMeanColorSpecificationHandle* handle = *handleIt;
	DesktopSamplingCallback callback = *callbackIt;
	assignedHandles[prevAssignment.first].erase(handleIt);
	assignedCallbacks[prevAssignment.first].erase(callbackIt);

	assignedHandles[monitorIdx].push_back(handle);
	assignedCallbacks[monitorIdx].push_back(callback);
	std::pair<size_t, size_t> newAssignment = {monitorIdx, assignedCallbacks[monitorIdx].size()};
	outputAssignments[outputIdx] = newAssignment;
	if (assignedCallbacks[prevAssignment.first].size() == 0) duplicators[prevAssignment.first].stop();
	if (assignedCallbacks[monitorIdx].size() == 1) duplicators[monitorIdx].start();
}

void DesktopCaptureController::start()
{
	if (isActive) return;
	isActive = true;
	for (int i = 0; i < duplicators.size(); i++)
	{
		if (assignedCallbacks[i].size() > 0)
		{
			duplicators[i].start();
		}
	}
}
void DesktopCaptureController::stop()
{
	if (!isActive) return;
	isActive = false;
	for (int i = 0; i < duplicators.size(); i++)
	{
		duplicators[i].stop();
	}
}

void DesktopCaptureController::frameCallback(UINT monitorIdx, ID3D11Texture2D* texture)
{
	if (texture && assignedCallbacks[monitorIdx].size() > 0)
	{
		colorSamplers[monitorIdx].setFrameData(texture);
		duplicators[monitorIdx].releaseFrame();
		// TODO: store previous results
		colorSamplers[monitorIdx].sample(assignedHandles[monitorIdx], captureRegions[monitorIdx]);
		for (int i = 0; i < assignedCallbacks[monitorIdx].size(); i++)
		{
			assignedCallbacks[monitorIdx][i](assignedHandles[monitorIdx][i]->getResults().data());
		}
	}
}

const UINT DesktopCaptureController::getNumberOfMonitors() const
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


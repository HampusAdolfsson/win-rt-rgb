#include "DesktopCaptureController.h"
#include "Logger.h"
#include <algorithm>

using namespace DesktopCapture;

DesktopCaptureController::DesktopCaptureController(const std::vector<std::pair<SamplingSpecification, DesktopSamplingCallback>>& outputSpecifications)
	: isActive(false),
	assignedCallbacks(outputSpecifications.size()),
	assignedHandles(outputSpecifications.size())
{
	// Initialize per-monitor data
	UINT nOutputs = getNumberOfMonitors();
	for (UINT i = 0; i < nOutputs; i++)
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
		duplicators.emplace_back(device, i);
		colorSamplers.emplace_back(device, duplicators[i].getFrameWidth(), duplicators[i].getFrameHeight());
		captureRegions.push_back({0, 0, duplicators[i].getFrameWidth(), duplicators[i].getFrameHeight()});
	}
	samplingWorkers = std::vector<std::thread>(nOutputs);
	workerRunning = std::vector<bool>(nOutputs, false);
	handlesLocks = std::vector<std::mutex>(nOutputs);

	this->outputSpecifications.reserve(outputSpecifications.size());
	for (UINT i = 0; i < outputSpecifications.size(); i++)
	{
		auto& spec = outputSpecifications[i];
		this->outputSpecifications.push_back({D3DMeanColorSpecificationHandle(spec.first), spec.second});
		// Assign all outputs to monitor 0 by default
		assignedHandles[0].push_back(&(this->outputSpecifications[i].first));
		assignedCallbacks[0].push_back(spec.second);
	}
}

DesktopCaptureController::~DesktopCaptureController()
{
	for (int i = 0; i < duplicators.size(); i++)
	{
		stopWorker(i);
	}
}

void DesktopCaptureController::setCaptureRegionForMonitor(UINT monitorIdx, Rect captureRegion)
{
	if (monitorIdx >= duplicators.size())
	{
		LOGSEVERE("Tried to sample monitor that does not exist");
		return;
	}
	// TODO: use a mutex
	captureRegions[monitorIdx] = captureRegion;
}
void DesktopCaptureController::setCaptureMonitorForOutput(UINT outputIdx, UINT monitorIdx)
{
	if (outputIdx >= outputSpecifications.size())
	{
		LOGSEVERE("Tried to set monitor for output spec that does not exist");
		return;
	}
	// Find if this output is already assigned to a monitor
	std::pair<size_t, size_t> prevAssignment(-1, -1);
	for (int mon = 0; mon < assignedHandles.size(); mon++)
	{
		for (int i = 0; i < assignedHandles.at(mon).size(); i++)
		{
			if (assignedHandles.at(mon).at(i) == &outputSpecifications.at(outputIdx).first)
			{
				prevAssignment = {mon, i};
				break;
			}
		}
	}
	if (prevAssignment.first == monitorIdx) { return; };

	// handlesLocks[prevAssignment.first].lock();
	if (prevAssignment.first >= 0)
	{
		auto handleIt = assignedHandles[prevAssignment.first].begin() + prevAssignment.second;
		auto callbackIt = assignedCallbacks[prevAssignment.first].begin() + prevAssignment.second;
		assignedHandles[prevAssignment.first].erase(handleIt);
		assignedCallbacks[prevAssignment.first].erase(callbackIt);
	}
	D3DMeanColorSpecificationHandle* handle = &outputSpecifications[outputIdx].first;
	DesktopSamplingCallback callback = outputSpecifications[outputIdx].second;

	// handlesLocks[monitorIdx].lock();

	assignedHandles[monitorIdx].push_back(handle);
	assignedCallbacks[monitorIdx].push_back(callback);
	if (assignedCallbacks[prevAssignment.first].size() == 0) stopWorker(prevAssignment.first);
	if (assignedCallbacks[monitorIdx].size() == 1 && !workerRunning[monitorIdx]) startWorker(monitorIdx);

	// handlesLocks[monitorIdx].unlock();
	// handlesLocks[prevAssignment.first].unlock();
}

void DesktopCaptureController::start()
{
	if (isActive) return;
	isActive = true;
	for (int i = 0; i < samplingWorkers.size(); i++)
	{
		if (assignedCallbacks[i].size() > 0)
		{
			startWorker(i);
		}
	}
}
void DesktopCaptureController::stop()
{
	if (!isActive) return;
	isActive = false;
	for (int i = 0; i < samplingWorkers.size(); i++)
	{
		stopWorker(i);
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

void DesktopCaptureController::startWorker(UINT monitorIdx)
{
	if (workerRunning[monitorIdx]) return;
	workerRunning[monitorIdx] = true;
	samplingWorkers[monitorIdx] = std::thread(&DesktopCaptureController::samplingLoop, this, monitorIdx);
}
void DesktopCaptureController::stopWorker(UINT monitorIdx)
{
	if (!workerRunning[monitorIdx]) return;
	workerRunning[monitorIdx] = false;
	samplingWorkers[monitorIdx].join();
}

void DesktopCaptureController::samplingLoop(UINT monitorIdx)
{
	DesktopDuplicator& dup = duplicators[monitorIdx];
	D3DMeanColorCalculator& sampler = colorSamplers[monitorIdx];
	std::mutex& lock = handlesLocks[monitorIdx];
	while(workerRunning[monitorIdx])
	{
		ID3D11Texture2D* frame = dup.captureFrame();
		// lock.lock();
		if (frame)
		{
			sampler.setFrameData(frame);
			dup.releaseFrame();
			sampler.sample(assignedHandles[monitorIdx], captureRegions[monitorIdx]);
		}
		for (int i = 0; i < assignedCallbacks[monitorIdx].size(); i++)
		{
			assignedCallbacks[monitorIdx][i](assignedHandles[monitorIdx][i]->getResults().data());
		}
		// lock.unlock();
	}
}
#include "DesktopCaptureController.h"
#include "Logger.h"
#include <algorithm>

using namespace DesktopCapture;

DesktopCaptureController::DesktopCaptureController()
	: isActive(false)
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

}

DesktopCaptureController::~DesktopCaptureController()
{
	for (int i = 0; i < duplicators.size(); i++)
	{
		stopWorker(i);
	}
}

void DesktopCaptureController::setOutputSpecifications(const std::vector<std::pair<size_t, DesktopSamplingCallback>>& outputSpecifications)
{
	UINT nOutputs = getNumberOfMonitors();
	assignedCallbacks.clear();
	assignedCallbacks.resize(nOutputs);
	assignedBuffers.clear();
	assignedBuffers.resize(nOutputs);
	this->outputSpecifications.clear();
	this->outputSpecifications.reserve(outputSpecifications.size());
	for (UINT i = 0; i < outputSpecifications.size(); i++)
	{
		auto& spec = outputSpecifications[i];
		this->outputSpecifications.push_back({ColorBuffer(spec.first), spec.second});
		// Assign all outputs to monitor 0 by default
		assignedBuffers[0].push_back(&(this->outputSpecifications[i].first));
		assignedCallbacks[0].push_back(spec.second);
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
		LOGSEVERE("Tried to set monitor for output spec %d, which does not exist", outputIdx);
		return;
	}
	// Find if this output is already assigned to a monitor
	std::pair<size_t, size_t> prevAssignment(-1, -1);
	for (int mon = 0; mon < assignedBuffers.size(); mon++)
	{
		for (int i = 0; i < assignedBuffers.at(mon).size(); i++)
		{
			if (assignedBuffers.at(mon).at(i) == &outputSpecifications.at(outputIdx).first)
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
		auto handleIt = assignedBuffers[prevAssignment.first].begin() + prevAssignment.second;
		auto callbackIt = assignedCallbacks[prevAssignment.first].begin() + prevAssignment.second;
		assignedBuffers[prevAssignment.first].erase(handleIt);
		assignedCallbacks[prevAssignment.first].erase(callbackIt);
	}
	ColorBuffer* buffer = &outputSpecifications[outputIdx].first;
	DesktopSamplingCallback callback = outputSpecifications[outputIdx].second;

	// handlesLocks[monitorIdx].lock();

	assignedBuffers[monitorIdx].push_back(buffer);
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

std::pair<UINT, UINT> DesktopCaptureController::getMonitorDimensions(UINT monitorIdx)
{
	return std::make_pair(this->duplicators[monitorIdx].getFrameWidth(), this->duplicators[monitorIdx].getFrameHeight());
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
		lock.lock();
		if (frame)
		{
			sampler.setFrameData(frame);
			dup.releaseFrame();
			sampler.sample(assignedBuffers[monitorIdx], captureRegions[monitorIdx]);
		}
		for (int i = 0; i < assignedCallbacks[monitorIdx].size(); i++)
		{
			assignedCallbacks[monitorIdx][i](assignedBuffers[monitorIdx][i]->data());
		}
		lock.unlock();
		Sleep(25);
	}
}
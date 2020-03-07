#include "DesktopCaptureController.h"
#include "Logger.h"

DesktopCaptureController::DesktopCaptureController(UINT initialOutputIdx)
{
	activeOutput = initialOutputIdx;

    expectedErrorEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    unexpectedErrorEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
	initialize();
}

DesktopCaptureController::~DesktopCaptureController() {
	for (int i = 0; i < samplers.size(); i++) {
		samplers[i]->stop();
		delete samplers[i];
	}
}

void DesktopCaptureController::setOutput(UINT outputIdx) {
	activeOutput = outputIdx;
}

Color DesktopCaptureController::getColor() {
	DesktopColorSampler *sampler = samplers[activeOutput];
	if (!sampler) {
		LOGSEVERE("Tried to sample output for which there was no sample");
		return { 0, 0, 0 };
	}
	return sampler->getSample();
}


void DesktopCaptureController::initialize() {
	UINT nOutputs = 1;//getNumberOfOutputs();
	samplers = std::vector<DesktopColorSampler*>(nOutputs, nullptr);
	for (int i = 0; i < nOutputs; i++) {
		samplers[i] = new DesktopColorSampler(i, expectedErrorEvent, unexpectedErrorEvent);
		samplers[i]->start();
	}
}


UINT DesktopCaptureController::getNumberOfOutputs() {
	ID3D11Device* device;
    HRESULT hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &device, NULL, NULL);
    if (hr != S_OK) {
        LOGSEVERE("Unable to enumerate outputs");
        return 0;
    }

	IDXGIDevice* dxgiDevice;
	hr = device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
	device->Release();
	if (FAILED(hr)) {
        LOGSEVERE("Unable to enumerate outputs");
		return 0;
	}
	IDXGIAdapter1* adapter;
	hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&adapter));
	dxgiDevice->Release();
	if (FAILED(hr)) {
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

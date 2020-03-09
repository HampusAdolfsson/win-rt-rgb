#pragma once
#include "DesktopDuplicator.h"
#include "D3DMeanColorCalculator.h"
#include "Color.h"
#include <thread>

/**
*	Samples the screen and returns a color representing its content.
*/
class DesktopColorSampler
{
	DesktopDuplicator desktopDuplicator;
	D3DMeanColorCalculator frameSampler;
	ID3D11Device* device;
	ID3D11DeviceContext* deviceContext;
	ID3D11Texture2D* frameBuffer;

	HANDLE sampleAvailSemaphore; // Signals when a sample is available
	HANDLE sampleRequestSemaphore; // Signals when a sample is needed
	std::thread samplerThread;
	bool isRunning;
	RgbColor currentSample;

	void sampleLoop();

public:
	/**
	*	Create a new sampler.
	*	@param outputIdx The index of the output (monitor) to sample
	*/
	DesktopColorSampler(const UINT& outputIdx);
	~DesktopColorSampler();

	RgbColor getSample();

	void start();
	void stop();
};


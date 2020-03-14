#pragma once
#include "DesktopDuplicator.h"
#include "D3DMeanColorCalculator.h"
#include "Color.h"
#include "Rect.h"
#include <thread>
#include <functional>

/**
*	Samples the screen and returns a color representing its content.
*/
class DesktopColorSampler
{
	DesktopDuplicator desktopDuplicator;
	D3DMeanColorCalculator frameSampler;
	Rect captureRegion;

	std::thread samplerThread;
	bool isRunning;
	std::function<void(const RgbColor&)> callback;

	void sampleLoop();

public:
	/**
	*	Create a new sampler.
	*	@param outputIdx The index of the output (monitor) to sample
	*	@param callback To call when a sample is generated
	*/
	DesktopColorSampler(const UINT& outputIdx, std::function<void(const RgbColor&)> callback);

	void setCaptureRegion(Rect captureRegion);

	void start();
	void stop();
};


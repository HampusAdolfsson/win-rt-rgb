#pragma once
#include "DesktopDuplicator.h"
#include "D3DMeanColorCalculator.h"
#include "Color.h"
#include "Rect.h"
#include <thread>
#include <functional>
#include <vector>

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
	std::function<void(RgbColor*)> callback;

	void sampleLoop();

public:
	/**
	*	Create a new sampler.
	*	@param outputIdx The index of the output (monitor) to sample
	*	@param nSamples	The number of color values to produce each frame
	*	@param callback	Called when samples are ready for a captured frame. The parameter points to an array of nSamples color values. The values of the array may be overwritten by the callee.
	*/
	DesktopColorSampler(const UINT& outputIdx, const UINT& nSamples, std::function<void(RgbColor*)> callback);

	void setCaptureRegion(Rect captureRegion);

	void start();
	void stop();
};


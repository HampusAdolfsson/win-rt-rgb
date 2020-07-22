#pragma once

#include "DesktopColorSampler.h"
#include "Color.h"
#include <vector>
#include <functional>

/**
*	Manages/multiplexes DesktopColorSamplers for different outputs, and restarts them when errors occur
*	(e.g. when entering fullscreen applications)
*/
class DesktopCaptureController
{
	UINT activeOutput;
	std::vector<DesktopColorSampler*> samplers;
	bool isActive;

	const UINT getNumberOfOutputs() const;

public:
	/**
	*	Creates a new capture controller
	*	@param initialOutputIdx Index of the output (i.e. monitor) to start capturing
	*	@param nSamples	The number of color values to produce each frame
	*	@param callback	Called when samples are ready for a captured frame. The parameter points to an array of nSamples color values. The values of the array may be overwritten by the callee.
	*/
	DesktopCaptureController(const UINT& initialOutputIdx, const UINT& nSamples, std::function<void(RgbColor*)> callback);
	~DesktopCaptureController();

	/**
	*	Sets the output (i.e. monitor) and region to capture from
	*/
	void setOutput(const UINT& outputIdx, Rect captureRegion);

	void start();
	void stop();

	DesktopCaptureController(DesktopCaptureController const&) = delete;
	DesktopCaptureController(DesktopCaptureController &&) = delete;
	DesktopCaptureController operator=(DesktopCaptureController const&) = delete;
};


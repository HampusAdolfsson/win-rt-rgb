#pragma once

#include "DesktopColorSampler.h"
#include "Types.h"
#include "Color.h"
#include <vector>
#include <functional>

/**
*	Continuosly captures and samples colors from one monitor at a time, calling back as
*	colors are generated. The sampling takes several parameters in the form of a SamplingSpecification object,
*	changing e.g. how to divide a frame into color regions. Every frame captured on a monitor will be
*	sampled according to the specification(s) given. Several SamplingSpecifications can be supplied,
*	meaning each frame will be sampled multiple times with different parameters.
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
	*	@param initialOutputIdx Index of the monitor to start capturing
	*	@param samplingParameters For each frame, an array of colors will be generated for each sampling specification,
	*		and the colors will provided to the associated callback function.
	*/
	DesktopCaptureController(UINT initialOutputIdx,
								const std::vector<std::pair<SamplingSpecification, DesktopSamplingCallback>>& samplingParameters);
	~DesktopCaptureController();

	/**
	*	Sets the monitor and region to capture from
	*/
	void setCaptureRegion(UINT outputIdx, Rect captureRegion);

	void start();
	void stop();

	DesktopCaptureController(DesktopCaptureController const&) = delete;
	DesktopCaptureController(DesktopCaptureController &&) = delete;
	DesktopCaptureController operator=(DesktopCaptureController const&) = delete;
};


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
	DesktopCaptureController(const UINT& initialOutputIdx, std::function<void(const RgbColor&)> callback);
	~DesktopCaptureController();

	void setOutput(const UINT& outputIdx, Rect captureRegion);

	void start();
	void stop();

	DesktopCaptureController(DesktopCaptureController const&) = delete;
	DesktopCaptureController(DesktopCaptureController &&) = delete;
	DesktopCaptureController operator=(DesktopCaptureController const&) = delete;
};


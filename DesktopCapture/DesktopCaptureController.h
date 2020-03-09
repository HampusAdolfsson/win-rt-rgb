#pragma once

#include "DesktopColorSampler.h"
#include "Color.h"
#include <vector>

/**
*	Manages/multiplexes DesktopColorSamplers for different outputs, and restarts them when errors occur
*	(e.g. when entering fullscreen applications)
*/
class DesktopCaptureController
{
	UINT activeOutput;
	std::vector<DesktopColorSampler*> samplers;

	const UINT getNumberOfOutputs() const;

public:
	DesktopCaptureController(const UINT& initialOutputIdx);
	~DesktopCaptureController();

	void setOutput(const UINT& outputIdx);

	RgbColor getColor();

	DesktopCaptureController(DesktopCaptureController const&) = delete;
	DesktopCaptureController(DesktopCaptureController &&) = delete;
	DesktopCaptureController operator=(DesktopCaptureController const&) = delete;
};


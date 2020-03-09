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

	UINT getNumberOfOutputs();

public:
	DesktopCaptureController(UINT initialOutputIdx);
	~DesktopCaptureController();

	void setOutput(UINT outputIdx);

	Color getColor();

	DesktopCaptureController(DesktopCaptureController const&) = delete;
	DesktopCaptureController(DesktopCaptureController &&) = delete;
	DesktopCaptureController operator=(DesktopCaptureController const&) = delete;
};


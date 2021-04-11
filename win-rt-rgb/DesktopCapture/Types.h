#pragma once
#include "Color.h"
#include <functional>
#include <vector>

namespace DesktopCapture
{
	typedef struct
	{
		unsigned int left, top, width, height;
	} Rect;

	/**
	 *	Callback for when colors have been sampled from a desktop frame.
	*	The parameter is an vector with the resulting colors (the size of the vector matches the number
	*	of groups in the specification that was used to sample).
	*/
	typedef std::function<void(const RgbColor*)> DesktopSamplingCallback;
}
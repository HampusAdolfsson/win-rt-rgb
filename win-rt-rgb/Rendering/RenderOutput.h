#pragma once
#include "RenderTarget.h"

namespace Rendering
{
	/**
	*	An output sink for color values (i.e. RenderTargets) that shows them somewhere, e.g. a WLED instance with an LED strip, or an RGB keyboard with programmatic per-key backlight
	*/
	class RenderOutput
	{
	public:
		virtual void draw(const RenderTarget& target) = 0;

		virtual ~RenderOutput() = 0;
	};
}
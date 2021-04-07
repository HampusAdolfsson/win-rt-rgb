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
		RenderOutput(unsigned int colorTemperature, float gamma);

		/**
		*	Outputs the colors in the RenderTarget to the device.
		*	The color temperature and gamma are applied to the render target before drawing.
		*/
		void draw(RenderTarget& target);

		virtual void drawImpl(const RenderTarget& target) = 0;

		virtual ~RenderOutput() = 0;

	private:
		RgbColor whitePoint;
		float gamma;
	};
}
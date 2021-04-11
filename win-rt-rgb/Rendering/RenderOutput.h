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
		RenderOutput(size_t ledCount, unsigned int colorTemperature, float gamma);

		size_t getLedCount() const;

		/**
		 * Called to initialize the data. Here e.g. system apis may be called.
		 */
		virtual void initialize() = 0;

		/**
		*	Outputs the colors in the RenderTarget to the device.
		*	The color temperature and gamma are applied to the render target before drawing.
		*/
		void draw(RenderTarget& target);

		virtual void drawImpl(const RenderTarget& target) = 0;

		virtual ~RenderOutput() = 0;

	private:
		size_t ledCount;
		RgbColor whitePoint;
		float gamma;
	};
}
#pragma once
#include "Color.h"
#include "MaskingBehaviour.h"
#include <memory>
#ifdef USE_SSE
#include <immintrin.h>
#endif

namespace Rendering
{
	#ifdef USE_SSE
	typedef std::unique_ptr<RgbColor[], decltype(&_aligned_free)> ColorBuffer;
	#else
	typedef std::unique_ptr<RgbColor[]> ColorBuffer;
	#endif
	/**
	*	Stores color data, and allows clients to write color values.
	*	The color data can be sent to/drawn onto a device using a RenderOutput.
	*/
	class RenderTarget
	{
	public:
		RenderTarget(const unsigned int& size, std::unique_ptr<MaskingBehaviour> maskBehaviour);

		void drawRange(const unsigned int& startIndex, const unsigned int& length, const RgbColor* toDraw);

		void beginFrame();

		void applyAdjustments(unsigned int startIndex, unsigned int length, float hue, float saturation, float value);

		void setIntensity(const float& intensity);

		void cloneFrom(const RenderTarget& other);

		inline const ColorBuffer& getColors() const
		{
			return colors;
		}
		inline const int& getSize() const {
			return size;
		}

	private:
		unsigned int size;
		ColorBuffer colors;
		std::unique_ptr<MaskingBehaviour> maskBehaviour;
	};
}
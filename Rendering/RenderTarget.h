#pragma once
#include "Color.h"
#include <vector>

/**
*	Stores color data, and allows clients to write color values.
*	The color data can be sent to/drawn onto a device using a RenderOutput.
*/
class RenderTarget
{
public:
	RenderTarget(unsigned int& size);

	void drawRange(unsigned int& startIndex, unsigned int& length, const RgbColor* toDraw);

	void beginFrame();

	inline const std::vector<RgbColor>& getColors() const
	{
		return colors;
	}

private:
	unsigned int size;
	std::vector<RgbColor> colors;
};


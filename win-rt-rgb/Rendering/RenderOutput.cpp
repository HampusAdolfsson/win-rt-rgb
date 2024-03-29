#include "RenderOutput.h"
#include "Logger.h"
#include <cmath>

using namespace Rendering;

RenderOutput::RenderOutput(size_t ledCount, unsigned int colorTemperature, float gamma)
	: ledCount(ledCount),
	whitePoint(getWhitePoint(colorTemperature)),
	gamma(gamma)
{
}

size_t RenderOutput::getLedCount() const
{
	return ledCount;
}

void RenderOutput::draw(RenderTarget& target)
{
	const auto& colors = target.getColors();

	#pragma omp parallel for
	for (int i = 0; i < target.getSize(); i++)
	{
		colors[i].red = std::pow(colors[i].red * whitePoint.red, gamma);
		colors[i].green = std::pow(colors[i].green * whitePoint.green, gamma);
		colors[i].blue = std::pow(colors[i].blue * whitePoint.blue, gamma);
	}
	drawImpl(target);
}

RenderOutput::~RenderOutput()
{
}

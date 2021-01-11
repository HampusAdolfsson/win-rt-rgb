#pragma once
#include "DesktopDuplicator.h"
#include "D3DMeanColorCalculator.h"
#include "Color.h"
#include "Types.h"
#include <thread>
#include <functional>
#include <vector>

namespace DesktopCapture
{
	/**
	*	Continuously samples the screen and returns an array of colors representing its content.
	*	For details on how the colors are sampled, see SamplingSpecification.
	*	The regions will be chosen as evenly sized
	*	sections divided over the horizontal space.
	*/
	class DesktopColorSampler
	{
		DesktopDuplicator desktopDuplicator;
		D3DMeanColorCalculator frameSampler;
		Rect captureRegion;

		std::thread samplerThread;
		bool isRunning;
		std::vector<DesktopSamplingCallback> callbacks;

		void sampleLoop();

	public:
		/**
		*	Creates a new sampler.
		*	@param outputIdx The index of the output (monitor) to sample
		*	@param specification Specifications for *how* to sample colors from each frame.
		*		For each frame, an array of colors will be generated for each sampling specification.
		*/
		DesktopColorSampler(UINT outputIdx,
							const std::vector<std::pair<SamplingSpecification, DesktopSamplingCallback>>& specifications);

		void setCaptureRegion(Rect captureRegion);

		void start();
		void stop();
	};
}
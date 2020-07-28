#pragma once
#include "DesktopDuplicator.h"
#include "D3DMeanColorCalculator.h"
#include "Color.h"
#include "Rect.h"
#include <thread>
#include <functional>
#include <vector>

/**
 *	Callback for when sampled colors have been generated. The first parameter is the index of
 *	the specification used to generate the colors (as passed into the constructor of DesktopColorSampler).
 *	The second parameter is an array with the colors itself (the size of the array matches the number
 *	of groups in the specification).
 */
typedef std::function<void(const unsigned int&, RgbColor*)> DesktopSamplingCallback;

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
	DesktopSamplingCallback callback;

	void sampleLoop();

public:
	/**
	*	Creates a new sampler.
	*	@param outputIdx The index of the output (monitor) to sample
	*	@param specification Specifications for *how* to sample colors from each frame.
			For each frame, an array of colors will be generated for each sampling specification.
	*	@param callback Called when samples for a specification are ready for a captured frame.
	*/
	DesktopColorSampler(const UINT& outputIdx,
						const std::vector<SamplingSpecification>& specifications,
						DesktopSamplingCallback callback);

	void setCaptureRegion(Rect captureRegion);

	void start();
	void stop();
};


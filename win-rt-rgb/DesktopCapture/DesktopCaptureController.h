#pragma once

#include "Types.h"
#include "Color.h"
#include "DesktopDuplicator.h"
#include "D3DMeanColorCalculator.h"
#include <vector>
#include <mutex>
#include <thread>

namespace DesktopCapture
{
	/**
	*	Continuosly captures and samples colors from one or several monitors, calling back as
	*	colors are generated. The class takes several output specifications, which contain parameters
	*	for how to sample (e.g. the number of regions to sample from). Each output specification can be
	*	assigned to a monitor, and every frame captured from that monitor will be sampled to generate
	*	colors for the output specifications assigned to it.
	*/
	class DesktopCaptureController
	{
		// Per-monitor data and classes
		std::vector<DesktopDuplicator> duplicators;
		std::vector<D3DMeanColorCalculator> colorSamplers;
		std::vector<Rect> captureRegions;
		std::vector<std::thread> samplingWorkers;
		std::vector<bool> workerRunning;
		std::vector<std::mutex> handlesLocks;
		std::vector<std::vector<D3DMeanColorSpecificationHandle*>> assignedHandles;
		std::vector<std::vector<DesktopSamplingCallback>> assignedCallbacks;

		std::vector<std::pair<D3DMeanColorSpecificationHandle, DesktopSamplingCallback>> outputSpecifications;
		// Keeps track of where each output specification is assigned (to which monitor and in what position)
		std::vector<std::pair<size_t, size_t>> outputAssignments;

		bool isActive;

		const UINT getNumberOfMonitors() const;

		void startWorker(UINT monitorIdx);
		void stopWorker(UINT monitorIdx);

		void samplingLoop(UINT monitorIdx);

	public:
		/**
		*	Creates a new capture controller
		*	@param outputSpecifications For each frame, an array of colors will be generated for each sampling specification,
		*		and the colors will provided to the associated callback function.
		*/
		DesktopCaptureController(const std::vector<std::pair<SamplingSpecification, DesktopSamplingCallback>>& outputSpecifications);
		~DesktopCaptureController();

		/**
		*	Sets the region to capture from for the given monitor
		*	@param monitorIdx The index of the monitor to set the region for
		*	@param captureRegion The region to capture from this monitor
		*/
		void setCaptureRegionForMonitor(UINT monitorIdx, Rect captureRegion);

		/**
		*	Sets the monitor to capture from for a specific output specification
		*	@param outputIdx The index of the output specification to set monitor for, as passed into the constructor.
		*	@param monitorIdx The index of the monitor to use for this specification
		*/
		void setCaptureMonitorForOutput(UINT outputIdx, UINT monitorIdx);

		void start();
		void stop();

		DesktopCaptureController(DesktopCaptureController const&) = delete;
		DesktopCaptureController(DesktopCaptureController &&) = delete;
		DesktopCaptureController operator=(DesktopCaptureController const&) = delete;
	};
}

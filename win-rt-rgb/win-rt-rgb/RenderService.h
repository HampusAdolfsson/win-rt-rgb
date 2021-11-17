#pragma once
#include "DesktopCaptureController.h"
#include "Types.h"
#include "AudioMonitor.h"
#include "RenderOutput.h"
#include "Profiles/ProfileManager.h"
#include <vector>
#include <memory>
#include <optional>
#include <chrono>
#include <map>

namespace WinRtRgb
{
	/**
	*	A device to render colors to. When passed to a RenderService, it will continuously render
	*	colors to the given RenderOutput using the other parameters.
	*/
	struct RenderDeviceConfig {
		std::unique_ptr<Rendering::RenderOutput> output;
		float audioAmount;
		float saturationAdjustment;
		float valueAdjustment;
	};

	struct RenderDevice;

	/**
	*   Renders colors captured from the desktop and (optionally) captured audio to a
	*   set of render outputs.
	*/
	class RenderService
	{
	public:
		/**
		 *	Creates a new renderer
		 *	@param defaultCaptureRegion The monitor region to capture when no profiles are active
		 */
		RenderService(DesktopCapture::Rect defaultCaptureRegion);

		/**
		 *	Add a render output to render colors to.
		 *	@param renderOutput Where to render colors to
		 *	@param desktopCaptureParams Specification for how to sample the colors, e.g. how many to generate.
		 		The number of colors must not exceed what the render output can handle.
		 *	@param useAudio Whether to use computer audio to set the brightness of the rendered colors
		 *	@param preferredMonitor	The monitor this device should prefer sampling from, when multiple monitors have active profiles.
		 */
		void setRenderOutputs(std::vector<RenderDeviceConfig> devices);

		void start();
		void stop();

		/**
		 *	Sets the active profile for a monitor.
		 *	@param profileData Specifices which monitor this concerns and what profile (if any) was activated
		 */
		void setActiveProfile(ProfileManager::ActiveProfileData profileData);

	private:
		bool started = false;

		std::vector<RenderDevice> devices;
		std::unique_ptr<DesktopCapture::DesktopCaptureController> desktopCaptureController = nullptr;
		std::unique_ptr<AudioCapture::AudioMonitor> audioMonitor = nullptr;

		std::map<unsigned int, ApplicationProfile> activeProfiles;
		DesktopCapture::Rect defaultCaptureRegion;

		// measuring fps
		unsigned int frames = 0;
		std::chrono::time_point<std::chrono::system_clock> lastFpsTime;

		void audioCallback(float intensity);
		void desktopCallback(unsigned int deviceIdx, const RgbColor* colors);
	};

	struct RenderDevice {
		std::unique_ptr<Rendering::RenderOutput> renderOutput;
		Rendering::RenderTarget desktopRenderTarget;
		std::optional<Rendering::RenderTarget> audioRenderTarget;
		float saturationAdjustment;
		float valueAdjustment;
		float audioAmount;
	};
}
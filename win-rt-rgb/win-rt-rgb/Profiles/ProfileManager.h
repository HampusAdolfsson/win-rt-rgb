#pragma once
#include "ApplicationProfile.h"
#include <Windows.h>
#include <functional>
#include <optional>
#include <utility>

namespace ProfileManager
{
	/**
	*	Starts listening to foreground window changes and notifies when a profile change occurs
	*	@param profileChangedCallback Called when a new profile is activated, with the new profile and the monitor the window is on. If no profile matched the active window, std::nullopt is sent instead.
	*	@param profiles The profiles to watch
	*/
	void start(std::function<void(std::optional<std::pair<ApplicationProfile, unsigned int>>)> profileChangedCallback, const std::vector<ApplicationProfile>& profiles);

	void stop();

	/**
	*	Sets the profiles to use
	*/
	void setProfiles(const std::vector<ApplicationProfile>& profiles);
}

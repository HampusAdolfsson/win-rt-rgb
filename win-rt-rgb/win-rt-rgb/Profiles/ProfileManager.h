#pragma once
#include "ApplicationProfile.h"
#include <Windows.h>
#include <functional>
#include <optional>
#include <utility>

namespace ProfileManager
{
	/**
	*	Starts listening to foreground window changes and notifies when focus is given to a new window.
	*	@param profileChangedCallback Called when a new window is focused, with a profile matching the window and the monitor the focused window is on.
	*		If no profile matched the active window, std::nullopt is sent instead.
	*	@param profiles The profiles to watch
	*/
	void start(std::function<void(std::optional<std::pair<ApplicationProfile, unsigned int>>)> profileChangedCallback, const std::vector<ApplicationProfile>& profiles);

	void stop();

	/**
	*	Sets the profiles to use
	*/
	void setProfiles(const std::vector<ApplicationProfile>& profiles);

	/**
	*	Locks a profile, sending it as the current profile and ignoring any further window focus changes.
	*	The index of the profile to lock.
	*/
	void lockProfile(const unsigned int& index);

	/**
	*	Undos any previous call to lockProfile, to once again listen to window focus changes and send profile updates.
	*/
	void unlock();
}

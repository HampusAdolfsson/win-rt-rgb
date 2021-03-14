#pragma once
#include "ApplicationProfile.h"
#include <Windows.h>
#include <functional>
#include <optional>
#include <utility>

namespace WinRtRgb
{
	namespace ProfileManager
	{
		typedef struct {
			unsigned int monitorIndex;
			std::optional<ApplicationProfile> profile;
			unsigned int profileIndex;
		} ActiveProfileData;

		/**
		*	Starts listening to foreground window changes and notifies with a matching profile when focus is given to a new window.
		*	@param profiles The profiles to use
		*/
		void start(const std::vector<ApplicationProfile>& profiles);

		void stop();

		/**
		*	Registers a function to be called when the active profile changes.
		*	@param profileChangedCallback Called when a new window is focused, with a profile matching the window and the monitor the focused window is on.
		*		If no profile matched the active window, std::nullopt is sent instead.
		*/
		void addCallback(std::function<void(ActiveProfileData)> profileChangedCallback);

		/**
		*	Sets the profiles to use
		*/
		void setProfiles(const std::vector<ApplicationProfile>& profiles);

		/**
		*	Locks a profile, sending it as the current profile and ignoring any further window focus changes.
		*	@param profileIndex The index of the profile to lock.
		*	@param monitorIndex The index of the monitor to use this profile on.
		*/
		void lockProfile(const unsigned int& profileIndex, const unsigned int& monitorIndex);

		/**
		*	Undos any previous call to lockProfile, to once again listen to window focus changes and send profile updates.
		*/
		void unlock();
	}
}
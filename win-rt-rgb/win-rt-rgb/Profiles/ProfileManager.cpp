#include "ProfileManager.h"
#include "Logger.h"
#include <Windows.h>
#include <functional>
#include <map>

std::vector<std::function<void(std::optional<ProfileManager::ActiveProfileData>)>> callbacks;
std::vector<ApplicationProfile> appProfiles;
// Stores the last active profile for each monitor
std::map<unsigned int, ProfileManager::ActiveProfileData> activeProfiles;
bool isLocked;
HWINEVENTHOOK eventHook;

void eventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD idEventThread, DWORD dwmsEventTime);

void ProfileManager::start(const std::vector<ApplicationProfile> &profiles)
{
	appProfiles = profiles;
	isLocked = false;
	eventHook = SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, nullptr, &eventProc, 0, 0, WINEVENT_OUTOFCONTEXT);
}

void ProfileManager::stop()
{
	UnhookWinEvent(eventHook);
}

void ProfileManager::addCallback(std::function<void(std::optional<ActiveProfileData>)> profileChangedCallback)
{
	callbacks.push_back(profileChangedCallback);
}

void ProfileManager::setProfiles(const std::vector<ApplicationProfile> &profiles)
{
	// TODO: technically, we should use a lock here,
	// since this method may be called by a thread separate from the one that has the event hook
	appProfiles = profiles;
}

void ProfileManager::lockProfile(const unsigned int &profileIndex, const unsigned int &monitorIndex)
{
	if (profileIndex >= appProfiles.size())
		return;
	LOGINFO("Locking profile %s on output %d.", appProfiles[profileIndex].regexSpecifier.c_str(), monitorIndex);
	isLocked = true;
	for (auto &callback : callbacks)
	{
		callback(std::optional<ProfileManager::ActiveProfileData>({ appProfiles[profileIndex], profileIndex, monitorIndex }));
	}
}

void ProfileManager::unlock()
{
	isLocked = false;
	for (auto &callback : callbacks)
	{
		callback(std::nullopt);
	}
}

void eventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD idEventThread, DWORD dwmsEventTime)
{
	if (event == EVENT_SYSTEM_FOREGROUND && hwnd)
	{
		if (isLocked) return;
		char title[255];
		GetWindowTextA(hwnd, title, 255);
		if (strnlen(title, 255) == 0) return; // Ignores focusing e.g. the task bar
		WINDOWINFO winInfo;
		winInfo.cbSize = sizeof(winInfo);
		GetWindowInfo(hwnd, &winInfo);
		unsigned int outputIdx = (winInfo.rcWindow.left + winInfo.cxWindowBorders) / 1920; // assumes 1080p monitors placed side by side
		if (outputIdx > 1) outputIdx = 0;

		for (unsigned int i = 0; i < appProfiles.size(); i++)
		{
			if (std::regex_search(title, appProfiles[i].windowTitle))
			{
				LOGINFO("Activating profile %s on output %d.", appProfiles[i].regexSpecifier.c_str(), outputIdx);
				auto prof = ProfileManager::ActiveProfileData{ appProfiles[i], i, outputIdx };
				activeProfiles.insert(std::make_pair(outputIdx, prof));
				for (auto &callback : callbacks)
				{
					callback(std::optional(prof));
				}
				return;
			}
		}
		activeProfiles.erase(outputIdx);
		std::optional<ProfileManager::ActiveProfileData> nextProfile = std::nullopt;
		if (activeProfiles.size() > 0)
		{
			auto prof = activeProfiles.begin()->second;
			nextProfile = std::optional(prof);
		}
		for (auto &callback : callbacks)
		{
			callback(nextProfile);
		}
	}
}

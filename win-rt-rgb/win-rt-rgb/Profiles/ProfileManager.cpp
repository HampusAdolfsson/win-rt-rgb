#include "ProfileManager.h"
#include "Logger.h"
#include <Windows.h>
#include <functional>
#include <map>

using namespace WinRtRgb;

std::vector<std::function<void(ProfileManager::ActiveProfileData)>> callbacks;
std::vector<ApplicationProfile> appProfiles;
// Stores the last active profile for each monitor
std::map<unsigned int, ProfileManager::ActiveProfileData> activeProfiles;
HWINEVENTHOOK eventHook;

void eventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD idEventThread, DWORD dwmsEventTime);

void ProfileManager::start(const std::vector<ApplicationProfile> &profiles)
{
	appProfiles = profiles;
	eventHook = SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, nullptr, &eventProc, 0, 0, WINEVENT_OUTOFCONTEXT);
}

void ProfileManager::stop()
{
	UnhookWinEvent(eventHook);
}

void ProfileManager::addCallback(std::function<void(ActiveProfileData)> profileChangedCallback)
{
	callbacks.push_back(profileChangedCallback);
}

void ProfileManager::setProfiles(const std::vector<ApplicationProfile> &profiles)
{
	// TODO: technically, we should use a lock here,
	// since this method may be called by a thread separate from the one that has the event hook
	appProfiles = profiles;
}

void eventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD idEventThread, DWORD dwmsEventTime)
{
	if (event == EVENT_SYSTEM_FOREGROUND && hwnd)
	{
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
				auto prof = ProfileManager::ActiveProfileData{ outputIdx, appProfiles[i], i };
				activeProfiles.insert(std::make_pair(outputIdx, prof));
				for (auto &callback : callbacks)
				{
					callback(prof);
				}
				return;
			}
		}
		activeProfiles.erase(outputIdx);
		for (auto &callback : callbacks)
		{
			callback(ProfileManager::ActiveProfileData{outputIdx, std::nullopt, 0});
		}
	}
}

#include "ProfileManager.h"
#include "Logger.h"
#include "Config.h"
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
		LOGINFO("Window: %s", title);
		if (strnlen(title, 255) == 0 ||
			strncmp("Task Switching", title, 255) == 0 ||
			strncmp("Search", title, 255) == 0) return; // Ignores focusing e.g. the task bar
		WINDOWINFO winInfo;
		winInfo.cbSize = sizeof(winInfo);
		GetWindowInfo(hwnd, &winInfo);

		unsigned int nMonitors = sizeof(Config::monitors) / sizeof(*Config::monitors);
		unsigned int outputIdx = nMonitors;
		for (int i = 0; i < nMonitors; i++) {
			auto& monitor = Config::monitors[i];
			if (winInfo.rcClient.left >= monitor.left &&
				winInfo.rcClient.left < monitor.left + static_cast<int>(monitor.width) &&
				winInfo.rcClient.top >= monitor.top &&
				winInfo.rcClient.top < monitor.top + static_cast<int>(monitor.height)) {
				outputIdx = i;
				break;
			}
		}
		if (outputIdx == nMonitors) {
			// A hack to detect some fullscreen windows
			if (winInfo.rcClient.left == -32000 && winInfo.rcClient.right == -32000) {
				outputIdx = 0;
			} else {
				LOGSEVERE("Couldn't match monitor to window '%s' at (%ld, %ld)", title, winInfo.rcClient.left, winInfo.rcClient.top);
				return;
			}
		}
		LOGINFO("Matched window at (%ld, %ld) to monitor %u", winInfo.rcClient.left, winInfo.rcClient.top, outputIdx);

		for (unsigned int i = 0; i < appProfiles.size(); i++)
		{
			if (std::regex_search(title, appProfiles[i].windowTitle))
			{
				LOGINFO("Activating profile %s on output %d.", appProfiles[i].regexSpecifier.c_str(), outputIdx);
				auto prof = ProfileManager::ActiveProfileData{ outputIdx, appProfiles[i] };
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
			callback(ProfileManager::ActiveProfileData{outputIdx, std::nullopt});
		}
	}
}
#include "ProfileManager.h"
#include "Logger.h"
#include <Windows.h>
#include <functional>

std::function<void(std::optional<std::pair<ApplicationProfile, unsigned int>>)> callback;
std::vector<ApplicationProfile> appProfiles;
HWINEVENTHOOK eventHook;

void eventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD idEventThread, DWORD dwmsEventTime);

void ProfileManager::start(std::function<void(std::optional<std::pair<ApplicationProfile, unsigned int>>)> profileChangedCallback, const std::vector<ApplicationProfile>& profiles)
{
    callback = profileChangedCallback;
    appProfiles = profiles;
	eventHook = SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, nullptr, &eventProc, 0, 0, WINEVENT_OUTOFCONTEXT);
}

void ProfileManager::stop()
{
	UnhookWinEvent(eventHook);
}

void ProfileManager::setProfiles(const std::vector<ApplicationProfile>& profiles)
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
        WINDOWINFO winInfo;
        winInfo.cbSize = sizeof(winInfo);
        GetWindowInfo(hwnd, &winInfo);
        unsigned int outputIdx = (winInfo.rcWindow.left + winInfo.cxWindowBorders) / 1920; // assumes 1080p monitors placed side by side
        if (outputIdx > 1) outputIdx = 0;

        for (int i = 0; i < appProfiles.size(); i++)
        {
            if (std::regex_search(title, appProfiles[i].windowTitle))
            {
                LOGINFO("Activating profile %d on output %d.", i, outputIdx);
                callback(std::make_pair(appProfiles[i], outputIdx));
                return;
            }
        }
        callback(std::nullopt);
    }
}

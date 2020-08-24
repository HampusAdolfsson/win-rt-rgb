#include "ProfileManager.h"
#include "Logger.h"
#include <Windows.h>
#include <functional>

std::function<void(std::optional<std::pair<ApplicationProfile, unsigned int>>)> callback;
std::vector<ApplicationProfile> appProfiles;
bool isLocked;
HWINEVENTHOOK eventHook;

void eventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD idEventThread, DWORD dwmsEventTime);

void ProfileManager::start(std::function<void(std::optional<std::pair<ApplicationProfile, unsigned int>>)> profileChangedCallback, const std::vector<ApplicationProfile>& profiles)
{
    callback = profileChangedCallback;
    appProfiles = profiles;
    isLocked = false;
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

void ProfileManager::lockProfile(const unsigned int& profileIndex, const unsigned int& monitorIndex)
{
    if (profileIndex >= appProfiles.size()) return;
    LOGINFO("Locking profile %s on output %d.", appProfiles[profileIndex].regexSpecifier.c_str(), monitorIndex);
    isLocked = true;
    callback(std::make_pair(appProfiles[profileIndex], monitorIndex));
}

void ProfileManager::unlock()
{
    isLocked = false;
    callback(std::nullopt);
}

void eventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD idEventThread, DWORD dwmsEventTime)
{
    if (event == EVENT_SYSTEM_FOREGROUND && hwnd)
    {
        if (isLocked) return;
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
                LOGINFO("Activating profile %s on output %d.", appProfiles[i].regexSpecifier.c_str(), outputIdx);
                callback(std::make_pair(appProfiles[i], outputIdx));
                return;
            }
        }
        callback(std::nullopt);
    }
}

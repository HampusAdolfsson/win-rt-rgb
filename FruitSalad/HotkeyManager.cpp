#include "HotkeyManager.h"
#include "Logger.h"
#include <Windows.h>

HotkeyManager::HotkeyManager() : nextId(1) {}

HotkeyManager::~HotkeyManager()
{
	for (const auto& kv : handlers)
	{
		UnregisterHotKey(NULL, kv.first);
	}
}

void HotkeyManager::addHotkey(unsigned int key, HotkeyHandler handler)
{
	handlers[nextId] = handler;
	RegisterHotKey(NULL, nextId, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, key);
	nextId += 1;
}

void HotkeyManager::runHandlerLoop()
{
	MSG msg;
	while (GetMessage(&msg, 0, WM_HOTKEY, 0) == 1)
	{
		switch (msg.message)
		{
			case WM_HOTKEY:
				if (handlers.count(msg.wParam))
				{
					bool quit = handlers[msg.wParam]();
					if (quit)
					{
						goto Exit;
					}
				}
				break;
		}
	}
Exit:
	LOGINFO("Exiting hotkey loop");
}

#pragma once
#include <map>
#include <functional>

typedef std::function<bool()> HotkeyHandler;

/**
* Manages a set of windows API hotkeys. Leaves no guarantees about hotkeys registered multiple times.
*/
class HotkeyManager
{
	std::map<unsigned int, HotkeyHandler> handlers;
	unsigned int nextId;

public:
	HotkeyManager();
	~HotkeyManager();

	void addHotkey(unsigned int key, HotkeyHandler handler);

	void runHandlerLoop();
};


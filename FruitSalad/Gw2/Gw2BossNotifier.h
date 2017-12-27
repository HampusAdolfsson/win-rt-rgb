#pragma once
#include <Windows.h>
#include <thread>
#include <string>
#include "RequestClient.h"

/**
*	Listens for when raid bosses are killed in Guild Wars 2, and plays an appropriate light effect
*/
// This class uses a gw2 addon called arcdps (a dps meter)
// Essentially this class just listens for new log files from ardcps
class Gw2BossNotifier
{
	HANDLE			hDir;
	RequestClient&	reqClient;
	std::thread		directoryListener;

	void listenToChanges();
	void onNewFileDetected(const std::string& fname);
public:
	Gw2BossNotifier(RequestClient& reqCl);
	~Gw2BossNotifier();
};
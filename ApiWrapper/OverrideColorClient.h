#pragma once

#include <string>
#include <WinSock2.h>
#include "Color.h"

/**
*	A client capable of sending temporary overriding colors to a fruitypi server.
*/
class OverrideColorClient
{
	SOCKET				sockHandle;
	struct sockaddr_in	sockAddr;
public:
	/**
	*	Initialize a new client to communicate with the server at the specified address and port
	*/
	OverrideColorClient(const std::string &serverAddr, const int &serverPort);
	OverrideColorClient(const OverrideColorClient&) = delete;
	OverrideColorClient& operator=(const OverrideColorClient&) = delete;
	~OverrideColorClient();

	/**
	*	Send an overriding color to the server
	*/
	void sendColor(const Color &color) const;
};

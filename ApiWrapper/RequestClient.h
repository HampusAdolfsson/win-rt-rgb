#pragma once
#include <string>
#include <WS2tcpip.h>
#include "LightEffect.h"

typedef enum {
	OFF = 0, ON = 1, TOGGLE = 2
} PowerState;

/**
*	A (TCP) client capable of sending various requests to a fruitypi server.
*/
class RequestClient
{
	std::string serverAddr;
	std::string serverPort;

	struct addrinfo *addrInfo;
	SOCKET sockHandle;

	bool initConnection();
	bool ensureIsConnected();

public:
	/**
	*	Creates a new client which connects to the specified address and port
	*/
	RequestClient(const std::string& addr, const std::string& port);
	~RequestClient();
	/**
	*	Sends a request for the server to play the given light effect.
	*/
	unsigned char sendLightEffect(const LightEffect& effect, bool fallback);
	/**
	*	Sends a power on/off request to the server.
	*/
	unsigned char sendOnOffRequest(const PowerState& state);
};

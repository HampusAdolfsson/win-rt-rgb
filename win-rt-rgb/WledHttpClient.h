#pragma once
#include <string>
#include <Windows.h>
#include <WinInet.h>

enum class WledPowerStatus
{
	Off = 0, On = 1, Toggle = 2
};

/**
 *	A client to the WLED http api, allowing e.g. toggling the power or setting the brightness.
 */
class WledHttpClient
{
public:
    WledHttpClient(const std::string& address, const unsigned int& port);
    ~WledHttpClient();

    void setPowerStatus(WledPowerStatus status);

private:
	std::string address;
	unsigned int port;
};


#include "WledHttpClient.h"


static HINTERNET hInternetRoot = nullptr;

WledHttpClient::WledHttpClient(const std::string &address, const unsigned int &port)
	: address(address),
	port(port)
{
	if (!hInternetRoot)
	{
		hInternetRoot = InternetOpen("win-rt-rgb", INTERNET_OPEN_TYPE_DIRECT, nullptr, nullptr, 0);
	}
}

WledHttpClient::~WledHttpClient()
{
}

void WledHttpClient::setPowerStatus(WledPowerStatus status)
{
	const auto apiTarget = std::string("/win&T=") + std::to_string((int)status);
	HINTERNET hInternetSession = InternetConnect(hInternetRoot, address.c_str(), port, nullptr, nullptr, INTERNET_SERVICE_HTTP, 0, 0);
	HINTERNET hRequest = HttpOpenRequest(hInternetSession, "GET", apiTarget.c_str(), nullptr, nullptr, nullptr, INTERNET_FLAG_NO_AUTH, 0);
	HttpSendRequest(hRequest, nullptr, 0, nullptr, 0);
}
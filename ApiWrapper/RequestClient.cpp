#include "Statics.h"
#include "RequestClient.h"

#define MAX_TRIES 3

#define MAGIC_BYTE_1 0x46
#define MAGIC_BYTE_2 0x50
#define PROTOCOL_VERSION 3

#define REQUEST_ID_ON_OFF 1
#define REQUEST_ID_LIGHTEFFECT 1

RequestClient::RequestClient(const std::string& addr, const std::string& port)
	: serverAddr(addr),
	serverPort(port),
	addrInfo(nullptr),
	sockHandle(INVALID_SOCKET) {}

RequestClient::~RequestClient()
{
	closesocket(sockHandle);
	freeaddrinfo(addrInfo);
}

bool RequestClient::initConnection()
{
	if (!addrInfo)
	{
		struct addrinfo hints;
		memset(&hints, 0, sizeof(hints));
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;

		int res = getaddrinfo(serverAddr.c_str(), serverPort.c_str(), &hints, &addrInfo);
		if (res != 0)
		{
			if (addrInfo)
			{
				freeaddrinfo(addrInfo);
				addrInfo = nullptr;
			}
			return false;
		}
	}
	sockHandle = socket(addrInfo->ai_family, addrInfo->ai_socktype, addrInfo->ai_protocol);
	if (sockHandle == INVALID_SOCKET)
	{
		return false;
	}

	int res = connect(sockHandle, addrInfo->ai_addr, addrInfo->ai_addrlen);
	if (res == SOCKET_ERROR)
	{
		closesocket(sockHandle);
		sockHandle = INVALID_SOCKET;
		return false;
	}
	
	return true;
}

bool RequestClient::ensureIsConnected()
{
	if (sockHandle == INVALID_SOCKET)
	{
		bool connected = false;
		int tries = 0;
		while (!connected && tries <= MAX_TRIES)
		{
			connected = initConnection();
			tries++;
		}
		if (!connected) return false;
	}
	return true;
}

unsigned char RequestClient::sendLightEffect(const LightEffect& effect, bool fallback)
{
	if (!ensureIsConnected()) return CONNECTION_ERROR;

	std::vector<unsigned char> data = effect.toByteVector();
	std::vector<unsigned char> header = { MAGIC_BYTE_1, MAGIC_BYTE_2, PROTOCOL_VERSION, REQUEST_ID_LIGHTEFFECT };
	data.insert(data.begin(), header.begin(), header.end());
	data.insert(data.begin() + header.size(), fallback ? 1 : 0);
	send(sockHandle, (char*) data.data(), data.size(), 0);

	char response;
	int read = recv(sockHandle, &response, 1, 0);
	if (read == 0)
	{
		closesocket(sockHandle);
		sockHandle = INVALID_SOCKET;
	}
	return response;
}

unsigned char RequestClient::sendOnOffRequest(const PowerState& state)
{
	return true;
}
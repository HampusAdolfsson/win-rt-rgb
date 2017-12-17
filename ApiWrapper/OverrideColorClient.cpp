#pragma once
 
#include <WS2tcpip.h>
#include "OverrideColorClient.h"

#pragma comment(lib, "Ws2_32.lib")

OverrideColorClient::OverrideColorClient(const std::string &serverAddr, const int &serverPort)
{
	sockHandle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (sockHandle == SOCKET_ERROR)
	{
		fprintf(stderr, "Error opening socket, code: %d", WSAGetLastError());
	}
	memset((char *)&sockAddr, 0, sizeof(sockAddr));
	sockAddr.sin_family = AF_INET;
	sockAddr.sin_port = htons(serverPort);
	InetPton(sockAddr.sin_family, serverAddr.c_str(), &sockAddr.sin_addr);
}

OverrideColorClient::~OverrideColorClient()
{
	closesocket(sockHandle);
}

void OverrideColorClient::sendColor(const Color &color)const
{
	uint8_t bytes[3] = { color.red, color.green, color.blue };
	sendto(sockHandle, (char *) bytes, 3, 0, (struct sockaddr *) &sockAddr, sizeof(sockAddr));
}

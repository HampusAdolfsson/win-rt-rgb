#include "WledRenderOutput.h"
#include <cassert>
#include <WS2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")

WledRenderOutput::WledRenderOutput(const unsigned int& size, const std::string& address, const unsigned int& port)
: outputBuffer(2 + 3 * size, 0)
{
	sockHandle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (sockHandle == SOCKET_ERROR)
	{
		fprintf(stderr, "Error opening socket, code: %d", WSAGetLastError());
	}
	memset((char *)&sockAddr, 0, sizeof(sockAddr));
	sockAddr.sin_family = AF_INET;
	sockAddr.sin_port = htons(port);
	InetPtonA(sockAddr.sin_family, address.c_str(), &sockAddr.sin_addr);

}

WledRenderOutput::~WledRenderOutput()
{
	closesocket(sockHandle);
}

void WledRenderOutput::draw(const RenderTarget& target)
{
	outputBuffer[0] = 2; // DRGB protocol
	outputBuffer[1] = 1;
	const auto& colors = target.getColors();
	for (int i = 0; i < colors.size(); i++)
	{
		const auto& color = colors[i];
		outputBuffer[2 + 3*i] = static_cast<uint8_t>(color.blue);
		outputBuffer[2 + 3*i + 1] = static_cast<uint8_t>(color.green);
		outputBuffer[2 + 3*i + 2] = static_cast<uint8_t>(color.red);
	}
	sendto(sockHandle, (char *) outputBuffer.data(), outputBuffer.size(), 0, (struct sockaddr *) &sockAddr, sizeof(sockAddr));
}

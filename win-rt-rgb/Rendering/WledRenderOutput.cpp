#include "WledRenderOutput.h"
#include <cassert>
#include <WS2tcpip.h>

using namespace Rendering;

#pragma comment(lib, "Ws2_32.lib")

WledRenderOutput::WledRenderOutput(size_t ledCount, const std::string& address, const unsigned int& port,
									unsigned int colorTemp, float gamma)
: RenderOutput(ledCount, colorTemp, gamma),
  address(address),
  port(port),
  sockHandle(INVALID_SOCKET),
  outputBuffer(2 + 3 * ledCount, 0)
{
}

void WledRenderOutput::initialize()
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
	if (sockHandle != INVALID_SOCKET)
	{
		closesocket(sockHandle);
		sockHandle = INVALID_SOCKET;
	}
}

void WledRenderOutput::drawImpl(const RenderTarget& target)
{
	assert(3*target.getSize()+2 <= outputBuffer.size());
	outputBuffer[0] = 2; // DRGB protocol
	outputBuffer[1] = 2;
	const auto& colors = target.getColors();
	for (int i = 0; i < target.getSize(); i++)
	{
		const auto& color = colors[i];
		outputBuffer[2 + 3*i] = static_cast<uint8_t>(color.red * 0xff);
		outputBuffer[2 + 3*i + 1] = static_cast<uint8_t>(color.green * 0xff);
		outputBuffer[2 + 3*i + 2] = static_cast<uint8_t>(color.blue * 0xff);
	}
	sendto(sockHandle, (char *) outputBuffer.data(), outputBuffer.size(), 0, (struct sockaddr *) &sockAddr, sizeof(sockAddr));
}

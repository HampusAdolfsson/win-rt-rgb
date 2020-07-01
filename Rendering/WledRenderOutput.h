#pragma once
#include "RenderOutput.h"
#include <WinSock2.h>
#include <vector>
#include <string>

class WledRenderOutput : public RenderOutput
{
public:
	WledRenderOutput(const unsigned int& size, const std::string& address, const unsigned int& port);
	~WledRenderOutput() override;

	void draw(const RenderTarget& target);

private:
	SOCKET				sockHandle;
	struct sockaddr_in	sockAddr;

	std::vector<uint8_t> outputBuffer;
};


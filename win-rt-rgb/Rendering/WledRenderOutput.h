#pragma once
#include "RenderOutput.h"
#include <WinSock2.h>
#include <vector>
#include <string>

namespace Rendering
{
	class WledRenderOutput : public RenderOutput
	{
	public:
		WledRenderOutput(size_t ledCount, const std::string& address, const unsigned int& port, unsigned int colorTemp, float gamma);
		~WledRenderOutput() override;

		void initialize() override;

		void drawImpl(const RenderTarget& target) override;

		WledRenderOutput(WledRenderOutput const&) = delete;
		WledRenderOutput(WledRenderOutput&&) = delete;
		WledRenderOutput& operator=(WledRenderOutput const&) = delete;

	private:
		std::string address;
		unsigned int port;
		SOCKET				sockHandle;
		struct sockaddr_in	sockAddr;

		std::vector<uint8_t> outputBuffer;
	};
}
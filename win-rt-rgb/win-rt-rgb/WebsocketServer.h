#pragma once
#include "Profiles/ApplicationProfile.h"
#include "RenderOutput.h"
#include "RenderService.h"
#include <string>
#include <vector>
#include <optional>
#define ASIO_STANDALONE
#define _WEBSOCKETPP_CPP11_TYPE_TRAITS_
#include "websocketpp/config/asio_no_tls.hpp"
#include "websocketpp/server.hpp"
#include "json.h"

namespace WinRtRgb
{
	typedef websocketpp::server<websocketpp::config::asio> server;

	/**
	*	Receives json requests over a websocket to manage settings (i.e. set outputs or profiles),
	*	and sends state changes to clients (i.e. when the active profile changes).
	*	This server is only meant to have a single client at once.
	*/
	class WebsocketServer
	{
	public:
		WebsocketServer(std::function<void(std::vector<ApplicationProfile>)> profilesCallback,
						std::function<void(std::vector<RenderDeviceConfig>)> devicesCallback);
		~WebsocketServer();

		void start(const unsigned int& port);

		void notifyActiveProfileChanged(unsigned int monitorIdx, const std::optional<unsigned int>& activeProfileId);

	private:
		server endpoint;
		std::optional<websocketpp::connection_hdl> client;

		std::function<void(std::vector<ApplicationProfile>)> profilesCallback;
		std::function<void(std::vector<RenderDeviceConfig>)> devicesCallback;

		void handleProfileMessage(const nlohmann::json& contents);
		void handleDeviceMessage(const nlohmann::json& contents);
	};
}
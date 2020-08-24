#pragma once
#include "Profiles/ApplicationProfile.h"
#include <string>
#include <vector>
#include <optional>
#define ASIO_STANDALONE
#define _WEBSOCKETPP_CPP11_TYPE_TRAITS_
#include "websocketpp/config/asio_no_tls.hpp"
#include "websocketpp/server.hpp"
#include "json.h"

typedef websocketpp::server<websocketpp::config::asio> server;

/**
*	Receives json requests over a websocket to manage settings (i.e. set outputs or profiles).
*/
class WebsocketServer
{
public:
	WebsocketServer(std::function<void(std::vector<ApplicationProfile>)> profilesCallback,
					std::function<void(std::optional<std::pair<unsigned int, unsigned int>>)> lockCallback);
	~WebsocketServer();

	void start(const unsigned int& port);

private:
	server endpoint;
	std::function<void(std::vector<ApplicationProfile>)> profilesCallback;
	std::function<void(std::optional<std::pair<unsigned int, unsigned int>>)> lockCallback;

	void handleProfileMessage(const nlohmann::json& contents);
	void handleLockMessage(const nlohmann::json& contents);
};

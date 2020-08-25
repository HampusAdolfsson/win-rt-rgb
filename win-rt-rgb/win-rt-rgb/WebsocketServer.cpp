#include "WebsocketServer.h"
#include "Logger.h"
#include <cstdio>


WebsocketServer::WebsocketServer(std::function<void(std::vector<ApplicationProfile>)> profilesCallback,
								std::function<void(std::optional<std::pair<unsigned int, unsigned int>>)> lockCallback)
 : profilesCallback(profilesCallback),
 lockCallback(lockCallback)
//  client(nullptr)
{ }

WebsocketServer::~WebsocketServer()
{
	endpoint.stop_listening();
	endpoint.stop();
}

void WebsocketServer::start(const unsigned int& port)
{
	endpoint.set_open_handler([&](websocketpp::connection_hdl handle) {
		LOGINFO("Connection opened\n");
		client = handle;
	});
	endpoint.set_close_handler([&](websocketpp::connection_hdl handle){
		LOGINFO("Connection closed\n");
		client = std::nullopt;
	});
	endpoint.set_message_handler([&](websocketpp::connection_hdl conn, std::shared_ptr<websocketpp::config::asio::message_type> message){
		nlohmann::json json = nlohmann::json::parse(message->get_payload());
		std::string subject = json["subject"].get<std::string>();
		nlohmann::json contents = json["contents"];
		if (subject == "profiles") {
			LOGINFO("Got profiles message");
			handleProfileMessage(contents);
		} else if (subject == "lock") {
			LOGINFO("Got lock message");
			handleLockMessage(contents);
		} else {
			LOGSEVERE("Received message with unknown subject: %s", subject);
		}
	});
	endpoint.init_asio();
	endpoint.listen(port);
	endpoint.start_accept();
	endpoint.run();
}

void WebsocketServer::notifyActiveProfileChanged(const std::optional<unsigned int>& activeProfileIndex)
{
	if (client.has_value())
	{
		nlohmann::json message;
		message["subject"] = "activeProfile";
		if (activeProfileIndex.has_value())
		{
			message["contents"] = *activeProfileIndex;
		}
		auto connection = endpoint.get_con_from_hdl(*client);
		connection->send(message.dump());
	}
}


void WebsocketServer::handleProfileMessage(const nlohmann::json& contents)
{
	std::vector<ApplicationProfile> receivedProfiles;
	for (const auto& profileJson : contents) {
		auto regex = profileJson["regex"].get<std::string>();
		nlohmann::json areaJson = profileJson["area"];
		unsigned int x = (unsigned int) areaJson["x"].get<int>();
		unsigned int y = (unsigned int) areaJson["y"].get<int>();
		unsigned int width = (unsigned int) areaJson["width"].get<int>();
		unsigned int height = (unsigned int) areaJson["height"].get<int>();
		receivedProfiles.push_back(ApplicationProfile(regex, {x, y, width, height}));
	}
	profilesCallback(receivedProfiles);
}

void WebsocketServer::handleLockMessage(const nlohmann::json& contents)
{
	if (contents.find("profile") != contents.end())
	{
		unsigned int profile = (unsigned int) contents["profile"].get<int>();
		unsigned int monitor = (unsigned int) contents["monitor"].get<int>();
		lockCallback(std::make_pair(profile, monitor));
	}
	else
	{
		lockCallback(std::nullopt);
	}
}
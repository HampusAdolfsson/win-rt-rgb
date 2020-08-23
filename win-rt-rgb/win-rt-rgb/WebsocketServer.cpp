#include "WebsocketServer.h"
#include "Logger.h"
#include <cstdio>


WebsocketServer::WebsocketServer(std::vector<ApplicationProfile> initialProfiles,
								std::function<void(std::vector<ApplicationProfile>)> profilesCallback)
 : profiles(initialProfiles),
 profilesCallback(profilesCallback)
{ }

WebsocketServer::~WebsocketServer()
{
	endpoint.stop_listening();
	endpoint.stop();
}

void WebsocketServer::start(const unsigned int& port)
{
	endpoint.set_open_handler([&](websocketpp::connection_hdl conn) {
		LOGINFO("Connection opened\n");
		auto handle = endpoint.get_con_from_hdl(conn);
		std::string message = makeProfileMessage();
		handle->send(message);
	});
	endpoint.set_close_handler([](websocketpp::connection_hdl conn){
		LOGINFO("Connection closed\n");
	});
	endpoint.set_message_handler([&](websocketpp::connection_hdl conn, std::shared_ptr<websocketpp::config::asio::message_type> message){
		LOGINFO("Received: %s\n", message->get_payload().c_str());
		nlohmann::json json = nlohmann::json::parse(message->get_payload());
		std::string subject = json["subject"].get<std::string>();
		nlohmann::json contents = json["contents"];
		if (subject == "profiles") {
				LOGINFO("Got profiles message");
				handleProfileMessage(contents);
		} else {
			LOGSEVERE("Received message with unknown subject: %s", subject);
		}
	});
	endpoint.init_asio();
	endpoint.listen(port);
	endpoint.start_accept();
	endpoint.run();
}

std::string WebsocketServer::makeProfileMessage()
{
	std::vector<nlohmann::json> profilesJson;
	for (const auto& profile : profiles)
	{
		nlohmann::json json;
		json["regex"] = profile.regexSpecifier;
		nlohmann::json rect;
		rect["x"] = profile.captureRegion.left;
		rect["y"] = profile.captureRegion.top;
		rect["width"] = profile.captureRegion.width;
		rect["height"] = profile.captureRegion.height;
		json["area"] = rect;
		profilesJson.push_back(json);
	}
	nlohmann::json message;
	message["subject"] = "profiles";
	message["contents"] = profilesJson;
	return message.dump();
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
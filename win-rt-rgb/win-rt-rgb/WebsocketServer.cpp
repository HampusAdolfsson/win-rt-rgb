#include "WebsocketServer.h"
#include "Logger.h"
#include "WledRenderOutput.h"
#include "Config.h"
#include <cstdio>

using namespace WinRtRgb;

WebsocketServer::WebsocketServer(std::function<void(std::vector<ApplicationProfile>)> profilesCallback,
								std::function<void(std::vector<RenderDeviceConfig>)> devicesCallback)
 : profilesCallback(profilesCallback),
 devicesCallback(devicesCallback)
{ }

WebsocketServer::~WebsocketServer()
{
	endpoint.stop_listening();
	endpoint.stop();
}

void WebsocketServer::start(const unsigned int& port)
{
	endpoint.set_error_channels(websocketpp::log::elevel::all);
	endpoint.set_access_channels(websocketpp::log::alevel::none);

	endpoint.set_open_handler([&](websocketpp::connection_hdl handle) {
		LOGINFO("Connection opened");
		client = handle;
	});
	endpoint.set_close_handler([&](websocketpp::connection_hdl handle){
		LOGINFO("Connection closed");
		client = std::nullopt;
	});
	endpoint.set_message_handler([&](websocketpp::connection_hdl conn, std::shared_ptr<websocketpp::config::asio::message_type> message){
		nlohmann::json json = nlohmann::json::parse(message->get_payload());
		LOGINFO("%s", message->get_payload().c_str());
		std::string subject = json["subject"].get<std::string>();
		nlohmann::json contents = json["contents"];
		if (subject == "profiles") {
			LOGINFO("Got profiles message");
			handleProfileMessage(contents);

		} else if (subject == "devices") {
			LOGINFO("Got devices message");
			handleDeviceMessage(contents);
		} else {
			LOGSEVERE("Received message with unknown subject: %s", subject.c_str());
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
		receivedProfiles.push_back(ApplicationProfile(regex, DesktopCapture::Rect{x, y, width, height}));
	}
	profilesCallback(receivedProfiles);
}

void WebsocketServer::handleDeviceMessage(const nlohmann::json& contents)
{
	std::vector<RenderDeviceConfig> receivedDevices;
	for (const auto& json : contents) {
		bool enabled = json["enabled"].get<bool>();
		if (!enabled) continue;

		const nlohmann::json& deviceJson = json["device"];
		int nLeds = deviceJson["numberOfLeds"].get<int>();
		float colorTemp = deviceJson["colorTemp"].get<int>();
		float gamma = deviceJson["gamma"].get<float>();
		float saturationAdjustment = deviceJson["saturationAdjustment"].get<int>() / 100.0f;
		float valueAdjustment = deviceJson["valueAdjustment"].get<int>() / 100.0f;
		bool useAudio = deviceJson["useAudio"].get<boolean>();
		int preferredMonitor = deviceJson["preferredMonitor"].get<int>();
		int type = deviceJson["type"].get<int>();

		RenderDeviceConfig deviceConfig;
		deviceConfig.saturationAdjustment = saturationAdjustment;
		deviceConfig.valueAdjustment = valueAdjustment;
		deviceConfig.useAudio = useAudio;
		deviceConfig.preferredMonitor = preferredMonitor;

		switch (type)
		{
		case 0:
		{
			std::string ipAddress = deviceJson["wledData"]["ipAddress"].get<std::string>();
			deviceConfig.output = std::make_unique<Rendering::WledRenderOutput>(nLeds == 50 ? 89 : nLeds, ipAddress, WLED_UDP_PORT, colorTemp, gamma);
			break;
		}
		default:
			LOGWARNING("Got unknown device type: %d", type);
			continue;
		}
		receivedDevices.push_back(std::move(deviceConfig));
	}
	devicesCallback(std::move(receivedDevices));
}

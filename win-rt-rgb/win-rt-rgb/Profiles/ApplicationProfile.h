#pragma once
#include "Types.h"
#include <vector>
#include <regex>
#include <variant>
#include <optional>
#include <utility>

namespace WinRtRgb
{
	typedef std::variant<float, unsigned int> MonitorDistance;
	struct AreaSpecification
	{
		std::optional<std::pair<unsigned int, unsigned int>> resolution;
		MonitorDistance x, y, width, height;
	};

	/**
	*	An application profile specifices a screen region to use to capture a specific
	*	application or window. The regex specifies the title of windows to use the profile for,
	*	and the rect gives the area to capture for those windows.
	*/
	struct ApplicationProfile
	{
		ApplicationProfile(unsigned int id, std::string regexSpecifier, std::vector<AreaSpecification> areas, int priority);
		unsigned int id;
		std::regex windowTitle;
		std::string regexSpecifier;
		std::vector<AreaSpecification> areas;
		int priority;
	};

	inline unsigned int resolveMonitorDistance(const MonitorDistance& distance, unsigned int fullLength)
	{
		if (std::holds_alternative<unsigned int>(distance))
		{
			return std::get<unsigned int>(distance);
		}
		else
		{
			return std::round(std::get<float>(distance) * fullLength);
		}
	}
}
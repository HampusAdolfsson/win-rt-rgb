#pragma once
#include <cinttypes>
#include <vector>
#include "Color.h"

typedef enum {
	Breathing = 0, Fading = 1, Flashing = 2, Static = 3
} EffectType;

/**
*	A lighteffect which can be sent to a fruitypi server
*/
class LightEffect
{
	uint64_t	duration;
	EffectType	type;
	std::vector<RgbColor> colors;
public:
	LightEffect(const uint64_t& duration, const EffectType& type, const std::vector<RgbColor>& colors);

	/**
	*	Serializes the effect to an array of bytes that is consistent with the request format used by a server
	*/
	std::vector<unsigned char> toByteVector() const;
};

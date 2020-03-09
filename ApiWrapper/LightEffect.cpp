#include <intrin.h>
#include "LightEffect.h"

LightEffect::LightEffect(const uint64_t& duration, const EffectType& type, const std::vector<RgbColor>& colors)
	: duration(duration),
	type(type),
	colors(colors) { }


std::vector<unsigned char> LightEffect::toByteVector() const
{
	std::vector<unsigned char> serialized;
	serialized.reserve(sizeof(unsigned char) + sizeof(duration) + sizeof(unsigned char) + colors.size() * 3);
	serialized.push_back(type);
	uint64_t beDuration = _byteswap_uint64(duration);
	for (int i = 0; i < sizeof(beDuration); i++)
	{
		serialized.push_back(beDuration & 0xff);
		beDuration >>= 8;
	}
	serialized.push_back(colors.size());
	for (std::vector<RgbColor>::const_iterator it = colors.begin(); it != colors.end(); it++)
	{
		serialized.push_back((*it).red);
		serialized.push_back((*it).green);
		serialized.push_back((*it).blue);
	}
	return serialized;
}
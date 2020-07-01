#include "Color.h"
#include <cassert>
#include <algorithm>

RgbColor operator*(const RgbColor& c, const float& factor)
{
	assert(factor >= 0 && factor <= 1);
	return {
		static_cast<uint8_t>(c.red * factor),
		static_cast<uint8_t>(c.green * factor),
		static_cast<uint8_t>(c.blue * factor)
	};
}

bool operator==(const RgbColor& c1, const RgbColor& c2)
{
    return c1.red == c2.red && c1.green == c2.green && c1.blue == c2.blue;
}
bool operator!=(const RgbColor& c1, const RgbColor& c2)
{
    return c1.red != c2.red || c1.green != c2.green || c1.blue != c2.blue;
}

RgbColor blendColors(RgbColor c1, RgbColor c2, const float& progress)
{
	assert(progress >= 0 && progress <= 1);
	c1 = c1 * (1 - progress);
	c2 = c2 * progress;
	return {
		static_cast<uint8_t>(c1.red + c2.red),
		static_cast<uint8_t>(c1.green + c2.green),
		static_cast<uint8_t>(c1.blue + c2.blue)
	};
}

RgbColor setSaturation(const RgbColor& color, uint8_t saturation)
{
    HsvColor hsv = rgbToHsv(color);
    hsv.saturation = saturation;
    return hsvToRgb(hsv);
}

RgbColor hsvToRgb(HsvColor hsv)
{
    RgbColor rgb;
    unsigned char region, remainder, p, q, t;

    if (hsv.saturation == 0)
    {
        rgb.red = hsv.value;
        rgb.green = hsv.value;
        rgb.blue = hsv.value;
        return rgb;
    }

    region = hsv.hue / 43;
    remainder = (hsv.hue - (region * 43)) * 6;

    p = (hsv.value * (255 - hsv.saturation)) >> 8;
    q = (hsv.value * (255 - ((hsv.saturation * remainder) >> 8))) >> 8;
    t = (hsv.value * (255 - ((hsv.saturation * (255 - remainder)) >> 8))) >> 8;

    switch (region)
    {
    case 0:
        rgb.red = hsv.value; rgb.green = t; rgb.blue = p;
        break;
    case 1:
        rgb.red = q; rgb.green = hsv.value; rgb.blue = p;
        break;
    case 2:
        rgb.red = p; rgb.green = hsv.value; rgb.blue = t;
        break;
    case 3:
        rgb.red = p; rgb.green = q; rgb.blue = hsv.value;
        break;
    case 4:
        rgb.red = t; rgb.green = p; rgb.blue = hsv.value;
        break;
    default:
        rgb.red = hsv.value; rgb.green = p; rgb.blue = q;
        break;
    }

    return rgb;
}

HsvColor rgbToHsv(RgbColor rgb)
{
    HsvColor hsv;
    unsigned char rgbMin, rgbMax;

    rgbMin = rgb.red < rgb.green ? (rgb.red < rgb.blue ? rgb.red : rgb.blue) : (rgb.green < rgb.blue ? rgb.green : rgb.blue);
    rgbMax = rgb.red > rgb.green ? (rgb.red > rgb.blue ? rgb.red : rgb.blue) : (rgb.green > rgb.blue ? rgb.green : rgb.blue);

    hsv.value = rgbMax;
    if (hsv.value == 0)
    {
        hsv.hue = 0;
        hsv.saturation = 0;
        return hsv;
    }

    hsv.saturation = 255 * long(rgbMax - rgbMin) / hsv.value;
    if (hsv.saturation == 0)
    {
        hsv.hue = 0;
        return hsv;
    }

    if (rgbMax == rgb.red)
        hsv.hue = 0 + 43 * (rgb.green - rgb.blue) / (rgbMax - rgbMin);
    else if (rgbMax == rgb.green)
        hsv.hue = 85 + 43 * (rgb.blue - rgb.red) / (rgbMax - rgbMin);
    else
        hsv.hue = 171 + 43 * (rgb.red - rgb.green) / (rgbMax - rgbMin);

    return hsv;
}

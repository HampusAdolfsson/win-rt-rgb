#include "Color.h"
#include <cassert>
#include <algorithm>

RgbColor operator*(const RgbColor& c, const float& factor)
{
	assert(factor >= 0 && factor <= 1);
	return {
		static_cast<ColorChannel>(c.blue * factor),
		static_cast<ColorChannel>(c.green * factor),
		static_cast<ColorChannel>(c.red * factor)
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
		static_cast<ColorChannel>(c1.red + c2.red),
		static_cast<ColorChannel>(c1.green + c2.green),
		static_cast<ColorChannel>(c1.blue + c2.blue)
	};
}

HsvColor rgbToHsv(RgbColor rgb)
{
	// Code taken from here: https://stackoverflow.com/a/6930407
	HsvColor    out;
	double      min, max, delta;

	min = rgb.red < rgb.green ? rgb.red : rgb.green;
	min = min < rgb.blue ? min : rgb.blue;

	max = rgb.red > rgb.green ? rgb.red : rgb.green;
	max = max > rgb.blue ? max : rgb.blue;

	out.value = max;								// v
	delta = max - min;
	if (delta < 0.00001)
	{
		out.saturation = 0;
		out.hue = 0; // undefined, maybe nan?
		return out;
	}
	if( max > 0.0 ) // NOTE: if Max is == 0, this divide would cause a crash
	{
		out.saturation = (delta / max);					// s
	}
	else
	{
		// if max is 0, then r = g = b = 0
		// s = 0, h is undefined
		out.saturation = 0.0;
		out.hue = NAN;							// its now undefined
		return out;
	}
	if( rgb.red >= max )							// > is bogus, just keeps compilor happy
	{
		out.hue = ( rgb.green - rgb.blue ) / delta;		// between yellow & magenta
	}
	else if( rgb.green >= max )
	{
		out.hue = 2.0 + ( rgb.blue - rgb.red ) / delta;	// between cyan & yellow
	}
	else
	{
		out.hue = 4.0 + ( rgb.red - rgb.green ) / delta;	// between magenta & cyan
	}

	out.hue *= 60.0;								// degrees

	if( out.hue < 0.0 )
	{
		out.hue += 360.0;
	}

	return out;
}

RgbColor hsvToRgb(HsvColor hsv)
{
	double		hh, p, q, t, ff;
	long		i;
	RgbColor	out;

	if(hsv.saturation <= 0.0) {       // < is bogus, just shuts up warnings
		out.red = hsv.value;
		out.green = hsv.value;
		out.blue = hsv.value;
		return out;
	}
	hh = hsv.hue;
	if(hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = (long)hh;
	ff = hh - i;
	p = hsv.value * (1.0 - hsv.saturation);
	q = hsv.value * (1.0 - (hsv.saturation * ff));
	t = hsv.value * (1.0 - (hsv.saturation * (1.0 - ff)));

	switch(i) {
	case 0:
		out.red = hsv.value;
		out.green = t;
		out.blue = p;
		break;
	case 1:
		out.red = q;
		out.green = hsv.value;
		out.blue = p;
		break;
	case 2:
		out.red = p;
		out.green = hsv.value;
		out.blue = t;
		break;

	case 3:
		out.red = p;
		out.green = q;
		out.blue = hsv.value;
		break;
	case 4:
		out.red = t;
		out.green = p;
		out.blue = hsv.value;
		break;
	case 5:
	default:
		out.red = hsv.value;
		out.green = p;
		out.blue = q;
		break;
	}
	return out;
}
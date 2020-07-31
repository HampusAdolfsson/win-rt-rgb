#include "CudaKernels.h"
#include "Logger.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>
#include <curand.h>
#include <assert.h>

__global__
void blurColorsKernel(RgbColor *id, RgbColor *od, int w, int r)
{
	// TODO: use coalesced memory accesses
	// TODO: try a guassian window
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < r || x >= w - r)
	{
		od[x] = id[x];
	} else {
		float scale = 1.0f / (float)((r << 1) + 1);
		RgbColor sum = {0.f, 0.f, 0.f};
		for (int i = -r; i <= r; i++)
		{
			sum.red += id[x + i].red;
			sum.green += id[x + i].green;
			sum.blue += id[x + i].blue;
		}
		od[x].red = saturate(sum.red * scale);
		od[x].green = saturate(sum.green * scale);
		od[x].blue = saturate(sum.blue * scale);
	}
}

__device__
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

__device__
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

__global__
void calculateMeanColorKernel(uint8_t* screen, int width, int height, size_t pitch, unsigned int* output, int outputWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;
	unsigned int* destination = output + 3 * (x / outputWidth);

	uint32_t* pixel = (uint32_t*)(screen + y * pitch + x * sizeof(uint32_t));
	uint32_t val = *pixel;
	// TODO: make this more efficient
	atomicAdd(destination, val & 0xFF);
	atomicAdd(destination + 1, (val >> 8) & 0xFF);
	atomicAdd(destination + 2, (val >> 16) & 0xFF);
}

__global__
void averageAndAdjustColorsKernel(unsigned int* channels, int pixelsPerChannel, RgbColor* colorOutputs, int nOutputs, float saturation, bool flip)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= nOutputs) return;
	RgbColor rgb;
	rgb.red = float(channels[3*x]) / pixelsPerChannel / 0xFF;
	rgb.green = float(channels[3*x+1]) / pixelsPerChannel / 0xFF;
	rgb.blue = float(channels[3*x+2]) / pixelsPerChannel / 0xFF;
	auto hsv = rgbToHsv(rgb);
	if (hsv.saturation > 0.001f)
	{
		hsv.saturation = min(hsv.saturation + saturation, 1.0f);
	}
	if (flip) x = nOutputs - 1 - x;
	colorOutputs[x] = hsvToRgb(hsv);
}

namespace CudaKernels
{
	void calculateMeanColor(uint8_t* pixels, int width, int height, size_t pitch, unsigned int* outputChannels, int outputWidth)
	{
		dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
		calculateMeanColorKernel<<<Dg, Db>>>(pixels, width, height, pitch, outputChannels, outputWidth);
	}

	void averageAndAdjustColors(unsigned int* channels, int pixelsPerChannel, RgbColor* colorOutputs, int outputSize, float saturationAdjustment, bool flip)
	{
		size_t blocksize = 32;
		size_t gridsize = (outputSize + blocksize - 1) / blocksize;
		averageAndAdjustColorsKernel<<<gridsize, blocksize>>>(channels, pixelsPerChannel, colorOutputs, outputSize, saturationAdjustment, flip);
	}

	void blurColors(RgbColor *id, RgbColor *od, int outputSize, int r)
	{
		size_t blocksize = 32;
		size_t gridsize = (outputSize + blocksize - 1) / blocksize;
		blurColorsKernel<<<blocksize, gridsize>>>(id, od, outputSize, r);
	}
}

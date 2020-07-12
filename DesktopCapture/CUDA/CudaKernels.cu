#include "CudaKernels.h"
#include "Logger.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>
#include <curand.h>
#include <assert.h>

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
void averageAndAdjustColorsKernel(unsigned int* channels, int pixelsPerChannel, RgbColor* colorOutputs, int nOutputs)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x > nOutputs) return;
	colorOutputs[x].red = channels[3*x] / pixelsPerChannel;
	colorOutputs[x].green = channels[3*x+1] / pixelsPerChannel;
	colorOutputs[x].blue = channels[3*x+2] / pixelsPerChannel;
	// TODO: do some color adjustments
}

__device__ unsigned int outputt[4] = { 0, 0, 0, 0 };

namespace CudaKernels
{
	void calculateMeanColor(uint8_t* pixels, int width, int height, size_t pitch, unsigned int* outputChannels, int outputWidth)
	{
		dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
		calculateMeanColorKernel<<<Dg, Db>>>(pixels, width, height, pitch, outputChannels, outputWidth);
	}

	void averageAndAdjustColors(unsigned int* channels, int pixelsPerChannel, RgbColor* colorOutputs, int outputSize)
	{
		size_t blocksize = 128;
		size_t gridsize = (outputSize + blocksize - 1) / blocksize;
		averageAndAdjustColorsKernel<<<gridsize, blocksize>>>(channels, pixelsPerChannel, colorOutputs, outputSize);
	}
}

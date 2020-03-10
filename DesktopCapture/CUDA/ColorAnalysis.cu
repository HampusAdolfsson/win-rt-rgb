#include "ColorAnalysis.h"
#include "Logger.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>
#include <curand.h>

__device__ unsigned int output[4] = { 0, 0, 0, 0 };

__global__
void calculateMeanColor(uint32_t *screen, int width, int height, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	uint32_t val = screen[y * width + x];
	atomicAdd(output, val & 0xFF);
	atomicAdd(output + 1, (val >> 8) & 0xFF);
	atomicAdd(output + 2, (val >> 16) & 0xFF);
}


namespace CudaUtils
{
	RgbColor getMeanColor(cudaGraphicsResource* texture, void* buf, int width, int height, size_t pitch)
	{
		cudaArray* cuArray;
		cudaError_t status = cudaGraphicsSubResourceGetMappedArray(&cuArray, texture, 0, 0);
		status = cudaMemcpy2DFromArray(buf, pitch, cuArray, 0, 0, pitch, height, cudaMemcpyDeviceToDevice);

		unsigned int result[4] = { 0,0,0,0 };
		status = cudaMemcpyToSymbol(output, result, sizeof(result), 0, cudaMemcpyHostToDevice);

		// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
		dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
		calculateMeanColor<<<Dg, Db>>>((uint32_t*) buf, width, height, pitch);

		cudaError_t error = cudaDeviceSynchronize();
		if (error != cudaSuccess)
		{
			LOGSEVERE("cuda mean color failed to launch wth error %d\n", error);
			return { 0,0,0 };
		}
		status = cudaMemcpyFromSymbol(result, output, sizeof(result), 0, cudaMemcpyDeviceToHost);
		int pixelCount = width * height;
		RgbColor c = { static_cast<uint8_t>(result[2] / pixelCount), static_cast<uint8_t>(result[1] / pixelCount), static_cast<uint8_t>(result[0] / pixelCount) };
		return c;
	}
}

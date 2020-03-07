#include "ColorAnalysis.h"
#include "Logger.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>
#include <curand.h>

__device__ int output[4] = { 0, 0, 0, 0 };

__global__
void calculateMeanColor(cudaTextureObject_t screen, int width, int height, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float u = x / (float)width;
	float v = y / (float)height;
	if (x < width && y < height) {
		int val = tex2D<int>(screen, u, v);
		//int val = surface[y * pitch + x];
		int channel = x % 4;
		printf("%f,%f,%.8X\n", u, v, val);
		//surface[y * pitch + x] = 0xff00ff;
		atomicAdd(output + channel, val);
	}
}
__global__
void calculateMeanColor2(int *screen, int width, int height, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float u = x / (float)width;
	float v = y / (float)height;
	if (x < width && y < height) {
		int val = screen[y * pitch + x];
		int channel = x % 4;
		printf("%f,%f,%.8X\n", u, v, val);
		screen[y * pitch + x] = 0xff00ff;
		atomicAdd(output + channel, val);
	}
}


namespace CudaUtils
{
	//Color getMeanColor(cudaGraphicsResource* texture, void* buf, int width, int height, size_t pitch)
	//{
	//	cudaArray* cuArray;
	//	cudaError_t status = cudaGraphicsSubResourceGetMappedArray(&cuArray, texture, 0, 0);
	//	status = cudaMemcpy2DFromArray(buf, pitch, cuArray, 0, 0, pitch, height, cudaMemcpyDeviceToDevice);

	//	struct cudaResourceDesc resDesc;
	//	memset(&resDesc, 0, sizeof(resDesc));
	//	resDesc.res.array.array = cuArray;
	//	resDesc.resType = cudaResourceTypeArray;

	//	struct cudaTextureDesc texDesc;
	//	memset(&texDesc, 0, sizeof(texDesc));
	//	texDesc.addressMode[0] = cudaAddressModeClamp;
	//	texDesc.addressMode[1] = cudaAddressModeClamp;
	//	texDesc.filterMode = cudaFilterModePoint;
	//	texDesc.readMode = cudaReadModeElementType;
	//	texDesc.normalizedCoords = 1;

	//	cudaTextureObject_t texObj = 0;
	//	status = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

	//	// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
	//	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
	//	calculateMeanColor2<<<Dg, Db>>>((int*)buf, width, height, pitch);

	//	cudaError_t error = cudaDeviceSynchronize();
	//	if (error != cudaSuccess)
	//	{
	//		LOGSEVERE("cuda mean color failed to launch wth error %d\n", error);
	//		return { 0,0,0 };
	//	}
	//	int result[4];
	//	status = cudaMemcpyFromSymbol(result, output, sizeof(result), 0, cudaMemcpyDeviceToHost);
	//	int pixelCount = width * height;
	//	Color c = { static_cast<uint8_t>(result[2] / pixelCount), static_cast<uint8_t>(result[1] / pixelCount), static_cast<uint8_t>(result[0] / pixelCount) };

	//	// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
	//	//status = cudaMemcpy2DToArray(
	//	//	cuArray, // dst array
	//	//	0, 0,    // offset
	//	//	buf, pitch,       // src
	//	//	width * 4 * sizeof(float), height, // extent
	//	//	cudaMemcpyDeviceToDevice); // kind
	//	cudaDestroyTextureObject(texObj);
	//	return c;
	//}

	Color getMeanColor(cudaGraphicsResource* texture, void* buf, int width, int height, size_t pitch) {
		cudaArray* cuArray;
		cudaError_t status = cudaGraphicsSubResourceGetMappedArray(&cuArray, texture, 0, 0);
		status = cudaMemcpy2DFromArray(buf, pitch, cuArray, 0, 0, pitch, height, cudaMemcpyDeviceToDevice);
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		//cuda Array
		cudaArray* d_cuArr;
		cudaMallocArray(&d_cuArr, &channelDesc, width, height);

		dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);
		calculateMeanColor2<< <Dg, Db >> > ((int*)buf, 128, 128, 128);
		cudaError_t error = cudaDeviceSynchronize();
		if (error != cudaSuccess)
		{
			LOGSEVERE("cuda mean color failed to launch wth error %d\n", error);
			return { 0,0,0 };
		}
		status = cudaMemcpy2DToArray(d_cuArr, 0, 0, buf, pitch, width * sizeof(float), height, cudaMemcpyDeviceToDevice);
			return { 0,0,0 };
	}
}

#include "CudaMeanColorCalculator.h"
#include "CudaKernels.h"
#include "Logger.h"
#include <cuda_d3d11_interop.h>
#include <cassert>

CudaMeanColorCalculator::CudaMeanColorCalculator()
	: nSamplesPerFrame(0), width(0), height(0), intermediaryBuffer(nullptr), textureBuffer(nullptr), cudaResource(nullptr), outputBuffer(nullptr)
{
}

CudaMeanColorCalculator::~CudaMeanColorCalculator()
{
	if (intermediaryBuffer)
	{
		cudaFree(intermediaryBuffer);
	}
	if (textureBuffer)
	{
		cudaFree(textureBuffer);
	}
	if (outputBuffer)
	{
		cudaFree(outputBuffer);
	}
	if (cudaResource)
	{
		cudaGraphicsUnregisterResource(cudaResource);
	}
}

void CudaMeanColorCalculator::initialize(const unsigned int& nSamplesPerFrame,
											ID3D11Texture2D* frameBuffer,
											const int& width,
											const int& height)
{
	this->nSamplesPerFrame = nSamplesPerFrame;
	this->width = width;
	this->height = height;

	cudaError_t status = cudaMalloc(&intermediaryBuffer, nSamplesPerFrame * 3 * sizeof(int));
	if (status != cudaSuccess)
	{
		LOGSEVERE("cudaMalloc failed");
	}
	status = cudaMallocPitch(&textureBuffer, &textureBufferPitch, width*sizeof(uint32_t), height);
	if (status != cudaSuccess)
	{
		LOGSEVERE("cudaMalloc failed");
	}
	status = cudaMalloc(&outputBuffer, sizeof(RgbColor) * nSamplesPerFrame);
	if (status != cudaSuccess)
	{
		LOGSEVERE("cudaMalloc failed");
	}

	assert(textureBufferPitch == width * sizeof(uint32_t)); // parts of the code assumes no padding
	status = cudaGraphicsD3D11RegisterResource(&cudaResource, frameBuffer, cudaGraphicsRegisterFlagsNone);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to register D3D resource with cuda");
	}
}

void CudaMeanColorCalculator::getMeanColors(Rect activeRegion, RgbColor* out)
{
	activeRegion.width = min(activeRegion.width, width - activeRegion.left);
	activeRegion.height = min(activeRegion.height, height - activeRegion.top);

	cudaError_t status = cudaGraphicsMapResources(1, &cudaResource, nullptr);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to map cuda resource");
	}
	cudaArray* cuArray;
	status = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
	status = cudaMemcpy2DFromArray(textureBuffer, textureBufferPitch, cuArray,
									activeRegion.left, activeRegion.top,
									sizeof(uint32_t)*activeRegion.width, activeRegion.height,
										cudaMemcpyDeviceToDevice);
	status = cudaMemset(intermediaryBuffer, 0, 3*sizeof(int)*nSamplesPerFrame);

	CudaKernels::calculateMeanColor((uint8_t*) textureBuffer, activeRegion.width, activeRegion.height,
									textureBufferPitch, (unsigned int*) intermediaryBuffer, activeRegion.width / nSamplesPerFrame);

	CudaKernels::averageAndAdjustColors((unsigned int*)intermediaryBuffer,
										activeRegion.height * (activeRegion.width / nSamplesPerFrame),
										(uint8_t*)outputBuffer, nSamplesPerFrame, sizeof(RgbColor));

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		LOGSEVERE("cuda mean color failed to launch wth error %d\n", status);
		return;
	}
	status = cudaMemcpy(out, outputBuffer, sizeof(RgbColor) * nSamplesPerFrame, cudaMemcpyDeviceToHost);

	status = cudaGraphicsUnmapResources(1, &cudaResource, nullptr);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to unmap cuda resource");
	}
}

#include "CudaMeanColorCalculator.h"
#include "CudaKernels.h"
#include "Logger.h"
#include <cuda_d3d11_interop.h>
#include <cassert>
#include <stdexcept>

CudaMeanColorCalculator::CudaMeanColorCalculator()
	: width(0), height(0), textureBuffer(nullptr), cudaResource(nullptr)
{
}

void CudaMeanColorCalculator::initialize(const std::vector<SamplingSpecification>& samplingParameters,
												ID3D11Texture2D* frameBuffer,
												const int& width, const int& height)
{
	this->width = width;
	this->height = height;
	this->specifications = samplingParameters;

	cudaError_t status = cudaMallocPitch(&textureBuffer, &textureBufferPitch, width*sizeof(uint32_t), height);
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

	bufferSets = std::vector<CudaBuffers>(samplingParameters.size());
	for (size_t i = 0; i < samplingParameters.size(); i++)
	{
		unsigned int nRegions = samplingParameters[i].numberOfRegions;

		cudaError_t status = cudaMalloc(&bufferSets[i].intermediaryBuffer, nRegions * 3 * sizeof(int));
		if (status != cudaSuccess)
		{
			LOGSEVERE("cudaMalloc failed");
		}
		status = cudaMalloc(&bufferSets[i].outputBuffer, sizeof(RgbColor) * nRegions);
		if (status != cudaSuccess)
		{
			LOGSEVERE("cudaMalloc failed");
		}
		status = cudaMalloc(&bufferSets[i].outputBufferBlurred, sizeof(RgbColor) * nRegions);
		if (status != cudaSuccess)
		{
			LOGSEVERE("cudaMalloc failed");
		}

	}
}

CudaMeanColorCalculator::~CudaMeanColorCalculator()
{
	for (auto buffers : bufferSets)
	{
		if (buffers.intermediaryBuffer)
		{
			cudaFree(buffers.intermediaryBuffer);
		}
		if (buffers.outputBuffer)
		{
			cudaFree(buffers.outputBuffer);
		}
		if (buffers.outputBufferBlurred)
		{
			cudaFree(buffers.outputBufferBlurred);
		}
	}
	if (textureBuffer)
	{
		cudaFree(textureBuffer);
	}
	if (cudaResource)
	{
		cudaGraphicsUnregisterResource(cudaResource);
	}
}

void CudaMeanColorCalculator::getMeanColors(Rect activeRegion, const std::vector<RgbColor*>& out)
{
	activeRegion.width = min(activeRegion.width, width - activeRegion.left);
	activeRegion.height = min(activeRegion.height, height - activeRegion.top);

	// ready resources
	cudaError_t status = cudaGraphicsMapResources(1, &cudaResource, nullptr);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to map cuda resource");
	}
	cudaArray* cuArray;
	status = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
	status = cudaMemcpy2DFromArray(textureBuffer, textureBufferPitch, cuArray,
									sizeof(uint32_t)*activeRegion.left, activeRegion.top,
									sizeof(uint32_t)*activeRegion.width, activeRegion.height,
										cudaMemcpyDeviceToDevice);
	for (size_t i = 0; i < specifications.size(); i++)
	{
		status = cudaMemset(bufferSets[i].intermediaryBuffer, 0, 3*sizeof(int)*specifications[i].numberOfRegions);

		// kernel calls
		CudaKernels::calculateMeanColor((uint8_t*) textureBuffer, activeRegion.width, activeRegion.height,
										textureBufferPitch, (unsigned int*) bufferSets[i].intermediaryBuffer,
										activeRegion.width / specifications[i].numberOfRegions);

		CudaKernels::averageAndAdjustColors((unsigned int*)bufferSets[i].intermediaryBuffer,
											activeRegion.height * (activeRegion.width / specifications[i].numberOfRegions),
											(RgbColor*)bufferSets[i].outputBuffer, specifications[i].numberOfRegions,
											specifications[i].saturationAdjustment, specifications[i].flipHorizontally);
		if (specifications[i].blurRadius > 0) {
			CudaKernels::blurColors((RgbColor*)bufferSets[i].outputBuffer, (RgbColor*)bufferSets[i].outputBufferBlurred,
									specifications[i].numberOfRegions, specifications[i].blurRadius);
		}
	}
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		LOGSEVERE("cuda mean color failed to launch with error %d\n", status);
	}
	for (size_t i = 0; i < specifications.size(); i++)
		// fetch results and release resources
		status = cudaMemcpy(out[i], specifications[i].blurRadius > 0 ? bufferSets[i].outputBufferBlurred : bufferSets[i].outputBuffer,
							sizeof(RgbColor) * specifications[i].numberOfRegions, cudaMemcpyDeviceToHost);
	}

	status = cudaGraphicsUnmapResources(1, &cudaResource, nullptr);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to unmap cuda resource");
	}
}

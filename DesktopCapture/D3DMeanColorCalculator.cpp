#include "D3DMeanColorCalculator.h"
#include "CUDA/ColorAnalysis.h"
#include "Logger.h"
#include <assert.h>
#include <cuda_d3d11_interop.h>

void D3DMeanColorCalculator::initialize(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight)
{
	width = textureWidth;
	height = textureHeight;
	device->GetImmediateContext(&deviceContext);

	// allocate our buffer
	D3D11_TEXTURE2D_DESC texDesc;
	RtlZeroMemory(&texDesc, sizeof(texDesc));
	texDesc.Width = textureWidth;
	texDesc.Height = textureHeight;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	texDesc.SampleDesc.Count = 1;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	HRESULT hr = device->CreateTexture2D(&texDesc, NULL, &frameBuffer);
	if (hr != S_OK)
	{
		LOGSEVERE("Failed to create texture");
		return;
	}

	cudaError_t status = cudaMallocPitch(&cudaBuffer, &cudaBufferPitch, width*sizeof(uint32_t), height);
	if (status != cudaSuccess)
	{
		LOGSEVERE("cudaMalloc failed");
	}
	assert(cudaBufferPitch == width * sizeof(uint32_t)); // parts of the code assumes no padding
	status = cudaGraphicsD3D11RegisterResource(&cudaResource, frameBuffer, cudaGraphicsRegisterFlagsNone);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to register D3D resource with cuda");
	}
}

void D3DMeanColorCalculator::setFrameData(ID3D11Texture2D *frame)
{
	deviceContext->CopyResource(frameBuffer, frame);
}

RgbColor D3DMeanColorCalculator::sample(Rect activeRegion)
{
	cudaError_t status = cudaGraphicsMapResources(1, &cudaResource, nullptr);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to map cuda resource");
	}
	RgbColor result =  CudaUtils::getMeanColor(cudaResource, cudaBuffer, cudaBufferPitch, width, height, activeRegion);
	status = cudaGraphicsUnmapResources(1, &cudaResource, nullptr);
	if (status != cudaSuccess)
	{
		LOGSEVERE("Failed to unmap cuda resource");
	}
	return result;
}

D3DMeanColorCalculator::~D3DMeanColorCalculator()
{
	if (cudaBuffer)
	{
		cudaFree(cudaBuffer);
	}
	if (frameBuffer)
	{
		frameBuffer->Release();
	}
	if (deviceContext)
	{
		deviceContext->Release();
	}
	if (cudaResource)
	{
		cudaGraphicsUnregisterResource(cudaResource);
	}
}

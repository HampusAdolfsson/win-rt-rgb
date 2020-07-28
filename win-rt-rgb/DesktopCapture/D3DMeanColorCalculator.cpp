#include "D3DMeanColorCalculator.h"
#include "Logger.h"
#include <assert.h>
#include <cuda_d3d11_interop.h>

void D3DMeanColorCalculator::initialize(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight,
				const std::vector<SamplingSpecification>& samplingParameters)
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

	cudaCalculator.initialize(samplingParameters, frameBuffer, width, height);

	results = std::vector<RgbColor*>(samplingParameters.size());
	for (size_t i = 0; i < samplingParameters.size(); i++)
	{
		outputBuffers.push_back(std::vector<RgbColor>(samplingParameters[i].numberOfRegions));
		// Create the vector we'll return on every sample call
		results[i] = outputBuffers[i].data();
	}
}

D3DMeanColorCalculator::D3DMeanColorCalculator() {}

void D3DMeanColorCalculator::setFrameData(ID3D11Texture2D *frame)
{
	deviceContext->CopyResource(frameBuffer, frame);
}

std::vector<RgbColor*> D3DMeanColorCalculator::sample(const Rect& activeRegion)
{
	for (size_t i = 0; i < results.size(); i++)
	{
		cudaCalculator.getMeanColors(activeRegion, results);
	}
	return results;
}

D3DMeanColorCalculator::~D3DMeanColorCalculator()
{
	if (frameBuffer)
	{
		frameBuffer->Release();
	}
	if (deviceContext)
	{
		deviceContext->Release();
	}
}

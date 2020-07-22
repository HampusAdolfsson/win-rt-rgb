#include "D3DMeanColorCalculator.h"
#include "Logger.h"
#include <assert.h>
#include <cuda_d3d11_interop.h>

void D3DMeanColorCalculator::initialize(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight, const UINT& nSamplesPerFrame)
{
	width = textureWidth;
	height = textureHeight;
	device->GetImmediateContext(&deviceContext);

	outputBuffer = std::vector<RgbColor>(nSamplesPerFrame);

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

	cudaCalculator.initialize(nSamplesPerFrame, frameBuffer, width, height);
}

D3DMeanColorCalculator::D3DMeanColorCalculator() {}

void D3DMeanColorCalculator::setFrameData(ID3D11Texture2D *frame)
{
	deviceContext->CopyResource(frameBuffer, frame);
}

RgbColor* D3DMeanColorCalculator::sample(const Rect& activeRegion)
{
	cudaCalculator.getMeanColors(activeRegion, outputBuffer.data());
	return outputBuffer.data();
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

#include "D3DMeanColorCalculator.h"
#include "Logger.h"
#include <assert.h>
#include <memory>
#include <immintrin.h>

constexpr unsigned int MIP_LEVEL = 2;
constexpr unsigned int SCALING_FACTOR = 1 << MIP_LEVEL;
constexpr unsigned int RGBA_COLOR_SIZE = 4;

void D3DMeanColorCalculator::initialize(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight,
				const std::vector<SamplingSpecification>& samplingParameters)
{
	width = textureWidth;
	height = textureHeight;
	specifications = samplingParameters;
	device->GetImmediateContext(&deviceContext);

	// allocate our buffers
	D3D11_TEXTURE2D_DESC texDesc;
	RtlZeroMemory(&texDesc, sizeof(texDesc));
	texDesc.Width = textureWidth;
	texDesc.Height = textureHeight;
	texDesc.MipLevels = MIP_LEVEL + 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	texDesc.SampleDesc.Count = 1;
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	texDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
	HRESULT hr = device->CreateTexture2D(&texDesc, NULL, &frameBuffer);
	if (hr != S_OK)
	{
		LOGSEVERE("Failed to create texture");
		return;
	}

	RtlZeroMemory(&texDesc, sizeof(texDesc));
	texDesc.Width = textureWidth / SCALING_FACTOR;
	texDesc.Height = textureHeight / SCALING_FACTOR;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	texDesc.SampleDesc.Count = 1;
	texDesc.Usage = D3D11_USAGE_STAGING;
	texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	hr = device->CreateTexture2D(&texDesc, NULL, &mappingBuffer);
	if (hr != S_OK)
	{
		LOGSEVERE("Failed to create texture");
		return;
	}

	// colorCalculator.initialize(samplingParameters, frameBuffer, width / SCALING_FACTOR, height / SCALING_FACTOR);
	// frameBufferView = device
	D3D11_SHADER_RESOURCE_VIEW_DESC resDesc;
	RtlZeroMemory(&resDesc, sizeof(resDesc));
	resDesc.Format = texDesc.Format;
	resDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	resDesc.Texture2D.MipLevels = -1;
	resDesc.Texture2D.MostDetailedMip = 0;
	hr = device->CreateShaderResourceView(frameBuffer, &resDesc, &frameBufferView);
	if (hr != S_OK)
	{
		LOGSEVERE("Failed to create resource view");
		return;
	}

	size_t cpuBufSize = RGBA_COLOR_SIZE * width / SCALING_FACTOR * height / SCALING_FACTOR;
	cpuBuffer =  (uint32_t*) _aligned_malloc(cpuBufSize + cpuBufSize % sizeof(__m256), sizeof(__m256));

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
	deviceContext->CopySubresourceRegion(frameBuffer, 0, 0, 0, 0, frame, 0, nullptr);
}

std::vector<RgbColor*> D3DMeanColorCalculator::sample(Rect activeRegion)
{
	activeRegion.left /= SCALING_FACTOR;
	activeRegion.width /= SCALING_FACTOR;
	activeRegion.top /= SCALING_FACTOR;
	activeRegion.height /= SCALING_FACTOR;

	copyToCpuBuffer(activeRegion, MIP_LEVEL);

   	memset(outputBuffers[0].data(), 0, sizeof(RgbColor) * outputBuffers[0].size());

	// TODO: make aligned
	size_t nColors = activeRegion.width + (activeRegion.width % (sizeof(__m256)/sizeof(float)));
	std::vector<uint32_t> verticalSums(nColors * 3);
	for (int y = 0; y < activeRegion.height; y++)
	{
		for (int x = 0; x < activeRegion.width; x += sizeof(__m256) / RGBA_COLOR_SIZE)
		{
			__m256i val = _mm256_load_si256((__m256i*) (cpuBuffer + y * width / SCALING_FACTOR + x));
			__m256i redSums = _mm256_loadu_si256((__m256i*) (verticalSums.data() + x));
			__m256i redVals = _mm256_and_si256(val, _mm256_set1_epi32(0xFF));
			redSums = _mm256_add_epi32(redSums, redVals);
			_mm256_storeu_si256((__m256i*) (verticalSums.data() + x), redSums);
			__m256i greenSums = _mm256_loadu_si256((__m256i*) (verticalSums.data() + nColors + x));
			__m256i greenVals = _mm256_and_si256(_mm256_srli_epi32(val, 8), _mm256_set1_epi32(0xFF));
			greenSums = _mm256_add_epi32(greenSums, greenVals);
			_mm256_storeu_si256((__m256i*) (verticalSums.data() + nColors + x), greenSums);
			__m256i blueSums = _mm256_loadu_si256((__m256i*) (verticalSums.data() + 2*nColors + x));
			__m256i blueVals = _mm256_and_si256(_mm256_srli_epi32(val, 16), _mm256_set1_epi32(0xFF));
			blueSums = _mm256_add_epi32(blueSums, blueVals);
			_mm256_storeu_si256((__m256i*) (verticalSums.data() + 2*nColors + x), blueSums);
		}
	}

	for (int o = 0; o < specifications.size(); o++)
	{
		double outputWidth = double(activeRegion.width) / specifications[o].numberOfRegions;
		std::vector<uint32_t> regionSums(3 * specifications[o].numberOfRegions);
		for (int i = 0; i < specifications[o].numberOfRegions; i++)
		{
			unsigned int regionStart = ceil(outputWidth * i);
			unsigned int regionEnd = ceil(outputWidth * (i+1));
			unsigned int pixelsPerRegion = activeRegion.height * (regionEnd - regionStart);
			for (int x = regionStart; x < regionEnd; x++)
			{
				regionSums[3*i] += verticalSums[x];
				regionSums[3*i+1] += verticalSums[nColors + x];
				regionSums[3*i+2] += verticalSums[2*nColors + x];
			}
			int index = specifications[o].flipHorizontally ? specifications[o].numberOfRegions - 1 - i : i;
			results[o][index].red = regionSums[3*i] / (pixelsPerRegion * 255.0);
			results[o][index].green = regionSums[3*i+1] / (pixelsPerRegion * 255.0);
			results[o][index].blue = regionSums[3*i+2] / (pixelsPerRegion * 255.0);
		}
	}

	adjustSaturation();

	return results;
}

void D3DMeanColorCalculator::copyToCpuBuffer(Rect region, unsigned int srcMipLevel)
{
	deviceContext->GenerateMips(frameBufferView);
	D3D11_BOX box;
	box.front = 0;
	box.back = 1;
	box.left = region.left;
	box.right = region.left + region.width;
	box.top = region.top;
	box.bottom = region.top + region.height;
	deviceContext->CopySubresourceRegion(mappingBuffer, 0, 0, 0, 0, frameBuffer, srcMipLevel, &box);

	// copy to cpu
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	HRESULT hr = deviceContext->Map(mappingBuffer, 0, D3D11_MAP_READ, 0, &mappedResource);
	if (hr != S_OK)
	{
		LOGSEVERE("Failed to map resource");
		return;
	}

	uint32_t* src = (uint32_t*)mappedResource.pData;
	memcpy(cpuBuffer, src, RGBA_COLOR_SIZE * region.width * region.height);
	deviceContext->Unmap(mappingBuffer, 0);
}

void D3DMeanColorCalculator::adjustSaturation()
{
	for (int o = 0; o < specifications.size(); o++)
	{
		if (specifications[o].saturationAdjustment == .0f) continue;
		#pragma omp parallel for
		for (int i = 0; i < specifications[o].numberOfRegions; i++)
		{
			HsvColor hsv = rgbToHsv(outputBuffers[o][i]);
			if (hsv.saturation > 0.001f)
			{
				hsv.saturation = min(hsv.saturation + specifications[0].saturationAdjustment, 1.0f);
				outputBuffers[o][i] = hsvToRgb(hsv);
			}
		}
	}
}

D3DMeanColorCalculator::~D3DMeanColorCalculator()
{
	if (frameBuffer)
	{
		frameBuffer->Release();
	}
	if (frameBufferView)
	{
		frameBufferView->Release();
	}
	if (mappingBuffer)
	{
		mappingBuffer->Release();
	}
	if (deviceContext)
	{
		deviceContext->Release();
	}
	if (cpuBuffer)
	{
		_aligned_free(cpuBuffer);
	}
}

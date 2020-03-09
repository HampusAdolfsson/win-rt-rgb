#include "D3DMeanColorCalculator.h"
#include "Logger.h"
#include <gdiplus.h>
#pragma comment(lib, "Gdiplus.lib")

#define RGBA_COLOR_SIZE 4

#define RETURN_ON_ERROR(hr) do {\
						if (hr != S_OK) {\
							LOGSEVERE("D3dColorSampler got error: %d, line %d", hr, __LINE__);\
							return;\
						}\
						} while(0)

void D3DMeanColorCalculator::initialize(ID3D11Device *device, const UINT& textureWidth, const UINT& textureHeight) {
	width = textureWidth;
	height = textureHeight;
	device->GetImmediateContext(&deviceContext);
	buffer = std::make_unique<uint8_t[]>(RGBA_COLOR_SIZE * width * height);
}

Color D3DMeanColorCalculator::sample(ID3D11Texture2D *texture) {
	copyToCpu(texture);
	uint32_t channels[RGBA_COLOR_SIZE] = {0,0,0,0};
	for (int i = 0; i < RGBA_COLOR_SIZE * width * height; i++) {
		channels[i % RGBA_COLOR_SIZE] += buffer[i];
	}

	for (int i = 0; i < RGBA_COLOR_SIZE; i++) {
		channels[i] /= width * height;
	}

	return { static_cast<uint8_t>(channels[2]), static_cast<uint8_t>(channels[1]), static_cast<uint8_t>(channels[0]) };
}

void D3DMeanColorCalculator::copyToCpu(ID3D11Texture2D *texture) {
	HRESULT hr;

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	// this or mapsubresource?
	hr = deviceContext->Map(texture, 0, D3D11_MAP_READ, 0, &mappedResource);
	RETURN_ON_ERROR(hr);

	uint8_t* src = (uint8_t*)mappedResource.pData;
	uint8_t* dst = buffer.get();
	memcpy(dst, src,  RGBA_COLOR_SIZE * width * height);
	deviceContext->Unmap(texture, 0);
}

D3DMeanColorCalculator::~D3DMeanColorCalculator() {
	if (deviceContext) {
		deviceContext->Release();
	}
}

void D3DMeanColorCalculator::saveAsBitmap(std::unique_ptr<uint8_t[]>& data, const UINT& width, const UINT& height) {
	Gdiplus::GdiplusStartupInput m_gdiplusStartupInput;
	ULONG_PTR m_gdiplusToken;
	Gdiplus::GdiplusStartup(&m_gdiplusToken, &m_gdiplusStartupInput, NULL);
	Gdiplus::Bitmap bitmap(width, height, PixelFormat32bppRGB);
	Gdiplus::Rect rect(0, 0, width, height);
	Gdiplus::BitmapData lockedData;
	bitmap.LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat32bppRGB, &lockedData);
	memcpy(lockedData.Scan0, data.get(), RGBA_COLOR_SIZE * width * height);
	bitmap.UnlockBits(&lockedData);
	//Save to PNG
	CLSID pngClsid;
	CLSIDFromString(L"{557CF406-1A04-11D3-9A73-0000F81EF32E}", &pngClsid);
	Gdiplus::Status st = bitmap.Save(L"file.png", &pngClsid, NULL);
}

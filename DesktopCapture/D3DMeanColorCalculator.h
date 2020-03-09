#pragma once
#include "Color.h"
#include <d3d11.h>
#include <memory>
#include <gdiplus.h>

/**
*	Samples a d3d texture and returns its average color
*/
class D3DMeanColorCalculator
{
	ID3D11DeviceContext* deviceContext;
	UINT width, height;
	std::unique_ptr<uint8_t[]> buffer;

	void copyToCpu(ID3D11Texture2D* texture);

	// for debugging purposes
	void saveAsBitmap(std::unique_ptr<uint8_t[]>& data, UINT width, UINT height);

public:
	void initialize(ID3D11Device *device, UINT textureWidth, UINT textureHeight);

	Color sample(ID3D11Texture2D *texture);

	~D3DMeanColorCalculator();
};

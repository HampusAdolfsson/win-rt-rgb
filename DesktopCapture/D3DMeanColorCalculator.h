#pragma once
#include "Color.h"
#include <d3d11.h>
#include <cuda_runtime_api.h>

/**
*	Samples a d3d texture and returns its average color
*/
class D3DMeanColorCalculator
{
	ID3D11DeviceContext* deviceContext = nullptr;
	ID3D11Texture2D *frameBuffer = nullptr;
	UINT width, height;

	cudaGraphicsResource* cudaResource = nullptr;
	void* cudaBuffer = nullptr;
	size_t cudaBufferPitch;

public:
	/**
	 *	Initialize this object.
	 *	@param device The device to expect textures from
	 *	@param textureWidth The width of textures to be sampled
	 *	@param textureHeight The height of textures to be sampled
	 */
	void initialize(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight);

	/**
	 * Set the frame to be used for the next calculation
	 */
	void setFrameData(ID3D11Texture2D *frame);

	/**
	 *	Calculate the mean color of the current frame
	 */
	RgbColor sample();

	~D3DMeanColorCalculator();
};


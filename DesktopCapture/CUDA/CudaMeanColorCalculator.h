#pragma once
#include "../Rect.h"
#include "Color.h"
#include <cuda_runtime_api.h>
#include <d3d11.h>

/**
 * Gets the average color from a number of sections in a texture
 */
class CudaMeanColorCalculator
{
public:
	CudaMeanColorCalculator();
	~CudaMeanColorCalculator();

	/**
	 *	Initialize with output specification and the texture where the caller will store frame data
	 *	before passing it to this class
	 *	@param nSamplesPerFrame The number of sections for each frame, i.e. how many colors to generate each time
	 *	@param frameBuffer The texture where frame data will be stored
	 *	@param width The width of the texture
	 *	@param height The height of the texture
	 */
	void initialize(const unsigned int& nSamplesPerFrame, ID3D11Texture2D* frameBuffer, const int& width, const int& height);

	/**
	 * Gets the specified number of colors from the frameBuffer specificed in the initialize call.
	 * @param activeRegion The part of the texture to actually use/evaluate
	 * @param out An array in which to place the resulting colors
	 */
	void getMeanColors(Rect activeRegion, RgbColor* out);

private:
	unsigned int nSamplesPerFrame;
	int width, height;

	cudaGraphicsResource* cudaResource = nullptr;
	void* textureBuffer = nullptr;
	size_t textureBufferPitch;
	void* intermediaryBuffer;
	void* outputBuffer;
};

#pragma once
#include "cuda/CudaMeanColorCalculator.h"
#include "Color.h"
#include "Rect.h"
#include <d3d11.h>
#include <vector>

/**
*	Samples a d3d texture and returns its average color.
*	Multiple colors can be generated per frame, and will be picked from evenly sized
*	sections divided over the horizontal space.
*/
class D3DMeanColorCalculator
{
	ID3D11DeviceContext* deviceContext = nullptr;
	ID3D11Texture2D *frameBuffer = nullptr;
	UINT width, height;
	std::vector<RgbColor> outputBuffer;

	CudaMeanColorCalculator cudaCalculator;

public:
	/**
	 *	Initialize this object.
	 *	@param device The device to expect textures from
	 *	@param textureWidth The width of textures to be sampled
	 *	@param textureHeight The height of textures to be sampled
	 *	@param nSamplesPerFrame The number of colors to generate for each call to sample.
	 */
	void initialize(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight, const UINT& nSamplesPerFrame);

	/**
	 * Set the frame to be used for the next calculation
	 */
	void setFrameData(ID3D11Texture2D *frame);

	/**
	 *	Calculate the mean color of nSamplesPerFrame sections within (a region of) the current frame.
	 *	Returns an array of the results. The contents of the array may be overwritten and are valid until this method is called again.
	 */
	RgbColor *sample(const Rect& activeRegion);

	D3DMeanColorCalculator();
	~D3DMeanColorCalculator();

	D3DMeanColorCalculator(D3DMeanColorCalculator const&) = delete;
	D3DMeanColorCalculator(D3DMeanColorCalculator &&) = delete;
	D3DMeanColorCalculator operator=(D3DMeanColorCalculator const&) = delete;
};


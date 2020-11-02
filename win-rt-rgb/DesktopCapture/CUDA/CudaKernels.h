#include "Color.h"
#include "../SamplingSpecification.h"
#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#include "../Rect.h"

// Cuda functions for analyzing the colors of D3D textures

namespace CudaKernels
{
	/**
	 * Gets the average color from a number of sections in a texture
	 * @param pixels A buffer containing the pixel values
	 * @param width The width of the texture
	 * @param height The height of the texture
	 * @param pitch The pitch of the pixel buffer
	 * @param outputChannels The array to place the resulting colors channel sums in
	 * @param outputWidth The width of each section
	 */
	void calculateMeanColor(uint8_t* pixels, int width, int height, size_t pitch, unsigned int* outputChannels, int outputWidth, cudaStream_t stream);

	/**
	 * Averages out a set of color channels produced by calculateMeanColors and produces an array of RgbColor:s
	 */
	void averageAndAdjustColors(unsigned int* channels, int pixelsPerChannel, RgbColor* colorOutputs, int outputSize, float saturationAdjustment, bool flip, cudaStream_t stream);

	/**
	 * Performs a 1d box blur on a color array
	 */
	void blurColors(RgbColor* id, RgbColor* od, int outputSize, int r, cudaStream_t stream);
};
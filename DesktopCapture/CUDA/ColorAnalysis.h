#include "Color.h"
#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#include "../Rect.h"

// Cuda functions for analyzing the colors of D3D textures

namespace CudaUtils
{
	/**
	 * Gets the average color from a texture
	 * @param texture A cuda registered texture to read from
	 * @param buf A 2d buffer of the same size as the texture (will be overwritten)
	 * @param pitch The pitch of the buffer
	 * @param width The width of the texture
	 * @param height The height of the texture
	 * @param activeRegion The part of the texture to actually use/evaluate
	 * @returns The average color of the active region within the texture
	 */
	RgbColor getMeanColor(cudaGraphicsResource* texture, void* buf, size_t pitch, int width, int height, Rect activeRegion);
};
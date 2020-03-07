#include "Color.h"
#include <d3d11.h>
#include <cuda_d3d11_interop.h>

// Cuda functions for analyzing the colors of D3D textures

namespace CudaUtils
{
	Color getMeanColor(cudaGraphicsResource* texture, void* buf, int width, int height, size_t pitch);
};
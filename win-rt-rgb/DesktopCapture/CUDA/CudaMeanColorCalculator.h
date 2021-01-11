#pragma once
#include "../Types.h"
#include "Color.h"
#include <cuda_runtime_api.h>
#include <d3d11.h>
#include <vector>

namespace DesktopCapture
{
	typedef struct
	{
		void* intermediaryBuffer;
		void* outputBuffer;
		void* outputBufferBlurred;
		cudaStream_t stream;
	} CudaBuffers;

	/**
	 * Gets the average color from a number of regions in a texture.
	 */
	class CudaMeanColorCalculator
	{
	public:
		CudaMeanColorCalculator();
		~CudaMeanColorCalculator();

		/**
		 *	Initialize with a sampling specification and the texture where the caller will store frame data
		*	before passing it to this class.
		*	@param samplingParameters Specifications for how to sample the texture
		*	@param frameBuffer The texture where frame data will be stored
		*	@param width The width of the texture
		*	@param height The height of the texture
		*/
		void initialize(const std::vector<SamplingSpecification>& samplingParameters,
						ID3D11Texture2D* frameBuffer,
						const int& width, const int& height);


		/**
		 * Samples the frameBuffer specificed in the initialize call according to the sampling specifications.
		 * @param activeRegion The part of the texture to actually use/evaluate
		 * @param out A list of arrays in which to place the resulting colors.
		 */
		void getMeanColors(Rect activeRegion, const std::vector<RgbColor*>& out);

	private:
		std::vector<SamplingSpecification> specifications;
		int width, height;

		cudaGraphicsResource* cudaResource;
		void* textureBuffer;
		size_t textureBufferPitch;

		std::vector<CudaBuffers> bufferSets;
	};
}
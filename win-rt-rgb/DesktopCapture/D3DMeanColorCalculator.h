#pragma once
#include "Color.h"
#include "Types.h"
#include <d3d11.h>
#include <vector>

namespace DesktopCapture
{
	/**
	*	Samples a d3d texture and returns its average color.
	*	Multiple colors can be generated per frame, and will be picked from evenly sized
	*	sections divided over the horizontal space.
	*/
	class D3DMeanColorCalculator
	{
		ID3D11DeviceContext* deviceContext = nullptr;
		ID3D11Texture2D *frameBuffer = nullptr;
		ID3D11ShaderResourceView *frameBufferView = nullptr;
		ID3D11Texture2D *mappingBuffer = nullptr;
		UINT width, height;
		uint32_t* cpuBuffer;

		std::vector<std::vector<RgbColor>> outputBuffers;
		std::vector<RgbColor*> results;
		std::vector<SamplingSpecification> specifications;

		// Places the given region from frameBuffer in cpuBuffer
		void copyToCpuBuffer(Rect region, unsigned int srcMipLevel);
		void adjustSaturation();

	public:
		/**
		 *	Initialize this object.
		*	@param device The device to expect textures from
		*	@param textureWidth The width of textures to be sampled
		*	@param textureHeight The height of textures to be sampled
		*	@param samplingParameters Specifications for how to sample the texture. Each call to sample will return one result
				per specification.
		*/
		void initialize(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight,
						const std::vector<SamplingSpecification>& samplingParameters);

		/**
		 * Set the frame to be used for the next calculation
		 */
		void setFrameData(ID3D11Texture2D *frame);

		/**
		 *	Calculate the mean color of nSamplesPerFrame sections within (a region of) the current frame.
		*	Returns one array of colors per specification given in the constructor.
		*	The contents of the color arrays may be overwritten and are valid until this method is called again.
		*/
		std::vector<RgbColor*> sample(Rect activeRegion);

		D3DMeanColorCalculator();
		~D3DMeanColorCalculator();

		D3DMeanColorCalculator(D3DMeanColorCalculator const&) = delete;
		D3DMeanColorCalculator(D3DMeanColorCalculator &&) = delete;
		D3DMeanColorCalculator operator=(D3DMeanColorCalculator const&) = delete;
	};
}

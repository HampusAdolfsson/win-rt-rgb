#pragma once
#include "Color.h"
#include "Types.h"
#include <d3d11.h>
#include <vector>
#include <mutex>

namespace DesktopCapture
{
	class D3DMeanColorCalculator;
	/**
	 * Encapsulates a sampling specification that is to be used by a D3DMeanColorCalculator,
	 * i.e. to be passed to the sample method.
	 * An handle can be used multiple times, and can even be used by several
	 * D3DMeanColorCalculators, in order to avoid allocating buffers too often.
	 */
	class D3DMeanColorSpecificationHandle
	{
		friend D3DMeanColorCalculator;
		SamplingSpecification specification;
		std::vector<RgbColor> outputBuffer;
	public:
		D3DMeanColorSpecificationHandle(SamplingSpecification specification);

		/** Gets the results from the last sample call involving this handle **/
		inline const std::vector<RgbColor>& getResults() const { return outputBuffer; };
	};

	/**
	*	Samples a d3d texture and returns one or more sets of its average colors,
	*	picked from evenly sized columns divided over the horizontal space.
	*/
	class D3DMeanColorCalculator
	{
		ID3D11DeviceContext* deviceContext = nullptr;
		ID3D11Texture2D *frameBuffer = nullptr;
		ID3D11ShaderResourceView *frameBufferView = nullptr;
		ID3D11Texture2D *mappingBuffer = nullptr;
		UINT width, height;
		uint32_t* cpuBuffer;
		std::mutex bufferLock;

		// Places the given region from frameBuffer in cpuBuffer
		void copyToCpuBuffer(Rect region, unsigned int srcMipLevel);

	public:
		/**
		 *	Initialize this object.
		*	@param device The device to expect textures from
		*	@param textureWidth The width of textures to be sampled
		*	@param textureHeight The height of textures to be sampled
		*/
		D3DMeanColorCalculator(ID3D11Device* device, const UINT& textureWidth, const UINT& textureHeight);
		~D3DMeanColorCalculator();

		/**
		 *	Set the frame to be used for the next calculation
		 */
		void setFrameData(ID3D11Texture2D *frame);

		/**
		*	Calculate the mean color of some columns within (a region of) the current frame.
		*	Produces one array of colors for each specification handle passed, and places the results in
		*	the respective handle.
		*	The contents of the color arrays may be overwritten and are valid until the next time the handles
		*	are used to sample a frame.
		*/
		void sample(std::vector<D3DMeanColorSpecificationHandle*> handles, Rect activeRegion);

		D3DMeanColorCalculator(D3DMeanColorCalculator const&) = delete;
		D3DMeanColorCalculator operator=(D3DMeanColorCalculator const&) = delete;
		D3DMeanColorCalculator(D3DMeanColorCalculator &&);
	};
}

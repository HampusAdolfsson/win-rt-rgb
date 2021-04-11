#pragma once
#include "Color.h"
#include "Types.h"
#include <d3d11.h>
#include <vector>
#include <mutex>

namespace DesktopCapture
{
	typedef std::vector<RgbColor> ColorBuffer;

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
		*	Calculate the mean color of some columns within (a portion of) the current frame.
		*	Produces one array of colors for each buffer passed, and places the results in
		*	the respective buffer.
		*	The contents of the color arrays may be overwritten and are valid until the next time the handles
		*	are used to sample a frame.
		*/
		void sample(std::vector<ColorBuffer*> buffers, Rect activeRegion);

		D3DMeanColorCalculator(D3DMeanColorCalculator const&) = delete;
		D3DMeanColorCalculator operator=(D3DMeanColorCalculator const&) = delete;
		D3DMeanColorCalculator(D3DMeanColorCalculator &&);
	};
}

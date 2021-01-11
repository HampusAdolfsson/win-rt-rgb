#pragma once
#include <dxgi1_2.h>
#include <d3d11.h>

namespace DesktopCapture
{
	/**
	*	Uses the Desktop Duplication API to capture a monitor as a D3D11 texture.
	*/
	class DesktopDuplicator
	{
		ID3D11Device* device;
		DXGI_OUTPUT_DESC outputDesc;
		IDXGIOutputDuplication* outputDuplication;
		UINT outputIdx;

		ID3D11Texture2D* currentFrame;

		static bool isExpectedError(const HRESULT& hr);
		void reInitialize();

	public:

		/**
		*	Get the duplicator ready to capture frames.
		*	@param device The device to capture from.
		*	@param outputIdx The index of the output (monitor) to capture
		*/
		void initialize(ID3D11Device *device, const UINT& outputIdx);

		ID3D11Texture2D* captureFrame();
		void releaseFrame();

		const UINT getFrameWidth() const;
		const UINT getFrameHeight() const;

		DesktopDuplicator();
		~DesktopDuplicator();
	};
}
#pragma once
#include <dxgi1_2.h>
#include <d3d11.h>
#include <thread>
#include <functional>

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

		std::function<void(ID3D11Texture2D*)> callback;
		bool isRunning;
		std::thread samplerThread;

		static bool isExpectedError(const HRESULT& hr);
		void reInitialize();

		void sampleLoop();

		ID3D11Texture2D* captureFrame();

	public:

		/**
		*	Get the duplicator ready to capture frames.
		*	@param outputIdx The index of the output (monitor) to capture
		*/
		DesktopDuplicator(ID3D11Device* device, UINT outputIdx,
							std::function<void(ID3D11Texture2D*)> callback);
		~DesktopDuplicator();

		void releaseFrame();

		const UINT getFrameWidth() const;
		const UINT getFrameHeight() const;

		void start();
		void stop();

		DesktopDuplicator(DesktopDuplicator const&) = delete;
		DesktopDuplicator operator=(DesktopDuplicator const&) = delete;
		DesktopDuplicator(DesktopDuplicator &&);
	};
}
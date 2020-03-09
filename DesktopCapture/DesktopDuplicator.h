#pragma once
#include <dxgi1_2.h>
#include <d3d11.h>

/**
*	Uses the Desktop Duplication API to capture the screen.
*/
class DesktopDuplicator
{
	ID3D11Device* device;
	DXGI_OUTPUT_DESC outputDesc;
	IDXGIOutputDuplication* outputDuplication;
	UINT outputIdx;

	ID3D11Texture2D* currentFrame;

	bool isExpectedError(HRESULT hr);
	void reInitialize();

public:

	/**
	*	Get the duplicator ready to capture frames.
	*	@param device The device to capture from.
	*	@param outputIdx The index of the output (monitor) to capture
	*/
	void initialize(ID3D11Device *device, UINT outputIdx);

	ID3D11Texture2D* captureFrame();
	void releaseFrame();

	UINT getFrameWidth();
	UINT getFrameHeight();

	DesktopDuplicator();
	~DesktopDuplicator();
};


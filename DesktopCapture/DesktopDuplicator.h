#pragma once
#include "Errors.h"
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

	ID3D11Texture2D* currentFrame;

public:

	/**
	*	Get the duplicator ready to capture frames.
	*	@param device The device to capture from.
	*	@param outputIdx The index of the output (monitor) to capture
	*/
	DuplReturn_t initialize(ID3D11Device *device, UINT outputIdx);

	DuplReturn_t captureFrame(_Out_ ID3D11Texture2D** frame, _Out_ bool *timedOut);
	DuplReturn_t releaseFrame();

	UINT getFrameWidth();
	UINT getFrameHeight();

	DesktopDuplicator();
	~DesktopDuplicator();
};


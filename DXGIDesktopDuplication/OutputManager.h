// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved

#ifndef _OUTPUTMANAGER_H_
#define _OUTPUTMANAGER_H_

#include <stdio.h>

#include "CommonTypes.h"
#include "warning.h"

//
// Handles the task of drawing into a window.
// Has the functionality to draw the mouse given a mouse shape buffer and position
//
class OUTPUTMANAGER
{
    public:
        OUTPUTMANAGER();
        ~OUTPUTMANAGER();
        DUPL_RETURN InitOutput(HWND Window, _Out_ UINT* OutCount, _Out_ RECT* DeskBounds);
        DUPL_RETURN UpdateApplicationWindow(_Inout_ bool* Occluded);
        void CleanRefs();
        HANDLE GetSharedHandle();
        void WindowResize();

    private:
    // Methods
        DUPL_RETURN MakeRTV();
        void SetViewPort(UINT Width, UINT Height);
        DUPL_RETURN InitShaders();
        DUPL_RETURN InitGeometry();
        DUPL_RETURN CreateSharedSurf(_Out_ UINT* OutCount, _Out_ RECT* DeskBounds);
        DUPL_RETURN DrawFrame();
        DUPL_RETURN ResizeSwapChain();

    // Vars
        IDXGISwapChain1* m_SwapChain;
        ID3D11Device* m_Device;
        IDXGIFactory2* m_Factory;
        ID3D11DeviceContext* m_DeviceContext;
        ID3D11RenderTargetView* m_RTV;
        ID3D11SamplerState* m_SamplerLinear;
        ID3D11BlendState* m_BlendState;
        ID3D11VertexShader* m_VertexShader;
        ID3D11PixelShader* m_PixelShader;
        ID3D11InputLayout* m_InputLayout;
        ID3D11Texture2D* m_SharedSurf;
        IDXGIKeyedMutex* m_KeyMutex;
        HWND m_WindowHandle;
        bool m_NeedsResize;
        DWORD m_OcclusionCookie;
};

#endif

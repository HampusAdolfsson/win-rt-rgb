#pragma once

#include "CommonTypes.h"

//
// Handles the task of duplicating an output.
//
class DuplicationManager
{
    public:
        DuplicationManager();
        ~DuplicationManager();
        DUPL_RETURN GetFrame(_Out_ FRAME_DATA* Data, _Out_ bool* Timeout);
        DUPL_RETURN DoneWithFrame();
        DUPL_RETURN InitDupl(_In_ ID3D11Device* Device, UINT Output);
        void GetOutputDesc(_Out_ DXGI_OUTPUT_DESC* DescPtr);

    private:

    // vars
        IDXGIOutputDuplication* m_DeskDupl;
        ID3D11Texture2D* m_AcquiredDesktopImage;
        _Field_size_bytes_(m_MetaDataSize) BYTE* m_MetaDataBuffer;
        UINT m_MetaDataSize;
        UINT m_OutputNumber;
        DXGI_OUTPUT_DESC m_OutputDesc;
        ID3D11Device* m_Device;
};

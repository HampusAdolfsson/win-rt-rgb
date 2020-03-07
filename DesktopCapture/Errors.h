#pragma once
#include <d3d11.h>

typedef enum
{
    DUPL_RETURN_SUCCESS             = 0,
    DUPL_RETURN_ERROR_EXPECTED      = 1,
    DUPL_RETURN_ERROR_UNEXPECTED    = 2
} DuplReturn_t;

// determines the type of error that occured (expected or unexpected)
DuplReturn_t ProcessError(ID3D11Device* device, HRESULT hr, HRESULT* expectedErrors);

// These are the errors we expect from general Dxgi API due to a transition
extern HRESULT SystemTransitionsExpectedErrors[4];
// These are the errors we expect from IDXGIOutput1::DuplicateOutput due to a transition
extern HRESULT CreateDuplicationExpectedErrors[5];
// These are the errors we expect from IDXGIOutputDuplication methods due to a transition
extern HRESULT FrameInfoExpectedErrors[3];
// These are the errors we expect from IDXGIAdapter::EnumOutputs methods due to outputs becoming stale during a transition
extern HRESULT EnumOutputsExpectedErrors[2];

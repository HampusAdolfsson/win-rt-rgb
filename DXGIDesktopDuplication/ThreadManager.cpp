// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved

#include "ThreadManager.h"
#include <system_error>

ThreadManager::ThreadManager() : threadCount(0)
{
}

ThreadManager::~ThreadManager()
{
    Clean();
}

//
// Clean up resources
//
void ThreadManager::Clean()
{
	for (UINT i = 0; i < threads.size(); ++i)
	{
        threads[i]->stop();
        delete threads[i];
	}
    threads.clear();

    threadCount = 0;
}

DUPL_RETURN InitializeDx(_Out_ DX_RESOURCES* Data)
{
    HRESULT hr = S_OK;

    // Driver types supported
    D3D_DRIVER_TYPE DriverTypes[] =
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    UINT NumDriverTypes = ARRAYSIZE(DriverTypes);

    // Feature levels supported
    D3D_FEATURE_LEVEL FeatureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_1
    };
    UINT NumFeatureLevels = ARRAYSIZE(FeatureLevels);

    D3D_FEATURE_LEVEL FeatureLevel;

    // Create device
    for (UINT DriverTypeIndex = 0; DriverTypeIndex < NumDriverTypes; ++DriverTypeIndex)
    {
        hr = D3D11CreateDevice(nullptr, DriverTypes[DriverTypeIndex], nullptr, 0, FeatureLevels, NumFeatureLevels,
            D3D11_SDK_VERSION, &Data->Device, &FeatureLevel, &Data->Context);
        if (SUCCEEDED(hr))
        {
            // Device creation success, no need to loop anymore
            break;
        }
    }
    if (FAILED(hr))
    {
        return ProcessFailure(nullptr, L"Failed to create device in InitializeDx", L"Error", hr);
    }

    // Set up sampler
    D3D11_SAMPLER_DESC SampDesc;
    RtlZeroMemory(&SampDesc, sizeof(SampDesc));
    SampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    SampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    SampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    SampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    SampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    SampDesc.MinLOD = 0;
    SampDesc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = Data->Device->CreateSamplerState(&SampDesc, &Data->SamplerLinear);
    if (FAILED(hr))
    {
        return ProcessFailure(Data->Device, L"Failed to create sampler state in InitializeDx", L"Error", hr, SystemTransitionsExpectedErrors);
    }

    return DUPL_RETURN_SUCCESS;
}

//
// Start up threads for DDA
//
DUPL_RETURN ThreadManager::Initialize(UINT OutputCount, HANDLE UnexpectedErrorEvent, HANDLE ExpectedErrorEvent, HANDLE SharedHandle, _In_ RECT* DesktopDim)
{
  
    threadCount = OutputCount;
    for (int i = 0; i < threadCount; i++) {
		DX_RESOURCES dxRes;
		InitializeDx(&dxRes);
        threads.push_back(new DuplicationThread(i, UnexpectedErrorEvent, ExpectedErrorEvent, SharedHandle, DesktopDim, dxRes));
        DUPL_RETURN ret = threads[i]->start();
        if (ret != DUPL_RETURN_SUCCESS) {
            return ret;
        }
    }
    return DUPL_RETURN_SUCCESS;
}

//
//
// Waits infinitely for all spawned threads to terminate
//
void ThreadManager::WaitForThreadTermination()
{
    for (int i = 0; i < threads.size(); i++) {
        threads[i]->stop();
    }
}


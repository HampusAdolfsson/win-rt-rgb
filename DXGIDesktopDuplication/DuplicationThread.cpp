#include "DuplicationThread.h"
#include "DuplicationManager.h"
#include <system_error>

DuplicationThread::DuplicationThread(UINT output, HANDLE unexpectedErrorEvent, HANDLE expectedErrorEvent, HANDLE texSharedHandle, RECT* desktopDim, DX_RESOURCES dxRes)
    : output(output),
    unexpectedErrorEvent(unexpectedErrorEvent),
    expectedErrorEvent(expectedErrorEvent),
    texSharedHandle(texSharedHandle),
	offsetX(desktopDim->left),
	offsetY(desktopDim->top),
	running(false),
    dxRes(dxRes)
{
}

DUPL_RETURN DuplicationThread::start() {
	running = true;
    //DUPL_RETURN ret = initializeDx();
    //if (ret != DUPL_RETURN_SUCCESS)
    //{
    //    return ret;
    //}
    try
    {
        thread = std::thread(&DuplicationThread::doThread, this);
    }
    catch (std::system_error err) {
		return ProcessFailure(nullptr, L"Failed to create thread", L"Error", E_FAIL);
	}
    return DUPL_RETURN_SUCCESS;
}
void DuplicationThread::stop() {
	running = false;
    if (thread.joinable()) {
        thread.join();
    }
}

void doStuff(ID3D11Texture2D* tex, UINT width, UINT height, DX_RESOURCES dxres) {

    D3D11_TEXTURE2D_DESC texDesc;
    RtlZeroMemory(&texDesc, sizeof(texDesc));
    texDesc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
    texDesc.BindFlags = 0;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.MiscFlags = 0;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_STAGING;
    ID3D11Texture2D* buf;
    HRESULT hr = dxres.Device->CreateTexture2D(&texDesc, nullptr, &buf);
    dxres.Context->CopyResource(buf, tex);
    D3D11_MAPPED_SUBRESOURCE subres;
    hr = dxres.Context->Map(buf, 0, D3D11_MAP_READ, 0, &subres);
    printf("%d\n", ((int*)subres.pData)[0]);
}

void DuplicationThread::doThread() {

    // Classes
    DuplicationManager DuplMgr;

    // D3D objects
    ID3D11Texture2D* SharedSurf = nullptr;
    IDXGIKeyedMutex* KeyMutex = nullptr;

    // Get desktop
    DUPL_RETURN Ret;
    HDESK CurrentDesktop = nullptr;
    CurrentDesktop = OpenInputDesktop(0, FALSE, GENERIC_ALL);
    if (!CurrentDesktop)
    {
        // We do not have access to the desktop so request a retry
        SetEvent(expectedErrorEvent);
        Ret = DUPL_RETURN_ERROR_EXPECTED;
        goto Exit;
    }

    // Attach desktop to this thread
    bool DesktopAttached = SetThreadDesktop(CurrentDesktop) != 0;
    CloseDesktop(CurrentDesktop);
    CurrentDesktop = nullptr;
    if (!DesktopAttached)
    {
        // We do not have access to the desktop so request a retry
        Ret = DUPL_RETURN_ERROR_EXPECTED;
        goto Exit;
    }

    // Obtain handle to sync shared Surface
    HRESULT hr = dxRes.Device->OpenSharedResource(texSharedHandle, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&SharedSurf));
    if (FAILED (hr))
    {
        Ret = ProcessFailure(dxRes.Device, L"Opening shared texture failed", L"Error", hr, SystemTransitionsExpectedErrors);
        goto Exit;
    }

    hr = SharedSurf->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void**>(&KeyMutex));
    if (FAILED(hr))
    {
        Ret = ProcessFailure(nullptr, L"Failed to get keyed mutex interface in spawned thread", L"Error", hr);
        goto Exit;
    }

    // Make duplication manager
    Ret = DuplMgr.InitDupl(dxRes.Device, output);
    if (Ret != DUPL_RETURN_SUCCESS)
    {
        goto Exit;
    }

    // Get output description
    DXGI_OUTPUT_DESC DesktopDesc;
    RtlZeroMemory(&DesktopDesc, sizeof(DXGI_OUTPUT_DESC));
    DuplMgr.GetOutputDesc(&DesktopDesc);

    // Main duplication loop
    bool WaitToProcessCurrentFrame = false;
    FRAME_DATA CurrentData;


	while (running) {
        printf("%d!\n", output);
		if (!WaitToProcessCurrentFrame)
		{
			// Get new frame from desktop duplication
			bool TimeOut;
			Ret = DuplMgr.GetFrame(&CurrentData, &TimeOut);
			if (Ret != DUPL_RETURN_SUCCESS)
			{
				// An error occurred getting the next frame drop out of loop which
				// will check if it was expected or not
				break;
			}
            doStuff(CurrentData.Frame, 1920, 1080, dxRes);

			// Check for timeout
			if (TimeOut)
			{
				// No new frame at the moment
				continue;
			}
		}

		// We have a new frame so try and process it
		// Try to acquire keyed mutex in order to access shared surface
		hr = KeyMutex->AcquireSync(0, 1000);
		if (hr == static_cast<HRESULT>(WAIT_TIMEOUT))
		{
			// Can't use shared surface right now, try again later
			WaitToProcessCurrentFrame = true;
			continue;
		}
		else if (FAILED(hr))
		{
			// Generic unknown failure
			Ret = ProcessFailure(dxRes.Device, L"Unexpected error acquiring KeyMutex", L"Error", hr, SystemTransitionsExpectedErrors);
			DuplMgr.DoneWithFrame();
			break;
		}

		// We can now process the current frame
		WaitToProcessCurrentFrame = false;

		// Process new frame
		//Ret = DispMgr.ProcessFrame(&CurrentData, SharedSurf, offsetX, offsetY, &DesktopDesc);
		//if (Ret != DUPL_RETURN_SUCCESS)
		//{
		//	DuplMgr.DoneWithFrame();
		//	KeyMutex->ReleaseSync(1);
		//	break;
		//}

		// Release acquired keyed mutex
		hr = KeyMutex->ReleaseSync(1);
		if (FAILED(hr))
		{
			Ret = ProcessFailure(dxRes.Device, L"Unexpected error releasing the keyed mutex", L"Error", hr, SystemTransitionsExpectedErrors);
			DuplMgr.DoneWithFrame();
			break;
		}

		// Release frame back to desktop duplication
		Ret = DuplMgr.DoneWithFrame();
		if (Ret != DUPL_RETURN_SUCCESS)
		{
			break;
		}

	}
Exit:
    if (Ret != DUPL_RETURN_SUCCESS)
    {
        if (Ret == DUPL_RETURN_ERROR_EXPECTED)
        {
            // The system is in a transition state so request the duplication be restarted
            SetEvent(expectedErrorEvent);
        }
        else
        {
            // Unexpected error so exit the application
            SetEvent(unexpectedErrorEvent);
        }
    }

    if (SharedSurf)
    {
        SharedSurf->Release();
        SharedSurf = nullptr;
    }

    if (KeyMutex)
    {
        KeyMutex->Release();
        KeyMutex = nullptr;
    }
    //cleanDx();
}

DUPL_RETURN DuplicationThread::initializeDx() {
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
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
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
                                D3D11_SDK_VERSION, &dxRes.Device, &FeatureLevel, &dxRes.Context);
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
    hr = dxRes.Device->CreateSamplerState(&SampDesc, &dxRes.SamplerLinear);
    if (FAILED(hr))
    {
        return ProcessFailure(dxRes.Device, L"Failed to create sampler state in InitializeDx", L"Error", hr, SystemTransitionsExpectedErrors);
    }

    return DUPL_RETURN_SUCCESS;
}

//
// Clean up DX_RESOURCES
//
void DuplicationThread::cleanDx()
{
    if (dxRes.Device)
    {
        dxRes.Device->Release();
        dxRes.Device = nullptr;
    }

    if (dxRes.Context)
    {
        dxRes.Context->Release();
        dxRes.Context = nullptr;
    }

    if (dxRes.SamplerLinear)
    {
        dxRes.SamplerLinear->Release();
        dxRes.SamplerLinear = nullptr;
    }
}



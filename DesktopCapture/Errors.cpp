#include "Errors.h"

DuplReturn_t ProcessError(ID3D11Device* device, HRESULT hr, HRESULT* expectedErrors) {
    HRESULT translatedHr;

    // On an error check if the DX device is lost
    if (device)
    {
        HRESULT DeviceRemovedReason = device->GetDeviceRemovedReason();

        switch (DeviceRemovedReason)
        {
        case DXGI_ERROR_DEVICE_REMOVED:
        case DXGI_ERROR_DEVICE_RESET:
            case static_cast<HRESULT>(E_OUTOFMEMORY) :
            {
                // Our device has been stopped due to an external event on the GPU so map them all to
                // device removed and continue processing the condition
                translatedHr = DXGI_ERROR_DEVICE_REMOVED;
                break;
            }

            case S_OK:
            {
                // Device is not removed so use original error
                translatedHr = hr;
                break;
            }

            default:
            {
                // Device is removed but not a error we want to remap
                translatedHr = DeviceRemovedReason;
            }
        }
    }
    else
    {
        translatedHr = hr;
    }

    // Check if this error was expected or not
    if (expectedErrors)
    {
        HRESULT* CurrentResult = expectedErrors;

        while (*CurrentResult != S_OK)
        {
            if (*(CurrentResult++) == translatedHr)
            {
                return DUPL_RETURN_ERROR_EXPECTED;
            }
        }
    }

    return DUPL_RETURN_ERROR_UNEXPECTED;
}

HRESULT SystemTransitionsExpectedErrors[] = {
                                                DXGI_ERROR_DEVICE_REMOVED,
                                                DXGI_ERROR_ACCESS_LOST,
                                                static_cast<HRESULT>(WAIT_ABANDONED),
                                                S_OK                                    // Terminate list with zero valued HRESULT
};

HRESULT CreateDuplicationExpectedErrors[] = {
                                                DXGI_ERROR_DEVICE_REMOVED,
                                                static_cast<HRESULT>(E_ACCESSDENIED),
                                                DXGI_ERROR_UNSUPPORTED,
                                                DXGI_ERROR_SESSION_DISCONNECTED,
                                                S_OK                                    // Terminate list with zero valued HRESULT
};

HRESULT FrameInfoExpectedErrors[] = {
                                        DXGI_ERROR_DEVICE_REMOVED,
                                        DXGI_ERROR_ACCESS_LOST,
                                        S_OK                                    // Terminate list with zero valued HRESULT
};

HRESULT EnumOutputsExpectedErrors[] = {
                                          DXGI_ERROR_NOT_FOUND,
                                          S_OK                                    // Terminate list with zero valued HRESULT
};


// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved

#include <limits.h>

#include "DisplayManager.h"
#include "DuplicationManager.h"
#include "OutputManager.h"
#include "ThreadManager.h"

//
// Globals
//
OUTPUTMANAGER OutMgr;

// Below are lists of errors expect from Dxgi API calls when a transition event like mode change, PnpStop, PnpStart
// desktop switch, TDR or session disconnect/reconnect. In all these cases we want the application to clean up the threads that process
// the desktop updates and attempt to recreate them.
// If we get an error that is not on the appropriate list then we exit the application

// These are the errors we expect from general Dxgi API due to a transition
HRESULT SystemTransitionsExpectedErrors[] = {
                                                DXGI_ERROR_DEVICE_REMOVED,
                                                DXGI_ERROR_ACCESS_LOST,
                                                static_cast<HRESULT>(WAIT_ABANDONED),
                                                S_OK                                    // Terminate list with zero valued HRESULT
                                            };

// These are the errors we expect from IDXGIOutput1::DuplicateOutput due to a transition
HRESULT CreateDuplicationExpectedErrors[] = {
                                                DXGI_ERROR_DEVICE_REMOVED,
                                                static_cast<HRESULT>(E_ACCESSDENIED),
                                                DXGI_ERROR_UNSUPPORTED,
                                                DXGI_ERROR_SESSION_DISCONNECTED,
                                                S_OK                                    // Terminate list with zero valued HRESULT
                                            };

// These are the errors we expect from IDXGIOutputDuplication methods due to a transition
HRESULT FrameInfoExpectedErrors[] = {
                                        DXGI_ERROR_DEVICE_REMOVED,
                                        DXGI_ERROR_ACCESS_LOST,
                                        S_OK                                    // Terminate list with zero valued HRESULT
                                    };

// These are the errors we expect from IDXGIAdapter::EnumOutputs methods due to outputs becoming stale during a transition
HRESULT EnumOutputsExpectedErrors[] = {
                                          DXGI_ERROR_NOT_FOUND,
                                          S_OK                                    // Terminate list with zero valued HRESULT
                                      };


//
// Forward Declarations
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
void ShowHelp();

//
// Class for progressive waits
//
typedef struct
{
    UINT    WaitTime;
    UINT    WaitCount;
}WAIT_BAND;

#define WAIT_BAND_COUNT 3
#define WAIT_BAND_STOP 0

class DYNAMIC_WAIT
{
    public :
        DYNAMIC_WAIT();
        ~DYNAMIC_WAIT();

        void Wait();

    private :

    static const WAIT_BAND   m_WaitBands[WAIT_BAND_COUNT];

    // Period in seconds that a new wait call is considered part of the same wait sequence
    static const UINT       m_WaitSequenceTimeInSeconds = 2;

    UINT                    m_CurrentWaitBandIdx;
    UINT                    m_WaitCountInCurrentBand;
    LARGE_INTEGER           m_QPCFrequency;
    LARGE_INTEGER           m_LastWakeUpTime;
    BOOL                    m_QPCValid;
};
const WAIT_BAND DYNAMIC_WAIT::m_WaitBands[WAIT_BAND_COUNT] = {
                                                                 {250, 20},
                                                                 {2000, 60},
                                                                 {5000, WAIT_BAND_STOP}   // Never move past this band
                                                             };

DYNAMIC_WAIT::DYNAMIC_WAIT() : m_CurrentWaitBandIdx(0), m_WaitCountInCurrentBand(0)
{
    m_QPCValid = QueryPerformanceFrequency(&m_QPCFrequency);
    m_LastWakeUpTime.QuadPart = 0L;
}

DYNAMIC_WAIT::~DYNAMIC_WAIT()
{
}

void DYNAMIC_WAIT::Wait()
{
    LARGE_INTEGER CurrentQPC = {0};

    // Is this wait being called with the period that we consider it to be part of the same wait sequence
    QueryPerformanceCounter(&CurrentQPC);
    if (m_QPCValid && (CurrentQPC.QuadPart <= (m_LastWakeUpTime.QuadPart + (m_QPCFrequency.QuadPart * m_WaitSequenceTimeInSeconds))))
    {
        // We are still in the same wait sequence, lets check if we should move to the next band
        if ((m_WaitBands[m_CurrentWaitBandIdx].WaitCount != WAIT_BAND_STOP) && (m_WaitCountInCurrentBand > m_WaitBands[m_CurrentWaitBandIdx].WaitCount))
        {
            m_CurrentWaitBandIdx++;
            m_WaitCountInCurrentBand = 0;
        }
    }
    else
    {
        // Either we could not get the current time or we are starting a new wait sequence
        m_WaitCountInCurrentBand = 0;
        m_CurrentWaitBandIdx = 0;
    }

    // Sleep for the required period of time
    Sleep(m_WaitBands[m_CurrentWaitBandIdx].WaitTime);

    // Record the time we woke up so we can detect wait sequences
    QueryPerformanceCounter(&m_LastWakeUpTime);
    m_WaitCountInCurrentBand++;
}


//
// Program entry point
//
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ INT nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // Synchronization
    HANDLE UnexpectedErrorEvent = nullptr;
    HANDLE ExpectedErrorEvent = nullptr;

    // Window
    HWND WindowHandle = nullptr;

    // Event used by the threads to signal an unexpected error and we want to quit the app
    UnexpectedErrorEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    if (!UnexpectedErrorEvent)
    {
        ProcessFailure(nullptr, L"UnexpectedErrorEvent creation failed", L"Error", E_UNEXPECTED);
        return 0;
    }

    // Event for when a thread encounters an expected error
    ExpectedErrorEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    if (!ExpectedErrorEvent)
    {
        ProcessFailure(nullptr, L"ExpectedErrorEvent creation failed", L"Error", E_UNEXPECTED);
        return 0;
    }

    // Load simple cursor
    HCURSOR Cursor = nullptr;
    Cursor = LoadCursor(nullptr, IDC_ARROW);
    if (!Cursor)
    {
        ProcessFailure(nullptr, L"Cursor load failed", L"Error", E_UNEXPECTED);
        return 0;
    }

    // Register class
    WNDCLASSEXW Wc;
    Wc.cbSize           = sizeof(WNDCLASSEXW);
    Wc.style            = CS_HREDRAW | CS_VREDRAW;
    Wc.lpfnWndProc      = WndProc;
    Wc.cbClsExtra       = 0;
    Wc.cbWndExtra       = 0;
    Wc.hInstance        = hInstance;
    Wc.hIcon            = nullptr;
    Wc.hCursor          = Cursor;
    Wc.hbrBackground    = nullptr;
    Wc.lpszMenuName     = nullptr;
    Wc.lpszClassName    = L"ddasample";
    Wc.hIconSm          = nullptr;
    if (!RegisterClassExW(&Wc))
    {
        ProcessFailure(nullptr, L"Window class registration failed", L"Error", E_UNEXPECTED);
        return 0;
    }

    // Create window
    RECT WindowRect = {0, 0, 800, 600};
    AdjustWindowRect(&WindowRect, WS_OVERLAPPEDWINDOW, FALSE);
    WindowHandle = CreateWindowW(L"ddasample", L"DXGI desktop duplication sample",
                           WS_OVERLAPPEDWINDOW,
                           0, 0,
                           WindowRect.right - WindowRect.left, WindowRect.bottom - WindowRect.top,
                           nullptr, nullptr, hInstance, nullptr);
    if (!WindowHandle)
    {
        ProcessFailure(nullptr, L"Window creation failed", L"Error", E_FAIL);
        return 0;
    }

    DestroyCursor(Cursor);

    ShowWindow(WindowHandle, nCmdShow);
    UpdateWindow(WindowHandle);

    ThreadManager ThreadMgr;
    RECT DeskBounds;
    UINT OutputCount;

    // Message loop (attempts to update screen when no other messages to process)
    MSG msg = {0};
    bool FirstTime = true;
    bool Occluded = true;
    DYNAMIC_WAIT DynamicWait;

    while (WM_QUIT != msg.message)
    {
        DUPL_RETURN Ret = DUPL_RETURN_SUCCESS;
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == OCCLUSION_STATUS_MSG)
            {
                // Present may not be occluded now so try again
                Occluded = false;
            }
            else
            {
                // Process window messages
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
        else if (WaitForSingleObjectEx(UnexpectedErrorEvent, 0, FALSE) == WAIT_OBJECT_0)
        {
            // Unexpected error occurred so exit the application
            break;
        }
        else if (FirstTime || WaitForSingleObjectEx(ExpectedErrorEvent, 0, FALSE) == WAIT_OBJECT_0)
        {
            if (!FirstTime)
            {
                // Terminate other threads
                ThreadMgr.WaitForThreadTermination();
                ResetEvent(ExpectedErrorEvent);

                // Clean up
                ThreadMgr.Clean();
                OutMgr.CleanRefs();

                // As we have encountered an error due to a system transition we wait before trying again, using this dynamic wait
                // the wait periods will get progressively long to avoid wasting too much system resource if this state lasts a long time
                DynamicWait.Wait();
            }
            else
            {
                // First time through the loop so nothing to clean up
                FirstTime = false;
            }

            // Re-initialize
            Ret = OutMgr.InitOutput(WindowHandle, &OutputCount, &DeskBounds);
            if (Ret == DUPL_RETURN_SUCCESS)
            {
                HANDLE SharedHandle = OutMgr.GetSharedHandle();
                if (SharedHandle)
                {
                    Ret = ThreadMgr.Initialize(OutputCount, UnexpectedErrorEvent, ExpectedErrorEvent, SharedHandle, &DeskBounds);
                }
                else
                {
                    DisplayMsg(L"Failed to get handle of shared surface", L"Error", S_OK);
                    Ret = DUPL_RETURN_ERROR_UNEXPECTED;
                }
            }

            // We start off in occluded state and we should immediate get a occlusion status window message
            Occluded = true;
        }
        else
        {
            // Nothing else to do, so try to present to write out to window if not occluded
            if (!Occluded)
            {
                Ret = OutMgr.UpdateApplicationWindow(&Occluded);
            }
        }

        // Check if for errors
        if (Ret != DUPL_RETURN_SUCCESS)
        {
            if (Ret == DUPL_RETURN_ERROR_EXPECTED)
            {
                // Some type of system transition is occurring so retry
                SetEvent(ExpectedErrorEvent);
            }
            else
            {
                // Unexpected error so exit
                break;
            }
        }
    }

    ThreadMgr.WaitForThreadTermination();

    // Clean up
    CloseHandle(UnexpectedErrorEvent);
    CloseHandle(ExpectedErrorEvent);

    if (msg.message == WM_QUIT)
    {
        // For a WM_QUIT message we should return the wParam value
        return static_cast<INT>(msg.wParam);
    }

    return 0;
}

//
// Shows help
//
void ShowHelp()
{
    DisplayMsg(L"The following optional parameters can be used -\n  /output [all | n]\t\tto duplicate all outputs or the nth output\n  /?\t\t\tto display this help section",
               L"Proper usage", S_OK);
}

//
// Window message processor
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        case WM_DESTROY:
        {
            PostQuitMessage(0);
            break;
        }
        case WM_SIZE:
        {
            // Tell output manager that window size has changed
            OutMgr.WindowResize();
            break;
        }
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }

    return 0;
}

_Post_satisfies_(return != DUPL_RETURN_SUCCESS)
DUPL_RETURN ProcessFailure(_In_opt_ ID3D11Device* Device, _In_ LPCWSTR Str, _In_ LPCWSTR Title, HRESULT hr, _In_opt_z_ HRESULT* ExpectedErrors)
{
    HRESULT TranslatedHr;

    // On an error check if the DX device is lost
    if (Device)
    {
        HRESULT DeviceRemovedReason = Device->GetDeviceRemovedReason();

        switch (DeviceRemovedReason)
        {
            case DXGI_ERROR_DEVICE_REMOVED :
            case DXGI_ERROR_DEVICE_RESET :
            case static_cast<HRESULT>(E_OUTOFMEMORY) :
            {
                // Our device has been stopped due to an external event on the GPU so map them all to
                // device removed and continue processing the condition
                TranslatedHr = DXGI_ERROR_DEVICE_REMOVED;
                break;
            }

            case S_OK :
            {
                // Device is not removed so use original error
                TranslatedHr = hr;
                break;
            }

            default :
            {
                // Device is removed but not a error we want to remap
                TranslatedHr = DeviceRemovedReason;
            }
        }
    }
    else
    {
        TranslatedHr = hr;
    }

    // Check if this error was expected or not
    if (ExpectedErrors)
    {
        HRESULT* CurrentResult = ExpectedErrors;

        while (*CurrentResult != S_OK)
        {
            if (*(CurrentResult++) == TranslatedHr)
            {
                return DUPL_RETURN_ERROR_EXPECTED;
            }
        }
    }

    // Error was not expected so display the message box
    DisplayMsg(Str, Title, TranslatedHr);

    return DUPL_RETURN_ERROR_UNEXPECTED;
}

//
// Displays a message
//
void DisplayMsg(_In_ LPCWSTR Str, _In_ LPCWSTR Title, HRESULT hr)
{
    if (SUCCEEDED(hr))
    {
        MessageBoxW(nullptr, Str, Title, MB_OK);
        return;
    }

    const UINT StringLen = (UINT)(wcslen(Str) + sizeof(" with HRESULT 0x########."));
    wchar_t* OutStr = new wchar_t[StringLen];
    if (!OutStr)
    {
        return;
    }

    INT LenWritten = swprintf_s(OutStr, StringLen, L"%s with 0x%X.", Str, hr);
    if (LenWritten != -1)
    {
        MessageBoxW(nullptr, OutStr, Title, MB_OK);
    }

    delete [] OutStr;
}

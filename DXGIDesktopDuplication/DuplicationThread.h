#pragma once

#include "CommonTypes.h"
#include <Windows.h>
#include <thread>

class DuplicationThread
{
public:
    DuplicationThread(UINT output, HANDLE unexpectedErrorEvent, HANDLE expectedErrorEvent, HANDLE texSharedHandle, RECT* desktopDim, DX_RESOURCES dxRes);

    DUPL_RETURN start();
    void stop();
private:
    void doThread();
    DUPL_RETURN initializeDx();
    void cleanDx();

    bool running;
    std::thread thread;

    // Used to indicate abnormal error condition
    HANDLE unexpectedErrorEvent;

    // Used to indicate a transition event occurred e.g. PnpStop, PnpStart, mode change, TDR, desktop switch and the application needs to recreate the duplication interface
    HANDLE expectedErrorEvent;

    HANDLE texSharedHandle;
    UINT output;
    INT offsetX;
    INT offsetY;
    DX_RESOURCES dxRes;
};


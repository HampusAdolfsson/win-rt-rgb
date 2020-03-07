#pragma once

#include "DuplicationThread.h"
#include <vector>
#include <thread>

class ThreadManager
{
    public:
        ThreadManager();
        ~ThreadManager();
        void Clean();
        DUPL_RETURN Initialize(UINT OutputCount, HANDLE UnexpectedErrorEvent, HANDLE ExpectedErrorEvent, HANDLE SharedHandle, _In_ RECT* DesktopDim);
        void WaitForThreadTermination();

    private:

        UINT threadCount;
        std::vector<DuplicationThread*> threads;
};

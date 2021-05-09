#include "QmkRenderOutput.h"
#include "Logger.h"
#include <Hidsdi.h>
#include <Hidclass.h>
#include <setupapi.h>
#include <memory>
#include <vector>
#include <sstream>
#include <iomanip>
#include <regex>

using namespace Rendering;

#define BUF_SIZ 8*1024

#define LOGLINE(hr) LOGSEVERE("QmkRenderOutput got error: 0x%08lx, line %d", hr, __LINE__);\

QmkRenderOutput::QmkRenderOutput(const std::string& hardwareId, size_t ledCount, unsigned int colorTemperature, float gamma)
 : RenderOutput(ledCount, colorTemperature, gamma),
 writeHandle(INVALID_HANDLE_VALUE),
 hardwareId(hardwareId)
{
}

QmkRenderOutput::~QmkRenderOutput()
{
    if (writeHandle != INVALID_HANDLE_VALUE) CloseHandle(writeHandle);
}

static GUID interfaceClassGuid = { 0x4d1e55b2, 0xf16f, 0x11cf, 0x88, 0xcb, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30};

void QmkRenderOutput::initialize()
{
    HRESULT error;

    HDEVINFO devInfoSet = SetupDiGetClassDevsA(&interfaceClassGuid, nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    SP_DEVINFO_DATA devInfoData;
    ZeroMemory(&devInfoData, sizeof(SP_DEVINFO_DATA));
    devInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
    SP_DEVICE_INTERFACE_DATA devInterfaceData;
    ZeroMemory(&devInterfaceData, sizeof(SP_DEVINFO_DATA));
    devInterfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);


    std::unique_ptr<BYTE> buffer(new BYTE[BUF_SIZ]);
    int devIndex = 0;
    bool found = false;

    while (SetupDiEnumDeviceInterfaces(
                             devInfoSet,
                             nullptr,
                             &interfaceClassGuid,
                             devIndex,
                             &devInterfaceData))
    {
        SetupDiEnumDeviceInfo(devInfoSet, devIndex, &devInfoData);

        devIndex++;
        DWORD regType = 0, regSize = 0;


        if (SetupDiGetDeviceRegistryPropertyA(devInfoSet, &devInfoData, SPDRP_HARDWAREID, &regType, buffer.get(), BUF_SIZ, nullptr))
        {
            if (std::strncmp((char*) buffer.get(), hardwareId.c_str(), min(BUF_SIZ, hardwareId.size())) == 0)
            {
                found = true;
                break;
            }
        }
        else
        {
            error = GetLastError();
            LOGLINE(error);
        }
    }
    if (!found)
    {
        LOGWARNING("Could not find a HID device with hardware id '%s'", hardwareId);
        return;
    }

    DWORD regSize = 0;
    SetupDiGetDeviceInterfaceDetail(devInfoSet, &devInterfaceData, nullptr, 0, &regSize, nullptr);
    PSP_DEVICE_INTERFACE_DETAIL_DATA_A devInterfaceDetailData = (PSP_DEVICE_INTERFACE_DETAIL_DATA_A) buffer.get();
    devInterfaceDetailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);
    if (!SetupDiGetDeviceInterfaceDetailA(devInfoSet, &devInterfaceData, devInterfaceDetailData, BUF_SIZ, nullptr, nullptr))
    {
        error = GetLastError();
        LOGLINE(error);
    }

    writeHandle = CreateFileA(devInterfaceDetailData->DevicePath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, 0);
    if (writeHandle == INVALID_HANDLE_VALUE)
    {
        error = GetLastError();
        LOGLINE(error);
    }

    if (devInfoSet) {
        SetupDiDestroyDeviceInfoList(devInfoSet);
    }
    PHIDP_PREPARSED_DATA ppd;
    auto a = HidD_GetPreparsedData(writeHandle, &ppd);
    HIDP_CAPS caps;
    auto b = HidP_GetCaps(ppd, &caps);
}

void QmkRenderOutput::drawImpl(const RenderTarget& target)
{
    if (writeHandle == INVALID_HANDLE_VALUE) return;
    std::vector<uint8_t> outputBuffer(33);
    outputBuffer[0] = 0x0;
    outputBuffer[1] = getLedCount();
    const auto& colors = target.getColors();
	for (int i = 0; i < target.getSize(); i++)
	{
		const auto& color = colors[i];
		outputBuffer[2 + 3*i] = static_cast<uint8_t>(color.blue * 0xff);
		outputBuffer[2 + 3*i + 1] = static_cast<uint8_t>(color.green * 0xff);
		outputBuffer[2 + 3*i + 2] = static_cast<uint8_t>(color.red * 0xff);
	}
    DWORD siz;
    auto a = WriteFile(writeHandle, outputBuffer.data(), outputBuffer.size(), &siz, nullptr);
    auto b = GetLastError();
}
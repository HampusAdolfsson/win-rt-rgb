#pragma once
#include "RenderOutput.h"
#include <Windows.h>
#include <string>

namespace Rendering
{
    class QmkRenderOutput : public RenderOutput
    {
    public:
        QmkRenderOutput(const std::string& hardwareId, size_t ledCount, unsigned int colorTemperature, float gamma);
		~QmkRenderOutput() override;

        void initialize() override;

		void drawImpl(const RenderTarget& target) override;
    private:
        HANDLE writeHandle;
        std::string hardwareId;
    };

}
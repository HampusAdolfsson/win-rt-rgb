#pragma once
#include "Color.h"
#include <memory>

namespace Rendering
{
    /**
    *   Specifies how to apply a mask to a render target. The mask takes a single 'opacity' parameter between 0 and 1.
    */
    class MaskingBehaviour
    {
    public:
        virtual void applyMask(RgbColor* colors, unsigned int size, float opacity) = 0;
    };

    /**
    * Applies the opacity value directly to all pixels.
    */
    class UniformMaskingBehaviour : public MaskingBehaviour
    {
    public:
        virtual void applyMask(RgbColor* colors, unsigned int size, float opacity) override;
    };

    /**
    * Applies the opacity value directly to all pixels.
    */
    class GradientMaskingBehaviour : public MaskingBehaviour
    {
    public:
        virtual void applyMask(RgbColor* colors, unsigned int size, float opacity) override;
    };
}

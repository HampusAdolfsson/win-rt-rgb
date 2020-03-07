// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved
//----------------------------------------------------------------------

Texture2D tx : register( t0 );
SamplerState samLinear : register( s0 );
#define CONTRAST 1.0f
#define BRIGHTNESS 0.0f

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD;
};

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS(PS_INPUT input) : SV_Target
{
    float4 color = tx.Sample( samLinear, input.Tex );

    // Apply contrast.
    color.rgb = ((color.rgb - 0.5f) * max(CONTRAST, 0)) + 0.5f;

    // Apply brightness.
    color.rgb += BRIGHTNESS;

    // Return final pixel color.
    color.rgb *= color.a;

    return color;
}
import ColourConverters;

cbuffer PerFrameCB
{
    SamplerState gSampler;

    int gQuantiseStep;
    float gLambdaDelta;
    float gOmegaDelta;
    float gLambdaPhi;
    float gOmegaPhi;

    uint gBlendMode;
    float gBlendStrength; 
    float3 gMinColour;
    float3 gMaxColour;
    bool gDithering;
}

texture2D gFDoGTex;
texture2D gSrcTex;

// Bayer matrices for dithering
static const int bayer2[2 * 2] = {
                0, 2,
                3, 1
            };

static const int bayer4[4 * 4] = {
                0, 8, 2, 10,
                12, 4, 14, 6,
                3, 11, 1, 9,
                15, 7, 13, 5
            };

static const int bayer8[8 * 8] = {
                0, 32, 8, 40, 2, 34, 10, 42,
                48, 16, 56, 24, 50, 18, 58, 26,  
                12, 44,  4, 36, 14, 46,  6, 38, 
                60, 28, 52, 20, 62, 30, 54, 22,  
                3, 35, 11, 43,  1, 33,  9, 41,  
                51, 19, 59, 27, 49, 17, 57, 25, 
                15, 47,  7, 39, 13, 45,  5, 37, 
                63, 31, 55, 23, 61, 29, 53, 21
            };

float GetBayer2(int x, int y) {
    return float(bayer2[(x % 2) + (y % 2) * 2]) * (1.0f / 4.0f) - 0.5f;
}

float GetBayer4(int x, int y) {
    return float(bayer4[(x % 4) + (y % 4) * 4]) * (1.0f / 16.0f) - 0.5f;
}

float GetBayer8(int x, int y) {
   return float(bayer8[(x % 8) + (y % 8) * 8]) * (1.0f / 64.0f) - 0.5f;
}

struct ShaderPSIn
{
    float2 texC : TEXCOORD;
};

float4 main(ShaderPSIn pIn) : SV_TARGET0
{
    // Colour quantisation (actually just quantises luminace but when converted back to RGB the quantise effect is revealed)
    float3 c = gSrcTex.Sample(gSampler, pIn.texC).xyz;
    float gx = 0.5f * (gSrcTex.Sample(gSampler, pIn.texC, int2(-1, 0)).x - gSrcTex.Sample(gSampler, pIn.texC, int2(1, 0)).x);
    float gy = 0.5f * (gSrcTex.Sample(gSampler, pIn.texC, int2(0, -1)).x - gSrcTex.Sample(gSampler, pIn.texC, int2(0, 1)).x);
    float grad  = sqrt(gx * gx + gy * gy);
    grad = clamp(grad, gLambdaDelta, gOmegaDelta);
    grad = (grad - gLambdaDelta) / (gOmegaDelta - gLambdaDelta);

    float phi_q = gLambdaPhi + grad * (gOmegaPhi - gLambdaPhi);
    float delta_q = 1.0f / gQuantiseStep;
    float qn = delta_q * (floor(c.x / delta_q) + 0.005f);
    float qc = qn + 0.5f * delta_q * tanh(phi_q * (c.x - qn));

    //return float4(c.x, c.y, c.z, 1.0f);
    
    // Convert LAB colour back to RGB
    float3 cRGB = lab2rgb(float3(qc * 100, (c.y - 0.5f) * 254, (c.z - 0.5f) * 254));

    //return float4(cRGB, 1.0f);

    // Blend RGB colour with FDoG output using multiplication
    float FDoG = gFDoGTex.Sample(gSampler, pIn.texC).x * gBlendStrength;
    float3 output; 
    if (gBlendMode == 0) // Fixed Colours
    {
        output = lerp(gMinColour, gMaxColour, FDoG);
    }
    else if (gBlendMode == 1) // Fixed to Output
    {
        output = lerp(gMinColour, cRGB, FDoG);
    }

    // If using dithering, use convert the FDoG output to just 2 tones using bayer dithering
    if (gDithering)
    {
        float bayerValues[3] = { 0, 0, 0 };

        uint width, height;
        gFDoGTex.GetDimensions(width, height);
        float2 dims = float2(width, height);
        int x = width * pIn.texC.x;
        int y = height * pIn.texC.y;

        bayerValues[0] = GetBayer2(x, y);
        bayerValues[1] = GetBayer4(x, y);
        bayerValues[2] = GetBayer8(x, y);

        //output += _Spread * bayerValues[_BayerLevel];
        output += 0.5f * bayerValues[2];

    }

    float3 finalColour = output * cRGB;

	return float4(finalColour, 1.0f);
}

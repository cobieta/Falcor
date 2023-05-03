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
    //float phi_q = 6.225f; // just for testing
    float delta_q = 1.0f / gQuantiseStep;
    float qn = delta_q * (floor(c.x / delta_q) + 0.005f);
    float qc = qn + 0.5f * delta_q * tanh(phi_q * (c.x - qn));

    //return float4(qc, c.y, c.z, 1.0f);

    // Convert LAB colour back to RGB
    //float3 cRGB = lab2rgb(float3(qc * 100, (c.y - 0.5f) * 254, (c.z - 0.5f) * 254));
    float3 cRGB = lab2rgb(float3(qc, c.y, c.z));

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

    float3 finalColour = output * cRGB;

	return float4(finalColour, 1.0f);
}

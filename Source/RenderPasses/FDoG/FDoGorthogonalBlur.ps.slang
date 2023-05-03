cbuffer PerFrameCB
{
    SamplerState gSampler;
    int gKernelRadius;
    float gtwoSigmaE2;
    float gtwoSigmaEK2;
    float gTau;
};

texture2D gSrcTex;
texture2D gEigenvectorTex;

struct ShaderPSIn
{
    float2 texC : TEXCOORD;
};

float4 main(ShaderPSIn pIn) : SV_TARGET0
{
    float2 t = gEigenvectorTex.Sample(gSampler, pIn.texC).xy;
    float2 n = float2(t.y, -t.x);
    float2 nabs = abs(n);
    float ds = 1.0 / ((nabs.x > nabs.y) ? nabs.x : nabs.y);

    uint width, height;
    gEigenvectorTex.GetDimensions(width, height);
    n /= float2(width, height);

    float2 gaussianColour = gSrcTex.Sample(gSampler, pIn.texC).xx; // Centre colour (is actually just Luminance)
    float2 gaussianSum = (1.0f, 1.0f); // For gaussian normalisation

    for (float d = ds; d <= gKernelRadius; d += ds)
    {
        float c1 = gSrcTex.Sample(gSampler, pIn.texC - (d * n)).x;
        float c2 = gSrcTex.Sample(gSampler, pIn.texC + (d * n)).x;

        float gaussian1 = exp((-d * d) / gtwoSigmaE2);
        float gaussian2 = exp((-d * d) / gtwoSigmaEK2);

        gaussianColour.x += (c1 + c2) * gaussian1;
        gaussianColour.y += (c1 + c2) * gaussian2;

        gaussianSum.x += 2.0f * gaussian1;
        gaussianSum.y += 2.0f * gaussian2;
    }

    gaussianColour /= gaussianSum;

    return float4(((1 + gTau) * gaussianColour.x) - (gTau * gaussianColour.y), 0.0f, 0.0f, 0.0f);
}

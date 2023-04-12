// Never put FBOs in the constant buffer!
texture2D gSrcTex;

SamplerState gSampler;

struct ShaderPSIn
{
    float2 texC : TEXCOORD;
};

float4 main(ShaderPSIn pIn) : SV_TARGET0
{
    float3 Sx = (
        -1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(-1, -1)).xyz +
        -2.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(-1, 0)).xyz +
        -1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(-1, 1)).xyz +
        1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(1, -1)).xyz +
        2.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(1, 0)).xyz +
        1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(1, 1)).xyz
    ) / 4.0f;

    float3 Sy = (
        -1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(-1, -1)).xyz +
        -2.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(0, -1)).xyz +
        -1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(1, -1)).xyz +
        1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(-1, 1)).xyz +
        2.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(0, 1)).xyz +
        1.0f * gSrcTex.Sample(gSampler, pIn.texC, int2(1, 1)).xyz
    ) / 4.0f;

    return float4(dot(Sx,Sx), dot(Sy,Sy), dot(Sx,Sy), 1.0f);
}

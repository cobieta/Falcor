// Never put FBOs in the constant buffer!
texture2D gSrcTex;

SamplerState gSampler;

struct ShaderPSIn
{
    float2 texC : TEXCOORD;
};

struct ShaderPSOUT
{
    float4 OutStructureTensor  : SV_TARGET0;
};

float3 structureTensor(float2 texCoord)
{
    float3 Sx = (
        -1.0f * gSrcTex.Sample(gSampler, texCoord, int2(-1, -1)).xyz +
        -2.0f * gSrcTex.Sample(gSampler, texCoord, int2(-1, 0)).xyz +
        -1.0f * gSrcTex.Sample(gSampler, texCoord, int2(-1, 1)).xyz +
        1.0f * gSrcTex.Sample(gSampler, texCoord, int2(1, -1)).xyz +
        2.0f * gSrcTex.Sample(gSampler, texCoord, int2(1, 0)).xyz +
        1.0f * gSrcTex.Sample(gSampler, texCoord, int2(1, 1)).xyz
        ) / 4.0f;

    float3 Sy = (
        -1.0f * gSrcTex.Sample(gSampler, texCoord, int2(-1, -1)).xyz +
        -2.0f * gSrcTex.Sample(gSampler, texCoord, int2(0, -1)).xyz +
        -1.0f * gSrcTex.Sample(gSampler, texCoord, int2(1, -1)).xyz +
        1.0f * gSrcTex.Sample(gSampler, texCoord, int2(-1, 1)).xyz +
        2.0f * gSrcTex.Sample(gSampler, texCoord, int2(0, 1)).xyz +
        1.0f * gSrcTex.Sample(gSampler, texCoord, int2(1, 1)).xyz
        ) / 4.0f;

    return float3(dot(Sx, Sx), dot(Sy, Sy), dot(Sx, Sy));
}

ShaderPSOUT main(ShaderPSIn pIn)
{
    ShaderPSOUT psOut;

    // Calculate structure Tensor
    psOut.OutStructureTensor = float4(structureTensor(pIn.texC), 1.0f);

    return psOut;
}

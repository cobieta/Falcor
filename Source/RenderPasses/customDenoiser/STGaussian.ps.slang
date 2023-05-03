import STcommonMath;

cbuffer PerFrameCB
{
    SamplerState gSampler;
    int gKernelWidth;
    Buffer<float> gWeights;
}

// Source image
texture2D gSrcTex;

struct BlurPSIn
{
    float2 texC : TEXCOORD;
};

struct ShaderPSOUT
{
    float4 OutStructureTensor  : SV_TARGET0;
    float2 OutEigenvector      : SV_TARGET1;
};

float4 blur(float2 texC)
{

    uint width, height;
    gSrcTex.GetDimensions(width, height);
    
#ifdef _HORIZONTAL_BLUR
    const float2 dir = float2(1 / (float)width, 0);
#elif defined _VERTICAL_BLUR
    const float2 dir = float2(0, 1 / (float)height);
#else
#error Please define either _HORIZONTAL_BLUR or _VERTICAL_BLUR
#endif

    //const int2 offset = -(gKernelWidth / 2) * dir;

    float4 c = float4(0,0,0,0);
    for (int i = 0; i <= gKernelWidth; ++i)
    {
        c += gSrcTex.Sample(gSampler, texC + (i*dir)) * gWeights[i];
    }

    return c;

    /* Old version
   const int2 offset = -(gKernelWidth / 2) * dir;

    float4 c = float4(0,0,0,0);
    $for(i in Range(gKernelWidth))
    {
        c += gSrcTex.Sample(gSampler, texC,offset + i*dir)*gWeights[i];
    }
    return c;
    */
}

float2 eigenvector(float3 g)
{
    /* Old version
    float lambda1 =  0.5 * (g.x + g.y + sqrt(g.x*g.x - 2.0f*g.x*g.y + g.y*g.y + 4.0f*g.z*g.z));

    float2 v = float2(lambda1 - g.x, g.z);
    float2 t = length(v) > 0.0 ? normalize(v) : float2(0.0f, 1.0f);
    */

    float2 t = st2tangent(float4(g, 1.0f));

    return t;
}

ShaderPSOUT main(BlurPSIn pIn) : SV_TARGET0
{
    ShaderPSOUT psOut;

    // Blur the structure tensor
    psOut.OutStructureTensor = blur(pIn.texC);

    // If this is the second pass, calculate the eigenvector in the direction of least change
#ifdef _EIGENVECTOR
    psOut.OutEigenvector = eigenvector(psOut.OutStructureTensor.xyz);
#else
    psOut.OutEigenvector;
#endif

    return psOut;
}

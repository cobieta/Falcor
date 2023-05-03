cbuffer PerFrameCB
{
    SamplerState gSampler;
    float gtwoSigmaD2;
    float gtwoSigmaR2;
    float gRadius;
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
#ifdef _GRADIENT
    t = float2(t.y, -t.x);
#endif

    float2 tabs = abs(t);
    float ds = 1.0f / ((tabs.x > tabs.y) ? tabs.x : tabs.y);

    float3 centreColour = gSrcTex.Sample(gSampler, pIn.texC).xyz;
    float3 sum = centreColour;
    float normalSum = 1.0f;

    uint width, height;
    gSrcTex.GetDimensions(width, height);
    float2 dims = float2(width, height);

    for (float d = ds; d <= gRadius; d += ds) {
        float2 dt = (d * t) / dims;
        float3 c1 = gSrcTex.Sample(gSampler, pIn.texC + dt).xyz;
        float3 c2 = gSrcTex.Sample(gSampler, pIn.texC - dt).xyz;

        float e1 = length(c1 - centreColour);
        float e2 = length(c2 - centreColour);
        
        float kd = exp( (-d * d) / gtwoSigmaD2 );
        float kr1 = exp( (-e1 * e1) / gtwoSigmaR2 );
        float kr2 = exp( (-e2 * e2) / gtwoSigmaR2 );
        
        sum += kd * kr1 * c1;
        sum += kd * kr2 * c2;
        normalSum += kd * (kr1 + kr2);
    }

    sum /= normalSum;

	return float4(sum, 1.0f);
}

import STcommonMath;

cbuffer PerFrameCB
{
    SamplerState gSampler;
    int gKernelRadius;
    float gtwoSigma2;

    // Thresholding parameters
#ifdef _THRESHOLD
    float gEpsilon;
    float gPhi;
    uint gThresholdMode;
    int gQuantizerStep;
    bool gInvert;
#endif
};

texture2D gSrcTex;
texture2D gEigenvectorTex;

struct ShaderPSIn
{
    float2 texC : TEXCOORD;
};

#ifdef _THRESHOLD
float threshold(float FDoGOutput)
{
    float output = FDoGOutput;
    
    if (gThresholdMode == 1) // Tanh
    {
        output = (FDoGOutput >= gEpsilon) ? 1 : 1 + tanh(gPhi * (FDoGOutput - gEpsilon));
    } 
    else if (gThresholdMode == 2) { // Basic Quantisation
        output = (FDoGOutput >= gEpsilon) ? 1 : floor(FDoGOutput * (gQuantizerStep - 1) + 0.5f) / (gQuantizerStep - 1);
    }
    else if (gThresholdMode == 3) { // Extended Quantisation
        float stepReciprocal = 1.0f / gQuantizerStep;
        output = (FDoGOutput >= gEpsilon) ? 1 : stepReciprocal * floor((pow(FDoGOutput, gPhi) - (stepReciprocal * gEpsilon / 2.0f)) / (stepReciprocal * gEpsilon) + 0.5f);
    }
    else if (gThresholdMode == 4) { // Smooth Quantisation
        float qn = floor(FDoGOutput * gQuantizerStep + 0.5f) / gQuantizerStep;
        float qs = smoothstep(-2.0f, 2.0f, gPhi * (FDoGOutput - qn)) - 0.5f;
                    
        output = qn + qs / gQuantizerStep;
    }
    // No Threshold otherwise

    // Invert colours?
    if (gInvert) {output = 1 - output;}

    output = clamp(output, 0.0f, 1.0f);
    return output;
}
#endif

float4 main(ShaderPSIn pIn) : SV_TARGET0
{
#ifdef _THRESHOLD
    float gaussianColour = gSrcTex.Sample(gSampler, pIn.texC).x; // Centre colour (is actually just Luminance hence the single component)
#else
    float3 gaussianColour = gSrcTex.Sample(gSampler, pIn.texC).xyz; // Centre colour
#endif
    float gaussianSum = 1.0f; // For gaussian normalisation

    uint width, height;
    gSrcTex.GetDimensions(width, height);
    float2 dims = float2(width, height);

    float2 t = st2tangent(gEigenvectorTex.Sample(gSampler, pIn.texC)); // Smoothed structure tensor converted to eigenvector (v0)

    float2 curVector = t / dims; // (v)
    float2 curSamplePos = pIn.texC + curVector; // (p), u = 1

#ifdef _THRESHOLD
    float curColour;
#else
    float3 curColour;
#endif
    float curGaussian;

    // Line Integral Convolution along eigenvector in positive direction
    for (int d = 1; d < gKernelRadius; ++d)
    {
#ifdef _THRESHOLD
        curColour = gSrcTex.Sample(gSampler, curSamplePos).x;
#else
        curColour = gSrcTex.Sample(gSampler, curSamplePos).xyz;
#endif
        curGaussian = exp((-d * d) / gtwoSigma2);

        gaussianColour += curColour * curGaussian;
        gaussianSum += curGaussian;

        // Calculation of the next sample position for line integral convolution using 4 eigenvector samples
        float2 k1 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos)) / dims;
        float vt = dot(curVector, k1);
        if (vt < 0) k1 = -k1;

        float2 k2 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos + 0.5f * k1)) / dims;
        vt = dot(curVector, k2);
        if (vt < 0) k2 = -k2;

        float2 k3 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos + 0.5f * k2)) / dims;
        vt = dot(curVector, k3);
        if (vt < 0) k3 = -k3;

        float2 k4 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos + k3)) / dims;
        vt = dot(curVector, k4);
        if (vt < 0) k4 = -k4;

        float2 newVector = (k1 + 2*k2 + 2*k3 + k4) / 6.0f;

        // Single sample calculation for testing
        //float2 newVector = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos)) / dims;

        float dotProduct = dot(curVector, newVector);
        if (abs(dotProduct) <= 0.0f) break; // the vectors are more than 90 degrees apart so continuing the path will likely cross a feature edge which is unwanted
        //if (dotProduct < 0) newVector = -newVector; // reverse the direction if the new vector is pointing in the opposite direction to the LIC direction

        curVector = newVector;
        curSamplePos += curVector;
    }

    // Reset sample position and reverse eigenvector
    curVector = -t / dims;
    curSamplePos = pIn.texC + curVector;

    // Line Integral Convolution along eigenvector in negative direction
    for (int d = 1; d < gKernelRadius; ++d)
    {
#ifdef _THRESHOLD
        curColour = gSrcTex.Sample(gSampler, curSamplePos).x;
#else
        curColour = gSrcTex.Sample(gSampler, curSamplePos).xyz;
#endif
        curGaussian = exp((-d * d) / gtwoSigma2);

        gaussianColour += curColour * curGaussian;
        gaussianSum += curGaussian;

        // Calculation of the next sample position for line integral convolution using 4 eigenvector samples
        float2 k1 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos)) / dims;
        float vt = dot(curVector, k1);
        if (vt < 0) k1 = -k1;

        float2 k2 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos + 0.5f * k1)) / dims;
        vt = dot(curVector, k2);
        if (vt < 0) k2 = -k2;

        float2 k3 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos + 0.5f * k2)) / dims;
        vt = dot(curVector, k3);
        if (vt < 0) k3 = -k3;

        float2 k4 = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos + k3)) / dims;
        vt = dot(curVector, k4);
        if (vt < 0) k4 = -k4;

        float2 newVector = (k1 + 2*k2 + 2*k3 + k4) / 6.0f;

        // Single sample calculation for testing
        //float2 newVector = st2tangent(gEigenvectorTex.Sample(gSampler, curSamplePos)) / dims;

        float dotProduct = dot(curVector, newVector);
        if (abs(dotProduct) <= 0.0f) break; // the vectors are more than 90 degrees apart so continuing the path will likely cross a feature edge which is unwanted
        //if (dotProduct < 0) newVector = -newVector; // reverse the direction if the new vector is pointing in the opposite direction to the LIC direction

        curVector = newVector;
        curSamplePos += curVector;
    }

    // Normalise
    gaussianColour /= gaussianSum;

    // If this is the first Line Integral Convolution then threshold the output
#ifdef _THRESHOLD
    float outputColour = threshold(gaussianColour);
    return float4(outputColour, 0.0f, 0.0f, 0.0f);
#else
    // If this is the second Line Integral Convolution then no thresholding is needed since the purpose of this render pass is antialiasing
    return float4(gaussianColour, 1.0f);
#endif
}

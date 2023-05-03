//import Utils.Color.ColorHelpers;
import ColourConverters;

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
    float  OutLuminance        : SV_TARGET1;
    float4 OutLABColour        : SV_TARGET2;
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

/* Old version
float2 checkerColour(float2 texCoord)
{
    // Old
    float3 colourRGB = gSrcTex.Sample(gSampler, texCoord).xyz;
    float3 colourYCgCo = RGBToYCgCo(sRGBToLinear(colourRGB));
    return ((texCoord.x & 1) == (texCoord.y & 1)) ? colourYCgCo.xy : colourYCgCo.xz;
    // Old

    float3 colourRGB = gSrcTex.Sample(gSampler, texCoord).xyz;


    colourRGB.x = (colourRGB.r > 0.04045) ? pow((colourRGB.r + 0.055) / 1.055, 2.4) : colourRGB.r / 12.92;
    colourRGB.y = (colourRGB.g > 0.04045) ? pow((colourRGB.g + 0.055) / 1.055, 2.4) : colourRGB.g / 12.92,
    colourRGB.z = (colourRGB.b > 0.04045) ? pow((colourRGB.b + 0.055) / 1.055, 2.4) : colourRGB.b / 12.92;
    float luminance = RGBtoXYZ_Rec709(colourRGB).y;

    luminance = (luminance > 0.008856) ? pow(luminance, 1.0 / 3.0) : (7.787 * luminance) + (16.0 / 116.0);
    return float2(((116.0 * luminance) - 16.0) / 100.0f, 0.0f); // Return luminance in the range [0,1] rather than [0,100]

    //return float2(luminance(colourRGB), 0.0f);

}
*/

float3 labConversion(float2 texCoord)
{
    float3 colourRGB = gSrcTex.Sample(gSampler, texCoord).xyz;
    float3 colourLAB = rgb2lab(colourRGB);
    // Use the commented out version if showing on screen
    return float3(colourLAB.x / 100.0f, 0.5 + 0.5 * (colourLAB.y / 127.0), 0.5 + 0.5 * (colourLAB.z / 127.0)); // Return luminance in the range [0,1] rather than [0,100]
    //return colourLAB;
}

ShaderPSOUT main(ShaderPSIn pIn)
{
    ShaderPSOUT psOut;

    // Calculate structure Tensor
    psOut.OutStructureTensor = float4(structureTensor(pIn.texC), 1.0f);

    // Calculate LAB colour buffer
    psOut.OutLABColour = float4(labConversion(pIn.texC), 1.0f);

    // Copy luminance to a separate buffer
    psOut.OutLuminance = psOut.OutLABColour.x;

    return psOut;
}

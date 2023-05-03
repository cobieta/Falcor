import ColourConverters;
import SVGFCommon;

texture2D gColourTex;
texture2D gAlbedoTex;
texture2D gEmissionTex;

struct PS_OUT
{
    float4 OutColour       : SV_TARGET0;
    float4 OutAlbedo       : SV_TARGET1;
    float4 OutEmissionTex  : SV_TARGET2;
};

PS_OUT main(FullScreenPassVsOut vsOut)
{
    const float4 posH = vsOut.posH;
    const int2 ipos = posH.xy;

    float3 colourRGB = gColourTex[ipos].rgb;
    float3 albedoRGB = gAlbedoTex[ipos].rgb;
    float3 emissionRGB = gEmissionTex[ipos].rgb;

    PS_OUT psOut;
    psOut.OutColour = float4(rgb2lab(colourRGB), 1.0f);
    psOut.OutAlbedo = float4(rgb2lab(albedoRGB), 1.0f);
    psOut.OutEmissionTex = float4(rgb2lab(emissionRGB), 1.0f);

	return psOut;
}

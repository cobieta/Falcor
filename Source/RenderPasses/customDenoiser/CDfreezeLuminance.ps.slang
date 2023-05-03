import Scene.ShadingData;
import SVGFCommon;
import ColourConverters;

cbuffer PerImageCB
{
    Texture2D   gIllumination;
    Texture2D   gHistoryLength;
    Texture2D   gFrozenIllumination;

    float       gHistoryThreshold;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    // number of temporally integrated pixels
    const float historyLength = gHistoryLength[ipos].x;

    // current illumination
    const float4 curIllumination = gIllumination[ipos];

    // frozen illumination
    float4 frozenIllumination = gFrozenIllumination[ipos];

    if ((frozenIllumination.x == 0.0f) || (historyLength < gHistoryThreshold)) // if there is no previous frozen illumination do not freeze any pixels
    {
        // Allow some frames to be accumulated first to allow the SVGF filter to work before freezing. 
        frozenIllumination = curIllumination;
    }

    return frozenIllumination;
}

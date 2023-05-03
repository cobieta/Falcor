// ***********************************************************
// Common math funstions for working with the structure tensor
//
// ***********************************************************

// Use the following to import
// import STcommonMath;

// Convert structure tensor to angle
float st2angle(float4 g) {
    return 0.5f * atan2(-2 * g.z, g.y - g.x);
}

// Get the tangent (normalized eigenvector) of the structure tensor
float2 st2tangent(float4 g) {
    float phi = st2angle(g);
    return float2(cos(phi), sin(phi));
}

// st2lfm should be float4(eigenvector, 1, 1)
// so lfm length is always 1. 

// ***********************************************************
// Converts RGB to LAB colour spaces and back again (plus XYZ)
//
// ***********************************************************

// Use the following to import
// import ColourConverters;

// Convert from SRGB to XYZ
float3 rgb2xyz( float3 c ) {
    float b = ( c.x > 0.04045f ) ? pow(( c.x + 0.055f ) / 1.055f, 2.4f ) : c.x / 12.92f;
    float g = ( c.y > 0.04045f ) ? pow(( c.y + 0.055f ) / 1.055f, 2.4f ) : c.y / 12.92f;
    float r = ( c.z > 0.04045f ) ? pow(( c.z + 0.055f ) / 1.055f, 2.4f ) : c.z / 12.92f;

    float3 XYZ = float3(100 * (0.4124f * r + 0.3576f * g + 0.1805f * b),
                        100 * (0.2126f * r + 0.7152f * g + 0.0722f * b),
                        100 * (0.0193f * r + 0.1192f * g + 0.9505f * b));

    return XYZ;
}

// Convert from SRGB to LAB
float3 rgb2lab( float3 c) {
    c = rgb2xyz(c);

    c.x /= 95.047f;
    c.y /= 100.0f;
    c.z /= 108.883f;

    float x = ( c.x > 0.008856f ) ? pow( c.x, 1.0f / 3.0f ) : ( 7.787f * c.x ) + ( 16.0f / 116.0f );
    float y = ( c.y > 0.008856f ) ? pow( c.y, 1.0f / 3.0f ) : ( 7.787f * c.y ) + ( 16.0f / 116.0f );
    float z = ( c.z > 0.008856f ) ? pow( c.z, 1.0f / 3.0f ) : ( 7.787f * c.z ) + ( 16.0f / 116.0f );

    return float3((116 * y) - 16, 500 * (x - y), 200 * (y - z));
}

// Convert from XYZ to SRGB
float3 xyz2rgb( float x, float y, float z) {
    float r = ( 3.2406f * x - 1.5372f * y - 0.4986f * z ) / 100.0f;
    float g = (-0.9689f * x + 1.8758f * y + 0.0415f * z ) / 100.0f;
    float b = ( 0.0557f * x - 0.2040f * y + 1.0570f * z ) / 100.0f;

    float3 rgb = float3(( b > 0.0031308f ) ? (( 1.055f * pow( b, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * b,
                        ( g > 0.0031308f ) ? (( 1.055f * pow( g, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * g,
                        ( r > 0.0031308f ) ? (( 1.055f * pow( r, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * r,);

    return rgb;
}

// Convert from LAB to SRGB
float3 lab2rgb( float3 c) {
    float fy = (c.x + 16.0f) / 116.0f;
    float fx = c.y / 500.0f + fy;
    float fz = fy - c.z / 200.0f;

    float3 rgb = xyz2rgb(95.047f * (( fx > 0.206897f ) ? fx * fx * fx : ( fx - 16.0f / 116.0f ) / 7.787f),
                         100.000f * (( fy > 0.206897f ) ? fy * fy * fy : ( fy - 16.0f / 116.0f ) / 7.787f),
                         108.883f * (( fz > 0.206897f ) ? fz * fz * fz : ( fz - 16.0f / 116.0f ) / 7.787f));

    return rgb;
}

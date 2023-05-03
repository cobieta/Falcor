cbuffer PerFrameCB
{
    // Shared Sampler
    SamplerState gSampler;

    // User controlled parameters
    uint gRadius;
    float gHardness;
    float gSharpness;
    float gAlpha;
    float gZeta;
    float gEta;
};

// Source image plus local orientation and anisotropy
texture2D gSrcTex;
texture2D gTensorTex;

struct ShaderPSIn
{
    float2 texC : TEXCOORD;
};

float4 main(ShaderPSIn pIn) : SV_TARGET0
{
    float4 t = gTensorTex.Sample(gSampler, pIn.texC);
    //return float4(clamp((gAlpha + t.y) / gAlpha , 0.1f, 2.0f), clamp(gAlpha / (gAlpha + t.y), 0.1f, 2.0f), 0.0f, 1.0f); //-debugv1

    float a = gRadius * clamp((gAlpha + t.y) / gAlpha , 0.1f, 2.0f);
    float b = gRadius * clamp(gAlpha / (gAlpha + t.y), 0.1f, 2.0f);

    //return float4(a, b, 0.0f, 1.0f);

    float cos_phi = cos(t.x);
    float sin_phi = sin(t.x);

    float2x2 R = {cos_phi, -sin_phi,
                  sin_phi, cos_phi};

    float2x2 S = {0.5f / a, 0.0f,
                  0.0f, 0.5f / b};

    float2x2 SR = mul(S, R); // Matrix SR maps points from the ellipse to a disc of radius 0.5.

    int max_x = int(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi));
    int max_y = int(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi));

    //return float4(max_x, max_y, 0.0f, 1.0f);
    //return float4(mul(SR, float2(max_x, max_y)), 0.0f, 1.0f); //-debugv2
    //return float4(gZeta, gEta, 0.0f, 1.0f);

    float4 m[8];
    float3 s[8];

    for (int k = 0; k < 8; ++k) {
        m[k] = 0.0f;
        s[k] = 0.0f;
    }

    //return float4(m[0].xyz, 1.0f);

    // Double loop to loop through each pixel in the filter:
    for (int j = -max_y ; j <= max_y ; ++ j) {
	    for (int i = -max_x ; i <= max_x ; ++ i) {
            
            float2 v = mul(SR, float2(i, j)); // Location to sample within the filter

            if (dot (v,v) <= 0.25f) { // Check the magnitude of the vector is <= sqrt(0.5), i.e, the pixel is inside the kernel

                
                uint width, height;
                gSrcTex.GetDimensions(width, height);
                float2 offset = float2(i / (float)width, j / (float)height);
                float3 c = gSrcTex.Sample(gSampler, pIn.texC + offset).xyz;

                // Polynomial weight calculation.
                float sum = 0;
                float w[8];
                float z, vxx, vyy;

                vxx = gZeta - gEta * v.x * v.x;
                vyy = gZeta - gEta * v.y * v.y;
                z = max(0, v.y + vxx); w[0] = z * z; sum += w[0];
                z = max(0, -v.x + vyy); w[2] = z * z; sum += w[2];
                z = max(0, -v.y + vxx); w[4] = z * z; sum += w[4];
                z = max(0, v.x + vyy); w[6] = z * z; sum += w[6];

                v = sqrt(2.0f) / 2.0f * float2(v.x - v.y, v.x + v.y);

                vxx = gZeta - gEta * v.x * v.x;
                vyy = gZeta - gEta * v.y * v.y;
                z = max(0, v.y + vxx); w[1] = z * z; sum += w[1];
                z = max(0, -v.x + vyy); w[3] = z * z; sum += w[3];
                z = max(0, -v.y + vxx); w[5] = z * z; sum += w[5];
                z = max(0, v.x + vyy); w[7] = z * z; sum += w[7];

                float g = exp(-3.125f * dot(v,v)) / sum;
                
                for (int k = 0; k < 8; ++k) {
                    float wk = w[k] * g;
                    m[k] += float4(c * wk, wk);
                    s[k] += c * c * wk;
                }

            }

        }
    }

    //return float4(m[0].xyz / m[0].w, 1.0f);

    // Choose a segment for the colour
    float4 output = 0;
    for (int k = 0; k < 8; ++k) {

        // Calculate the variance and mean for each segment
        m[k].xyz /= m[k].w;
        s[k] = abs(s[k] / m[k].w - m[k].xyz * m[k].xyz);

        //return float4(m[k].xyz /= m[k].w, 1.0f);

        // Pick the segment with the lowest variance according to the new weighting function. 
        float sigma2 = s[k].x + s[k].y + s[k].z;
        //float w = 1.0f / (1.0f + pow(255.0f * sigma2, 0.5f * gSharpness)); 
        float w = 1.0f / (1.0f + pow(gHardness * 100.0f * sigma2, 0.5f * gSharpness));

        output += float4(m[k].xyz * w, w);
    }
    
	return float4(output.xyz / output.w, 1.0f);
}

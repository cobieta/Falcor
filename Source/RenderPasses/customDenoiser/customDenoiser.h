/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"

using namespace Falcor;

class customDenoiser : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<customDenoiser>;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;

private:
    customDenoiser(const Dictionary& dict);

    Scene::SharedPtr mpScene;

    // Custom denoiser parameters
    float mHistoryThreshold = 16.0f;

    // Custom denoiser methods
    void computeFreeze(RenderContext* pRenderContext);

    // Custom denoiser passes
    FullScreenPass::SharedPtr mpCIELABcolour;
    FullScreenPass::SharedPtr mpFreezeLuminance;

    // Custom denoiser framebuffers 
    Fbo::SharedPtr mpCIELABcolourFbo;
    Fbo::SharedPtr mpFreezeFbo[2];

    // FDoG functions
    float getCoefficient(float sigma, float x);
    Buffer::SharedPtr updateKernel(float sigma);

    // FDoG passes
    void computeEigenvector(RenderContext* pRenderContext);
    void computeOABilateral(RenderContext* pRenderContext);
    void computeOrthogonalFDoG(RenderContext* pRenderContext);
    void computeLICFDoG(RenderContext* pRenderContext);

    // FDoG Filter parameters:
    // Sigma values for all the different Gaussian blur passes:
    float mSigmaC = 4.0f; // Standard Deviation of the Gaussian blur used to blur the Structure
    float mSigmaD = 3.0f; // Standard Deviation of the Gaussian weights used in the bilateral filter to weight closeness
    float mSigmaR = 4.25f; // Standard Deviation of the Gaussian weights used in the bilateral filter to weight colour similarity
    float mSigmaE = 2.0f; // Standard Deviation of the Gaussian blur used to blur across edge lines (tangent to edge lines) for the FDoG
    float mSigmaM = 7.0f; // Standard Deviation of the Gaussian blur used to blur along edge lines during the first line integral convolution
    float mSigmaA = 2.0f; // Standard Deviation of the Gaussian blur used to blur along edge lines during the second line integral convolution

    // FDoG Filter radius/precision:
    float mSigmaCPrecision = 1.0f;
    float mSigmaDRgradPrecision = 2.0f;
    float mSigmaDRtangPrecision = 2.0f;
    float mSigmaEPrecision = 1.8f;
    float mSigmaMPrecision = 1.0f;

    // Orientation Aligned Bilateral Filter Iterations
    int mNE = 1; // Number of iterations used for FDoG smoothing
    int mNA = 2; // Number of iterations used for quantization smoothing

    // Parameters controlling the visual style fo the filter.
    float mTau = 100.0f; // Controls sharpness, in particular the scaling of the two DoG blurs
    float mEpsilon = 0.8f; // Controls the threshold at which the DoG filter returns the colour white.
    float mPhi = 1.8f; // Controls the steepness angle of the threshold function
    float mK = 1.6f; // Controls how much the second DoG standard deviation is scaled up from the first.

    // Threshold parameters
    enum mThresholdModesEnum
    {
        NoThreshold = 0,
        Tanh,
        BasicQuantization,
        ExtendedQuantization,
        SmoothQuantization
    };

    const Gui::RadioButtonGroup mThresholdSelectionButtons =
    {
        { (uint32_t)NoThreshold, "No Threshold", false },
        { (uint32_t)Tanh, "Tanh", false },
        { (uint32_t)BasicQuantization, "Basic Quantization", false },
        { (uint32_t)ExtendedQuantization, "Extended Quantization", false },
        { (uint32_t)SmoothQuantization, "Smooth Quantization", false }
    };

    uint32_t mThresholdMode = NoThreshold; // Thresholding function used for non-white output of DoG
    int mQuantizerStep = 2; // Quantization step value
    bool mInvert = false; // When set to true this inverts the final output.

    // Blending parameters
    float mBlendStrength = 1.0f; // Blending strength
    float3 mMinColour = float3(0.0f, 0.0f, 0.0f); // Minimum colour to use for blending, default is black
    float3 mMaxColour = float3(1.0f, 1.0f, 1.0f); // Maximum colour to use for blending, default is white

    // Colour quantisation parameters
    int mColourQuantiseStep = 8;
    float mLambdaDelta = 0.0f;
    float mOmegaDelta = 0.2f;
    float mLambdaPhi = 0.34f;
    float mOmegaPhi = 0.106f;

    // Dithering
    bool mEnableDithering = false; // Enable dithering to be used to mimic screentones

    // FDoG Passes
    FullScreenPass::SharedPtr mpStructureTensor; // Calculate Structure Tensor
    FullScreenPass::SharedPtr mpSTHorizontalBlur; // Do a 1D Gaussian blur horizontally across the structure tensor
    FullScreenPass::SharedPtr mpEigenvector; // Do a 1D Gaussian blur vertically across the structure tensor and calculate the eigenvector in the direction of least change
    FullScreenPass::SharedPtr mpOABilateralGradient; // Do a 1D Orientation-aligned bilateral filter of the image along the eigenvector gradient
    FullScreenPass::SharedPtr mpOABilateralTangent; // Do a 1D Orientation-aligned bilateral filter of the image along the eigenvector tangent
    FullScreenPass::SharedPtr mpFDogTangentBlur; // Use the eigenvector to calculate the FDoG by blurring tangent to the eigenvector
    FullScreenPass::SharedPtr mpFDogLIC1Blur; // Use the eigenvector to calculate the second FDoG blur pass and the first Line Integral Convolution
    FullScreenPass::SharedPtr mpFinalBlend; // Blending stage that combines the FDoG output with the colour output from the bilateral filter
    FullScreenPass::SharedPtr mpFDogLIC2Blur; // Use the eigenvector to calculate the second FDoG Line Integral Convolution pass for anti-aliasing

    // Intermediate framebuffers
    Fbo::SharedPtr mpStructureTensorFbo;
    Fbo::SharedPtr mpStructureTensorBlurFbo;
    Fbo::SharedPtr mpEigenvectorFbo;
    Fbo::SharedPtr mpPastEigenvectorFbo;
    Fbo::SharedPtr mpOABilateralGradientFbo;
    Fbo::SharedPtr mpOABilateralEdgeTargetFbo;
    Fbo::SharedPtr mpOABilateralQuantTargetFbo;
    Fbo::SharedPtr mpFDogTangentBlurFbo;
    Fbo::SharedPtr mpFDogLIC1BlurFbo;
    Fbo::SharedPtr mpFinalBlendFbo;
    Fbo::SharedPtr mpFDogLIC2BlurFbo;

    // Bilinear interpolation sampler
    Sampler::SharedPtr mpSampler;

    // Buffer containing the Gaussian weights for the structure tensor blur
    Buffer::SharedPtr mSigmaCbuffer;


    // SVGF functions
    bool init(const Dictionary& dict);
    void allocateFbos(uint2 dim, RenderContext* pRenderContext);
    void clearBuffers(RenderContext* pRenderContext, const RenderData& renderData);

    void computeLinearZAndNormal(RenderContext* pRenderContext, Texture::SharedPtr pLinearZTexture,
        Texture::SharedPtr pWorldNormalTexture);
    void computeReprojection(RenderContext* pRenderContext, 
        Texture::SharedPtr pMotionVectorTexture,
        Texture::SharedPtr pPositionNormalFwidthTexture,
        Texture::SharedPtr pPrevLinearZAndNormalTexture);
    void computeFilteredMoments(RenderContext* pRenderContext);
    void computeAtrousDecomposition(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture);

    // SVGF parameters
    bool    mFilterEnabled = true;
    int32_t mFilterIterations = 4;
    int32_t mFeedbackTap = 1;
    float   mVarainceEpsilon = 1e-4f;
    float   mPhiColor = 10.0f;
    float   mPhiNormal = 128.0f;
    float   mAlpha = 0.05f;
    float   mMomentsAlpha = 0.2f;

    bool mBuffersNeedClear = false;

    // SVGF passes
    FullScreenPass::SharedPtr mpPackLinearZAndNormal;
    FullScreenPass::SharedPtr mpReprojection;
    FullScreenPass::SharedPtr mpFilterMoments;
    FullScreenPass::SharedPtr mpAtrous;
    FullScreenPass::SharedPtr mpFinalModulate;

    // Intermediate framebuffers
    Fbo::SharedPtr mpPingPongFbo[2];
    Fbo::SharedPtr mpLinearZAndNormalFbo;
    Fbo::SharedPtr mpFilteredPastFbo;
    Fbo::SharedPtr mpCurReprojFbo;
    Fbo::SharedPtr mpPrevReprojFbo;
    Fbo::SharedPtr mpFilteredIlluminationFbo;
    Fbo::SharedPtr mpFinalFbo;
};

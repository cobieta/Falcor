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
#include "FDoG.h"


namespace
{
    const char kDesc[] = "Perform Flow-Based Difference of Gaussians";

    // Input and Output buffer names
    const char kSrc[] = "source";
    const char kDst[] = "destination";

    // Names of valid entries in the parameter dictionary.
    const char kSigmaC[] = "sigmaC";
    const char kSigmaD[] = "sigmaD";
    const char kSigmaR[] = "sigmaR";
    const char kSigmaE[] = "sigmaE";
    const char kSigmaM[] = "sigmaM";
    const char kSigmaA[] = "sigmaA";

    const char kSigmaCPrecision[] = "sigmaCPrecision";
    const char kSigmaDRgradPrecision[] = "sigmaDRgradPrecision";
    const char kSigmaDRtangPrecision[] = "sigmaDRtangPrecision";
    const char kSigmaEPrecision[] = "sigmaEPrecision";
    const char kSigmaMPrecision[] = "sigmaMPrecision";

    const char kNE[] = "NE";
    const char kNA[] = "NA";

    const char kTau[] = "tau";
    const char kEpsilon[] = "epsilon";
    const char kPhi[] = "phi";
    const char kK[] = "k";

    const char kThresholdMode[] = "thresholdMode";
    const char kQuantizerStep[] = "quantizerStep";
    const char kInvert[] = "invert";

    const char kBlendingMode[] = "blendingMode";
    const char kBlendStrength[] = "blendStrength";
    const char kMinColour[] = "minColour";
    const char kMaxColour[] = "maxColour";

    const char kColourQuantiseStep[] = "colourQuantiseStep";
    const char kLambdaDelta[] = "lambdaDelta";
    const char kOmegaDelta[] = "omegaDelta";
    const char kLambdaPhi[] = "lambdaPhi";
    const char kOmegaPhi[] = "omegaPhi";

    //const char kEnableTexture[] = "enableTexture";

    // Shader source files
    const char kStructureTensorShader[] = "RenderPasses/FDoG/StrucTensColourChecker.ps.slang";
    const char kGaussianShader[] = "RenderPasses/FDoG/STGaussian.ps.slang";
    const char kOABilateralShader[] = "RenderPasses/FDoG/OABilateralFilter.ps.slang";
    const char kOrthogonalFDoGShader[] = "RenderPasses/FDoG/FDoGorthogonalBlur.ps.slang";
    const char kFDogLICShader[] = "RenderPasses/FDoG/FDoGLICBlur.ps.slang";
    const char kFinalBlendShader[] = "RenderPasses/FDoG/FinalBlend.ps.slang";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("FDoG", kDesc, FDoG::create);
}

FDoG::FDoG(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kSigmaC) mSigmaC = value;
        else if (key == kSigmaD) mSigmaD = value;
        else if (key == kSigmaR) mSigmaR = value;
        else if (key == kSigmaE) mSigmaE = value;
        else if (key == kSigmaM) mSigmaM = value;
        else if (key == kSigmaA) mSigmaA = value;
        else if (key == kSigmaCPrecision) mSigmaCPrecision = value;
        else if (key == kSigmaDRgradPrecision) mSigmaDRgradPrecision = value;
        else if (key == kSigmaDRtangPrecision) mSigmaDRtangPrecision = value;
        else if (key == kSigmaEPrecision) mSigmaEPrecision = value;
        else if (key == kSigmaMPrecision) mSigmaMPrecision = value;
        else if (key == kNE) mNE = value;
        else if (key == kNA) mNA = value;
        else if (key == kTau) mTau = value;
        else if (key == kEpsilon) mEpsilon = value;
        else if (key == kPhi) mPhi = value;
        else if (key == kK) mK = value;
        else if (key == kThresholdMode) mThresholdMode = value;
        else if (key == kQuantizerStep) mQuantizerStep = value;
        else if (key == kInvert) mInvert = value;
        else if (key == kBlendStrength) mBlendStrength = value;
        else if (key == kMinColour) mMinColour = value;
        else if (key == kMaxColour) mMaxColour = value;
        else if (key == kColourQuantiseStep) mColourQuantiseStep = value;
        else if (key == kLambdaDelta) mLambdaDelta = value;
        else if (key == kOmegaDelta) mOmegaDelta = value;
        else if (key == kLambdaPhi) mLambdaPhi = value;
        else if (key == kOmegaPhi) mOmegaPhi = value;
        else logWarning("Unknown field '" + key + "' in FDoG Filter dictionary");
    }

    mpStructureTensor = FullScreenPass::create(kStructureTensorShader);
    mpFDogTangentBlur = FullScreenPass::create(kOrthogonalFDoGShader);
    mpFinalBlend = FullScreenPass::create(kFinalBlendShader);

    // Setup two Gaussian passes, one for the horizontal pass and one for the vertical pass.
    Program::DefineList defines;
    defines.add("_HORIZONTAL_BLUR");
    mpSTHorizontalBlur = FullScreenPass::create(kGaussianShader, defines);
    defines.remove("_HORIZONTAL_BLUR");
    defines.add("_VERTICAL_BLUR");
    defines.add("_EIGENVECTOR");
    mpEigenvector = FullScreenPass::create(kGaussianShader, defines);

    // Make the programs share the shader vars
    mpEigenvector->setVars(mpSTHorizontalBlur->getVars());

    // Setup two bilateral filter passes, one for the gradient aligned and the other tangent aligned
    Program::DefineList definesOABF;
    defines.add("_GRADIENT");
    mpOABilateralGradient = FullScreenPass::create(kOABilateralShader, definesOABF);
    defines.remove("_GRADIENT");
    mpOABilateralTangent = FullScreenPass::create(kOABilateralShader, definesOABF);

    // Setup two line integral convolution passes for the FDoG.
    Program::DefineList definesLIC;
    definesLIC.add("_THRESHOLD");
    mpFDogLIC1Blur = FullScreenPass::create(kFDogLICShader, definesLIC);
    definesLIC.remove("_THRESHOLD");
    mpFDogLIC2Blur = FullScreenPass::create(kFDogLICShader, definesLIC);

    // Create and setup sampler.
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpSampler = Sampler::create(samplerDesc);
}

FDoG::SharedPtr FDoG::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new FDoG(dict));
}

std::string FDoG::getDesc() { return kDesc; }

Dictionary FDoG::getScriptingDictionary()
{
    Dictionary dict;
    dict[kSigmaC] = mSigmaC;
    dict[kSigmaD] = mSigmaD;
    dict[kSigmaR] = mSigmaR;
    dict[kSigmaE] = mSigmaE;
    dict[kSigmaM] = mSigmaM;
    dict[kSigmaA] = mSigmaA;
    dict[kSigmaCPrecision] = mSigmaCPrecision;
    dict[kSigmaDRgradPrecision] = mSigmaDRgradPrecision;
    dict[kSigmaDRtangPrecision] = mSigmaDRtangPrecision;
    dict[kSigmaEPrecision] = mSigmaEPrecision;
    dict[kSigmaMPrecision] = mSigmaMPrecision;
    dict[kNE] = mNE;
    dict[kNA] = mNA;
    dict[kTau] = mTau;
    dict[kEpsilon] = mEpsilon;
    dict[kPhi] = mPhi;
    dict[kK] = mK;
    dict[kThresholdMode] = mThresholdMode;
    dict[kQuantizerStep] = mQuantizerStep;
    dict[kInvert] = mInvert;
    dict[kBlendStrength] = mBlendStrength;
    dict[kMinColour] = mMinColour;
    dict[kMaxColour] = mMaxColour;
    dict[kColourQuantiseStep] = mColourQuantiseStep;
    dict[kLambdaDelta] = mLambdaDelta;
    dict[kOmegaDelta] = mOmegaDelta;
    dict[kLambdaPhi] = mLambdaPhi;
    dict[kOmegaPhi] = mOmegaPhi;
    //dict[kEnableTexture] = mEnableDithering;
    return dict;
}

RenderPassReflection FDoG::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addOutput(kDst, "output filtered image");
    reflector.addInput(kSrc, "input image to be filtered");
    return reflector;
}

void FDoG::compile(RenderContext* pContext, const CompileData& compileData)
{
    // Allocate the double buffer for the structure tensor and colour buffers
    Fbo::Desc tripleBufferdesc;
    tripleBufferdesc.setSampleCount(0);
    tripleBufferdesc.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float); // Structure Tensor
    tripleBufferdesc.setColorTarget(1, Falcor::ResourceFormat::R16Float);    // Luminance of LAB
    tripleBufferdesc.setColorTarget(2, Falcor::ResourceFormat::RGBA16Float); // LAB colour image
    mpStructureTensorFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, tripleBufferdesc);

    // Allocate the double buffer for the smoothed structure tensor and eigenvector
    Fbo::Desc doubleBufferdesc;
    doubleBufferdesc.setSampleCount(0);
    doubleBufferdesc.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float); // Structure Tensor
    doubleBufferdesc.setColorTarget(1, Falcor::ResourceFormat::RG16Float);   // Eigenvector
    mpEigenvectorFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, doubleBufferdesc);

    // Allocate and setup a single buffer for the Orientation Aligned Bilateral Filter FBOs
    Fbo::Desc descOABF;
    descOABF.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float);
    mpOABilateralGradientFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, descOABF);
    mpOABilateralEdgeTargetFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, descOABF);
    mpOABilateralQuantTargetFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, descOABF);

    // Allocate and setup internal Screen-size FrameBufferObject for blurred structure tensor, blending and
    // antialiasing stages with 1 RGBA16Float buffer
    Fbo::Desc desc;
    desc.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float);
    mpStructureTensorBlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpFinalBlendFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpFDogLIC2BlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);

    // These FBOs only need 1 component.
    desc.setColorTarget(0, Falcor::ResourceFormat::R16Float);
    mpFDogTangentBlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpFDogLIC1BlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);

    // Gaussian weight buffers
    mSigmaCbuffer = updateKernel(mSigmaC);

    mFBOsNeedClear = true;
}

void FDoG::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // Setup the external resources
    Texture::SharedPtr pSrc = renderData[kSrc]->asTexture();
    Texture::SharedPtr pDst = renderData[kDst]->asTexture();

    // If the screen size changes, the render graph will be recompiled so clear the old FBOs. 
    if (mFBOsNeedClear)
    {
        clearBuffers(pRenderContext);
        mFBOsNeedClear = false;
    }

    assert(mpFDogLIC2BlurFbo && mpFDogLIC2BlurFbo->getWidth() == pSrc->getWidth() && mpFDogLIC2BlurFbo->getHeight() == pSrc->getHeight());

    // Calculate structure tensor, smoothed eigenvector and new colour space
    computeEigenvector(pRenderContext, pSrc);

    // Perform bilateral filtering on the LAB source image
    computeOABilateral(pRenderContext);

    // FDoG Blur orthogonal to the Eigenvector tangent
    computeOrthogonalFDoG(pRenderContext);
    
    // FDoG Blur along Eigenvector tangent using Line Integral Convolution and blend the colour and FDoG outputs together
    computeLICFDoG(pRenderContext, pSrc);

    // Blit into the output texture.
    pRenderContext->blit(mpFDogLIC2BlurFbo->getColorTexture(0)->getSRV(), pDst->getRTV());
    //pRenderContext->blit(mpEigenvectorFbo->getColorTexture(1)->getSRV(), pDst->getRTV());
    
}

void FDoG::clearBuffers(RenderContext* pRenderContext)
{
    pRenderContext->clearFbo(mpStructureTensorFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpStructureTensorBlurFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpEigenvectorFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpOABilateralGradientFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpOABilateralEdgeTargetFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpOABilateralQuantTargetFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpFDogTangentBlurFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpFDogLIC1BlurFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpFinalBlendFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpFDogLIC2BlurFbo.get(), float4(0), 1.0f, 0);
}

void FDoG::computeEigenvector(RenderContext* pRenderContext, Texture::SharedPtr source)
{
    // Structure Tensor pass
    mpStructureTensor["gSampler"] = mpSampler;
    mpStructureTensor["gSrcTex"] = source;
    mpStructureTensor->execute(pRenderContext, mpStructureTensorFbo);

    // Horizontal Gaussian pass of the Structure Tensor
    auto perFrameCBST = mpSTHorizontalBlur["PerFrameCB"];
    perFrameCBST["gSampler"] = mpSampler;
    perFrameCBST["gKernelWidth"] = 1 + 2 * ((int)std::max(1.0f, ceilf(mSigmaC * mSigmaCPrecision)));
    perFrameCBST["gWeights"] = mSigmaCbuffer;
    mpSTHorizontalBlur["gSrcTex"] = mpStructureTensorFbo->getColorTexture(0);
    mpSTHorizontalBlur->execute(pRenderContext, mpStructureTensorBlurFbo);

    // Vertical Gaussian pass of the Structure Tensor and Eigenvector calculation
    mpEigenvector["gSrcTex"] = mpStructureTensorBlurFbo->getColorTexture(0);
    mpEigenvector->execute(pRenderContext, mpEigenvectorFbo);
}

void FDoG::computeOABilateral(RenderContext* pRenderContext)
{
    // Orientation-aligned bilateral filter pass along the eigenvector gradients
    auto perFrameOABF = mpOABilateralGradient["PerFrameCB"];
    perFrameOABF["gSampler"] = mpSampler;
    perFrameOABF["gtwoSigmaD2"] = 2.0f * mSigmaD * mSigmaD;
    perFrameOABF["gtwoSigmaR2"] = 2.0f * mSigmaR * mSigmaR;
    perFrameOABF["gRadius"] = ceilf(mSigmaD * mSigmaDRgradPrecision);
    mpOABilateralGradient["gEigenvectorTex"] = mpEigenvectorFbo->getColorTexture(1); // Eigenvectors
    mpOABilateralGradient["gSrcTex"] = mpStructureTensorFbo->getColorTexture(2); // LAB colour image

    // Orientation-aligned bilateral filter pass along the eigenvector tangents
    auto perFrameOABFT = mpOABilateralTangent["PerFrameCB"];
    perFrameOABFT["gSampler"] = mpSampler;
    perFrameOABFT["gtwoSigmaD2"] = 2.0f * mSigmaD * mSigmaD;
    perFrameOABFT["gtwoSigmaR2"] = 2.0f * mSigmaR * mSigmaR;
    perFrameOABFT["gRadius"] = ceilf(mSigmaD * mSigmaDRtangPrecision);
    mpOABilateralTangent["gEigenvectorTex"] = mpEigenvectorFbo->getColorTexture(1); // Eigenvectors
    mpOABilateralTangent["gSrcTex"] = mpOABilateralGradientFbo->getColorTexture(0);

    // mNA >= mNE always
    if (mNA <= mNE)
    {
        mNA = mNE + 1;
    }

    for (int i = 0; i < mNE; i++) {
        mpOABilateralGradient->execute(pRenderContext, mpOABilateralGradientFbo);
        mpOABilateralTangent->execute(pRenderContext, mpOABilateralEdgeTargetFbo);
        mpOABilateralGradient["gSrcTex"] = mpOABilateralEdgeTargetFbo->getColorTexture(0); // Feed colour buffer in for loop
    }

    // Start rendering into EdgeTargetFbo and once the target loops are reached, render into QuantTargetFbo:
    for (int i = mNE; i < mNA; i++) {
        mpOABilateralGradient->execute(pRenderContext, mpOABilateralGradientFbo);
        mpOABilateralTangent->execute(pRenderContext, mpOABilateralQuantTargetFbo);
        mpOABilateralGradient["gSrcTex"] = mpOABilateralQuantTargetFbo->getColorTexture(0);
    }
}

void FDoG::computeOrthogonalFDoG(RenderContext* pRenderContext)
{
    // FDoG Blur orthogonal to the Eigenvector tangent
    auto perFrameCBOE = mpFDogTangentBlur["PerFrameCB"];
    perFrameCBOE["gSampler"] = mpSampler;
    perFrameCBOE["gKernelRadius"] = (int)std::max(1.0f, ceilf(mSigmaE * mSigmaEPrecision * mK));
    perFrameCBOE["gtwoSigmaE2"] = 2.0f * mSigmaE * mSigmaE;
    perFrameCBOE["gtwoSigmaEK2"] = 2.0f * mSigmaE * mSigmaE * mK * mK;
    perFrameCBOE["gTau"] = mTau;
    mpFDogTangentBlur["gEigenvectorTex"] = mpEigenvectorFbo->getColorTexture(1); // Eigenvectors
    mpFDogTangentBlur["gSrcTex"] = mpOABilateralEdgeTargetFbo->getColorTexture(0); // Bilaterally smoothed LAB colour image 
    mpFDogTangentBlur->execute(pRenderContext, mpFDogTangentBlurFbo);
}

void FDoG::computeLICFDoG(RenderContext* pRenderContext, Texture::SharedPtr source)
{
    // FDoG First Line Integral Convolution blur
    auto perFrameCBLIC = mpFDogLIC1Blur["PerFrameCB"];
    perFrameCBLIC["gSampler"] = mpSampler;
    perFrameCBLIC["gKernelRadius"] = (int)std::max(1.0f, ceilf(mSigmaM * mSigmaMPrecision));
    perFrameCBLIC["gtwoSigma2"] = 2.0f * mSigmaM * mSigmaM;
    // Threshold function settings
    perFrameCBLIC["gEpsilon"] = mEpsilon;
    perFrameCBLIC["gPhi"] = mPhi;
    perFrameCBLIC["gThresholdMode"] = mThresholdMode;
    perFrameCBLIC["gQuantizerStep"] = mQuantizerStep;
    perFrameCBLIC["gInvert"] = mInvert;
    mpFDogLIC1Blur["gEigenvectorTex"] = mpEigenvectorFbo->getColorTexture(0); // smoothed structure tensor (not calculated eigenvectors)
    mpFDogLIC1Blur["gSrcTex"] = mpFDogTangentBlurFbo->getColorTexture(0); // FDoG luminance after orthogonal blur
    mpFDogLIC1Blur->execute(pRenderContext, mpFDogLIC1BlurFbo);

    // Blend the FDoG output with the colour output
    auto perframeCBblend = mpFinalBlend["PerFrameCB"];
    perframeCBblend["gSampler"] = mpSampler;
    // Colour quantisation settings
    perframeCBblend["gQuantiseStep"] = mColourQuantiseStep;
    perframeCBblend["gLambdaDelta"] = mLambdaDelta;
    perframeCBblend["gOmegaDelta"] = mOmegaDelta;
    perframeCBblend["gLambdaPhi"] = mLambdaPhi;
    perframeCBblend["gOmegaPhi"] = mOmegaPhi;
    // FDoG blend settings
    perframeCBblend["gBlendStrength"] = mBlendStrength;
    perframeCBblend["gMinColour"] = mMinColour;
    perframeCBblend["gMaxColour"] = mMaxColour;
    perframeCBblend["gDithering"] = mEnableDithering;
    mpFinalBlend["gFDoGTex"] = mpFDogLIC1BlurFbo->getColorTexture(0); // FDoG features
    mpFinalBlend["gSrcTex"] = mpOABilateralQuantTargetFbo->getColorTexture(0); // LAB bilaterally filtered colour
    mpFinalBlend->execute(pRenderContext, mpFinalBlendFbo);

    // FDoG Second Line Integral Convolution blur (Anti-Aliasing Pass)
    auto perFrameCBLIC2 = mpFDogLIC2Blur["PerFrameCB"];
    perFrameCBLIC2["gSampler"] = mpSampler;
    perFrameCBLIC2["gKernelRadius"] = (int)std::max(1.0f, ceilf(mSigmaA));
    perFrameCBLIC2["gtwoSigma2"] = 2.0f * mSigmaA * mSigmaA;
    mpFDogLIC2Blur["gEigenvectorTex"] = mpEigenvectorFbo->getColorTexture(0);
    mpFDogLIC2Blur["gSrcTex"] = mpFinalBlendFbo->getColorTexture(0);
    mpFDogLIC2Blur->execute(pRenderContext, mpFDogLIC2BlurFbo);
}

float FDoG::getCoefficient(float sigma, float x)
{
    float sigmaSquared = sigma * sigma;
    float p = -(x * x) / (2 * sigmaSquared);
    float e = std::exp(p);

    float a = 2 * (float)M_PI * sigmaSquared;
    return e / a;
}

Buffer::SharedPtr FDoG::updateKernel(float sigma)
{
    uint32_t kernelRadius = (uint32_t)std::max(1.0f, floor(sigma * 2.45f));
    float sum = 0;
    std::vector<float> weights(kernelRadius + 1);
    for (uint32_t i = 0; i <= kernelRadius; i++)
    {
        weights[i] = getCoefficient(sigma, (float)i);
        sum += (i == 0) ? weights[i] : 2 * weights[i];
    }

    Buffer::SharedPtr pBuf = Buffer::createTyped<float>((kernelRadius*2)+1, Resource::BindFlags::ShaderResource);

    for (uint32_t i = 0; i <= kernelRadius; i++)
    {
        float w = weights[i] / sum;
        pBuf->setElement(kernelRadius + i, w);
        pBuf->setElement(kernelRadius - i, w);
    }

    return pBuf;
}

void FDoG::renderUI(Gui::Widgets& widget)
{
    int dirty = 0;

    widget.text("");
    widget.text("Sigma values for the different standard");
    widget.text("   deviations of the Gaussian blur passes");
    widget.text("Structure Tensor blur:");
    if (widget.var("Sigma C", mSigmaC, 0.1f, 5.0f, 0.1f)) {
        mSigmaCbuffer = updateKernel(mSigmaC);
    }
    //dirty |= (int)widget.var("Sigma C", mSigmaC, 0.0f, 5.0f, 0.1f);
    widget.text("Centre closeness bilateral filter weight:");
    dirty |= (int)widget.var("Sigma D", mSigmaD, 0.0f, 20.0f, 0.1f);
    widget.text("Colour similarity bilateral filter weight:");
    dirty |= (int)widget.var("Sigma R", mSigmaR, 0.0f, 100.0f, 0.1f);
    widget.text("Edge line tangent blur:");
    dirty |= (int)widget.var("Sigma E", mSigmaE, 0.1f, 7.0f, 0.1f);
    widget.text("First line integral convolution blur:");
    dirty |= (int)widget.var("Sigma M", mSigmaM, 0.1f, 20.0f, 0.1f);
    widget.text("Second line integral convolution blur:");
    dirty |= (int)widget.var("Sigma A", mSigmaA, 0.1f, 10.0f, 0.1f);

    widget.text("");
    widget.text("Filter precision/radius");
    widget.text("Structure Tensor blur precision:");
    dirty |= (int)widget.var("Sigma C Precision", mSigmaCPrecision, 1.0f, 10.0f, 0.5f);
    widget.text("Bilateral filter Gradient Precision:");
    dirty |= (int)widget.var("Gradient Precision", mSigmaDRgradPrecision, 1.0f, 10.0f, 0.5f);
    widget.text("Bilateral filter Tangent Precision:");
    dirty |= (int)widget.var("Tangent Precision", mSigmaDRtangPrecision, 1.0f, 10.0f, 0.5f);
    widget.text("Edge line tangent blur Precision:");
    dirty |= (int)widget.var("Sigma E Precision", mSigmaEPrecision, 1.0f, 5.0f, 0.1f);
    widget.text("First line integral convolution blur Precision:");
    dirty |= (int)widget.var("Sigma M Precision", mSigmaMPrecision, 1.0f, 5.0f, 0.1f);

    widget.text("");
    widget.text("Parameters controlling the bilateral filter iterations");
    widget.text("Iterations needed for FDoG:");
    dirty |= (int)widget.var("NE", mNE, 1, 10, 1);
    widget.text("Iterations needed for colour quantisation:");
    dirty |= (int)widget.var("NA", mNA, 2, 11, 1);

    widget.text("");
    widget.text("Parameters controlling the visual style");
    widget.text("Sharpness:");
    dirty |= (int)widget.var("Tau", mTau, 1.0f, 150.0f, 1.0f);
    widget.text("White Threshold:");
    dirty |= (int)widget.var("Epsilon", mEpsilon, 0.0f, 1.0f, 0.01f);
    widget.text("Threshold sensitivity:");
    dirty |= (int)widget.var("Phi", mPhi, 0.0f, 50.0f, 0.05f);
    widget.text("Second Gausssian scaling:");
    dirty |= (int)widget.var("K", mK, 0.1f, 10.0f, 0.1f);

    widget.text("");
    widget.text("Threshold parameters");
    widget.text("Threshold Mode:");
    dirty |= (int)widget.radioButtons(mThresholdSelectionButtons, mThresholdMode);
    widget.text("Quantization Step Value");
    dirty |= (int)widget.var("Quantizer Step", mQuantizerStep, 1, 16, 1);
    widget.text("Invert the output of the filter");
    dirty |= (int)widget.checkbox(mInvert ? "Output Inverted" : "Output Normal", mInvert);

    widget.text("");
    widget.text("Blending parameters");
    widget.text("Blending Strength:");
    dirty |= (int)widget.var("Strength", mBlendStrength, 0.0f, 2.0f, 0.1f);
    widget.text("Minimum Blend Colour");
    dirty |= (int)widget.rgbColor("Minimum Color", mMinColour);
    widget.text("Maximum Blend Colour");
    dirty |= (int)widget.rgbColor("Maximum Color", mMaxColour);
    widget.text("Colour Quantization Step Value");
    dirty |= (int)widget.var("Quantizer Step", mColourQuantiseStep, 1, 16, 1);
    widget.text("Minimum Quantisation Sharpness:");
    dirty |= (int)widget.var("Lambda Delta", mLambdaDelta, 0.0f, 1.0f, 0.01f);
    widget.text("Maximum Quantisation Sharpness:");
    dirty |= (int)widget.var("Omega Delta", mOmegaDelta, 0.0f, 1.0f, 0.01f);
    widget.text("Minimum Quantisation Gradient:");
    dirty |= (int)widget.var("Lambda Phi", mLambdaPhi, 0.0f, 1.0f, 0.01f);
    widget.text("Maximum Quantisation Gradient:");
    dirty |= (int)widget.var("Omega Phi", mOmegaPhi, 0.0f, 1.0f, 0.01f);
    
    //widget.text("Optional Texture parameters");
    //dirty |= (int)widget.checkbox(mEnableDithering ? "Enable Texture" : "Disable Texture", mEnableDithering);

    if (dirty) {
        mFBOsNeedClear = true;
    }
}

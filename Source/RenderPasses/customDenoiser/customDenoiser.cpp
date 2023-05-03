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
#include "customDenoiser.h"

namespace
{
    const char kDesc[] = "Custom real-time ray tracer denoiser";

    // Custom denoiser user parameters
    const char kHistoryThreshold[] = "HistoryThreshold";

    // Names of valid entries in the parameter dictionary.
    const char kEnabled[] = "Enabled";
    const char kIterations[] = "Iterations";
    const char kFeedbackTap[] = "FeedbackTap";
    const char kVarianceEpsilon[] = "VarianceEpsilon";
    const char kPhiColor[] = "PhiColor";
    const char kPhiNormal[] = "PhiNormal";
    const char kAlpha[] = "Alpha";
    const char kMomentsAlpha[] = "MomentsAlpha";

    // Input buffer names
    const char kInputBufferAlbedo[] = "Albedo";
    const char kInputBufferColor[] = "Color";
    const char kInputBufferEmission[] = "Emission";
    const char kInputBufferWorldPosition[] = "WorldPosition";
    const char kInputBufferWorldNormal[] = "WorldNormal";
    const char kInputBufferPosNormalFwidth[] = "PositionNormalFwidth";
    const char kInputBufferLinearZ[] = "LinearZ";
    const char kInputBufferMotionVector[] = "MotionVec";

    // Internal buffer names
    const char kInternalBufferPreviousLinearZAndNormal[] = "Previous Linear Z and Packed Normal";
    const char kInternalBufferPreviousLighting[] = "Previous Lighting";
    const char kInternalBufferPreviousMoments[] = "Previous Moments";

    // Output buffer name
    const char kOutputBufferFilteredImage[] = "Filtered image";

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

    // Shader source files
    const char kStructureTensorShader[] = "RenderPasses/customDenoiser/StrucTensColourChecker.ps.slang";
    const char kGaussianShader[] = "RenderPasses/customDenoiser/STGaussian.ps.slang";
    const char kOABilateralShader[] = "RenderPasses/customDenoiser/OABilateralFilter.ps.slang";
    const char kOrthogonalFDoGShader[] = "RenderPasses/customDenoiser/FDoGorthogonalBlur.ps.slang";
    const char kFDogLICShader[] = "RenderPasses/customDenoiser/FDoGLICBlur.ps.slang";
    const char kFinalBlendShader[] = "RenderPasses/customDenoiser/FinalBlend.ps.slang";

    const char kPackLinearZAndNormalShader[] = "RenderPasses/customDenoiser/SVGFPackLinearZAndNormal.ps.slang";
    const char kReprojectShader[] = "RenderPasses/customDenoiser/SVGFReproject.ps.slang";
    const char kAtrousShader[] = "RenderPasses/customDenoiser/SVGFAtrous.ps.slang";
    const char kFilterMomentShader[] = "RenderPasses/customDenoiser/SVGFFilterMoments.ps.slang";
    const char kFinalModulateShader[] = "RenderPasses/customDenoiser/SVGFFinalModulate.ps.slang";

    const char kRGB2LAB[] = "RenderPasses/customDenoiser/CDrgb2lab.ps.slang";
    const char kFreezeLuminance[] = "RenderPasses/customDenoiser/CDfreezeLuminance.ps.slang";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerClass("customDenoiser", kDesc, customDenoiser::create);
}

customDenoiser::SharedPtr customDenoiser::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new customDenoiser(dict));
}

std::string customDenoiser::getDesc() { return kDesc; }

void customDenoiser::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;

    // Reset buffers when the scene changes.
    mBuffersNeedClear = true;
}

customDenoiser::customDenoiser(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kEnabled) mFilterEnabled = value;
        else if (key == kHistoryThreshold) mHistoryThreshold = value;
        else if (key == kIterations) mFilterIterations = value;
        else if (key == kFeedbackTap) mFeedbackTap = value;
        else if (key == kVarianceEpsilon) mVarainceEpsilon = value;
        else if (key == kPhiColor) mPhiColor = value;
        else if (key == kPhiNormal) mPhiNormal = value;
        else if (key == kAlpha) mAlpha = value;
        else if (key == kMomentsAlpha) mMomentsAlpha = value;
        else if (key == kSigmaC) mSigmaC = value;
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
        else logWarning("Unknown field '" + key + "' in SVGFPass dictionary");
    }

    // custom denoiser passes
    mpCIELABcolour = FullScreenPass::create(kRGB2LAB);
    mpFreezeLuminance = FullScreenPass::create(kFreezeLuminance);


    // SVGF passes
    mpPackLinearZAndNormal = FullScreenPass::create(kPackLinearZAndNormalShader);
    mpReprojection = FullScreenPass::create(kReprojectShader);
    mpAtrous = FullScreenPass::create(kAtrousShader);
    mpFilterMoments = FullScreenPass::create(kFilterMomentShader);
    mpFinalModulate = FullScreenPass::create(kFinalModulateShader);
    assert(mpPackLinearZAndNormal && mpReprojection && mpAtrous && mpFilterMoments && mpFinalModulate);

    // FDoG passes
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

Dictionary customDenoiser::getScriptingDictionary()
{
    Dictionary dict;
    dict[kHistoryThreshold] = mHistoryThreshold;

    dict[kEnabled] = mFilterEnabled;
    dict[kIterations] = mFilterIterations;
    dict[kFeedbackTap] = mFeedbackTap;
    dict[kVarianceEpsilon] = mVarainceEpsilon;
    dict[kPhiColor] = mPhiColor;
    dict[kPhiNormal] = mPhiNormal;
    dict[kAlpha] = mAlpha;
    dict[kMomentsAlpha] = mMomentsAlpha;

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

    return dict;
}

/*
Reproject:
  - takes: motion, color, prevLighting, prevMoments, linearZ, prevLinearZ, historyLen
    returns: illumination, moments, historyLength
Variance/filter moments:
  - takes: illumination, moments, history length, normal+depth
  - returns: filtered illumination+variance (to ping pong fbo)
a-trous:
  - takes: albedo, filtered illumination+variance, normal+depth, history length
  - returns: final color
*/

RenderPassReflection customDenoiser::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput(kInputBufferAlbedo, "Albedo");
    reflector.addInput(kInputBufferColor, "Color");
    reflector.addInput(kInputBufferEmission, "Emission");
    reflector.addInput(kInputBufferWorldPosition, "World Position");
    reflector.addInput(kInputBufferWorldNormal, "World Normal");
    reflector.addInput(kInputBufferPosNormalFwidth, "PositionNormalFwidth");
    reflector.addInput(kInputBufferLinearZ, "LinearZ");
    reflector.addInput(kInputBufferMotionVector, "Motion vectors");

    reflector.addInternal(kInternalBufferPreviousLinearZAndNormal, "Previous Linear Z and Packed Normal")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource)
        ;
    reflector.addOutput(kOutputBufferFilteredImage, "Filtered image").format(ResourceFormat::RGBA16Float);

    return reflector;
}

void customDenoiser::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    // Allocate a triple buffer for CIELAB colour, albedo and emission:
    Fbo::Desc tripleCIELABBufferdesc;
    tripleCIELABBufferdesc.setSampleCount(0);
    tripleCIELABBufferdesc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float); // Colour
    tripleCIELABBufferdesc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float); // Albedo
    tripleCIELABBufferdesc.setColorTarget(2, Falcor::ResourceFormat::RGBA32Float); // Emission
    mpCIELABcolourFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, tripleCIELABBufferdesc);

    allocateFbos(compileData.defaultTexDims, pRenderContext);

    // Allocate the double buffer for the smoothed structure tensor and eigenvector
    Fbo::Desc doubleBufferdesc;
    doubleBufferdesc.setSampleCount(0);
    doubleBufferdesc.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float); // Structure Tensor
    doubleBufferdesc.setColorTarget(1, Falcor::ResourceFormat::RG16Float);   // Eigenvector
    mpEigenvectorFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, doubleBufferdesc);
    mpPastEigenvectorFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, doubleBufferdesc);

    // Allocate and setup a single buffer for the Orientation Aligned Bilateral Filter FBOs
    Fbo::Desc descOABF;
    descOABF.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float);
    mpOABilateralGradientFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, descOABF);
    mpOABilateralEdgeTargetFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, descOABF);
    mpOABilateralQuantTargetFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, descOABF);

    // Allocate and setup internal Screen-size FrameBufferObject for blurred/unblurred structure tensor, blending and
    // antialiasing stages with 1 RGBA16Float buffer
    Fbo::Desc desc;
    desc.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float);
    mpStructureTensorBlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpFinalBlendFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpFDogLIC2BlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpStructureTensorFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);

    // These FBOs only need 1 component.
    desc.setColorTarget(0, Falcor::ResourceFormat::R16Float);
    mpFDogTangentBlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpFDogLIC1BlurFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);

    // Gaussian weight buffers
    mSigmaCbuffer = updateKernel(mSigmaC);

    mBuffersNeedClear = true;
}

void customDenoiser::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    Texture::SharedPtr pAlbedoTexture = renderData[kInputBufferAlbedo]->asTexture();
    Texture::SharedPtr pColorTexture = renderData[kInputBufferColor]->asTexture();
    Texture::SharedPtr pEmissionTexture = renderData[kInputBufferEmission]->asTexture();
    Texture::SharedPtr pWorldPositionTexture = renderData[kInputBufferWorldPosition]->asTexture();
    Texture::SharedPtr pWorldNormalTexture = renderData[kInputBufferWorldNormal]->asTexture();
    Texture::SharedPtr pPosNormalFwidthTexture = renderData[kInputBufferPosNormalFwidth]->asTexture();
    Texture::SharedPtr pLinearZTexture = renderData[kInputBufferLinearZ]->asTexture();
    Texture::SharedPtr pMotionVectorTexture = renderData[kInputBufferMotionVector]->asTexture();

    Texture::SharedPtr pOutputTexture = renderData[kOutputBufferFilteredImage]->asTexture();

    assert(mpPingPongFbo[0] &&
        mpPingPongFbo[0]->getWidth() == pAlbedoTexture->getWidth() &&
        mpPingPongFbo[0]->getHeight() == pAlbedoTexture->getHeight());

    if (mBuffersNeedClear)
    {
        clearBuffers(pRenderContext, renderData);
        mBuffersNeedClear = false;
    }

    if (mFilterEnabled)
    {
        // Grab linear z and its derivative and also pack the normal into
        // the last two channels of the mpLinearZAndNormalFbo.
        computeLinearZAndNormal(pRenderContext, pLinearZTexture, pWorldNormalTexture);

        // Convert RGB to CIELAB
        mpCIELABcolour["gColourTex"] = pColorTexture;
        mpCIELABcolour["gAlbedoTex"] = pAlbedoTexture;
        mpCIELABcolour["gEmissionTex"] = pEmissionTexture;
        mpCIELABcolour->execute(pRenderContext, mpCIELABcolourFbo);

        // Demodulate input color & albedo to get illumination and lerp in
        // reprojected filtered illumination from the previous frame.
        // Stores the result as well as initial moments and an updated
        // per-pixel history length in mpCurReprojFbo.
        Texture::SharedPtr pPrevLinearZAndNormalTexture =
            renderData[kInternalBufferPreviousLinearZAndNormal]->asTexture();
        computeReprojection(pRenderContext, pMotionVectorTexture, pPosNormalFwidthTexture, pPrevLinearZAndNormalTexture);

        // Do a first cross-bilateral filtering of the illumination and
        // estimate its variance, storing the result into a float4 in
        // mpPingPongFbo[0].  Takes mpCurReprojFbo as input.
        // Only filters if there is less than 4 frames of illumination history for a pixel!
        computeFilteredMoments(pRenderContext);

        // Filter illumination from mpCurReprojFbo[0], storing the result
        // in mpPingPongFbo[0].  Along the way (or at the end, depending on
        // the value of mFeedbackTap), save the filtered illumination for
        // next time into mpFilteredPastFbo.
        computeAtrousDecomposition(pRenderContext, pAlbedoTexture);

        // Compute albedo * filtered illumination and add emission back in.
        // Also freeze the filter output.
        //computeFreeze(pRenderContext, pMotionVectorTexture);

        // Check if the camera moved to reset the frozen buffer
        if (mpScene) // If there is a scene
        {
            auto sceneUpdates = mpScene->getUpdates();
            if ((sceneUpdates & ~Scene::UpdateFlags::CameraPropertiesChanged) != Scene::UpdateFlags::None)
            {
                // Clear the old frozen buffer to prevent ghosting
                pRenderContext->clearFbo(mpFreezeFbo[0].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
            }
            if (is_set(sceneUpdates, Scene::UpdateFlags::CameraPropertiesChanged))
            {
                auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
                auto cameraChanges = mpScene->getCamera()->getChanges();
                if ((cameraChanges & ~excluded) != Camera::Changes::None) {
                    // Clear the old frozen buffer to prevent ghosting
                    pRenderContext->clearFbo(mpFreezeFbo[0].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
                }
            }
        }

        // Freeze the filter output.
        computeFreeze(pRenderContext);

        // Compute albedo * filtered illumination and add emission back in.
        auto perImageCB = mpFinalModulate["PerImageCB"];
        perImageCB["gAlbedo"] = mpCIELABcolourFbo->getColorTexture(1);
        perImageCB["gEmission"] = mpCIELABcolourFbo->getColorTexture(2);
        perImageCB["gIllumination"] = mpFreezeFbo[1]->getColorTexture(0);
        mpFinalModulate->execute(pRenderContext, mpFinalFbo);

        // Calculate smoothed structure tensor and eigenvector
        computeEigenvector(pRenderContext);

        // Perform bilateral filtering on the LAB source image
        computeOABilateral(pRenderContext);

        // FDoG Blur orthogonal to the Eigenvector tangent
        computeOrthogonalFDoG(pRenderContext);

        // FDoG Blur along Eigenvector tangent using Line Integral Convolution and blend the colour and FDoG outputs together
        computeLICFDoG(pRenderContext);

        // Blit into the output texture.
        //pRenderContext->blit(mpFinalBlendFbo->getColorTexture(0)->getSRV(), pOutputTexture->getRTV());
        //pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), pOutputTexture->getRTV());
        //pRenderContext->blit(mpPingPongFbo[0]->getColorTexture(0)->getSRV(), pOutputTexture->getRTV());
        //pRenderContext->blit(mpEigenvectorFbo->getColorTexture(1)->getSRV(), pOutputTexture->getRTV());
        pRenderContext->blit(mpFDogLIC2BlurFbo->getColorTexture(0)->getSRV(), pOutputTexture->getRTV());

        // Swap resources so we're ready for next frame.
        std::swap(mpCurReprojFbo, mpPrevReprojFbo);
        pRenderContext->blit(mpLinearZAndNormalFbo->getColorTexture(0)->getSRV(),
            pPrevLinearZAndNormalTexture->getRTV());
    }
    else
    {
        pRenderContext->blit(pColorTexture->getSRV(), pOutputTexture->getRTV());
    }
}

void customDenoiser::allocateFbos(uint2 dim, RenderContext* pRenderContext)
{
    {
        // Screen-size FBOs with 3 MRTs: one that is RGBA32F, one that is
        // RG32F for the luminance moments, and one that is R16F.
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float); // illumination + variance
        desc.setColorTarget(1, Falcor::ResourceFormat::RG32Float);   // moments
        desc.setColorTarget(2, Falcor::ResourceFormat::R16Float);    // history length
        mpCurReprojFbo = Fbo::create2D(dim.x, dim.y, desc);
        mpPrevReprojFbo = Fbo::create2D(dim.x, dim.y, desc);
    }

    {
        // Screen-size RGBA32F buffer for linear Z, derivative, and packed normal
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpLinearZAndNormalFbo = Fbo::create2D(dim.x, dim.y, desc);
    }

    {
        // Screen-size FBOs with 1 RGBA32F buffer for filtered illumination + variance
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpPingPongFbo[0] = Fbo::create2D(dim.x, dim.y, desc);
        mpPingPongFbo[1] = Fbo::create2D(dim.x, dim.y, desc);
        mpFilteredPastFbo = Fbo::create2D(dim.x, dim.y, desc);
        mpFreezeFbo[0] = Fbo::create2D(dim.x, dim.y, desc);
        mpFreezeFbo[1] = Fbo::create2D(dim.x, dim.y, desc);
    }

    {
        // Screen-size FBO with 1 RGBA32F buffer for the final buffer
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpFinalFbo = Fbo::create2D(dim.x, dim.y, desc);
    }

    mBuffersNeedClear = true;
}

void customDenoiser::clearBuffers(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->clearFbo(mpPingPongFbo[0].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPingPongFbo[1].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFreezeFbo[0].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFreezeFbo[1].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpLinearZAndNormalFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFilteredPastFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpCurReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPrevReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);

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

    pRenderContext->clearFbo(mpCIELABcolourFbo.get(), float4(0), 1.0f, 0);

    pRenderContext->clearTexture(renderData[kInternalBufferPreviousLinearZAndNormal]->asTexture().get());

}

// Extracts linear z and its derivative from the linear Z texture and packs
// the normal from the world normal texture and packes them into the FBO.
// (It's slightly wasteful to copy linear z here, but having this all
// together in a single buffer is a small simplification, since we make a
// copy of it to refer to in the next frame.)
void customDenoiser::computeLinearZAndNormal(RenderContext* pRenderContext, Texture::SharedPtr pLinearZTexture,
    Texture::SharedPtr pWorldNormalTexture)
{
    auto perImageCB = mpPackLinearZAndNormal["PerImageCB"];
    perImageCB["gLinearZ"] = pLinearZTexture;
    perImageCB["gNormal"] = pWorldNormalTexture;

    mpPackLinearZAndNormal->execute(pRenderContext, mpLinearZAndNormalFbo);
}

void customDenoiser::computeReprojection(RenderContext* pRenderContext,
    Texture::SharedPtr pMotionVectorTexture,
    Texture::SharedPtr pPositionNormalFwidthTexture,
    Texture::SharedPtr pPrevLinearZTexture)
{
    auto perImageCB = mpReprojection["PerImageCB"];

    // Setup textures for our reprojection shader pass
    perImageCB["gMotion"] = pMotionVectorTexture;
    perImageCB["gColor"] = mpCIELABcolourFbo->getColorTexture(0);
    perImageCB["gEmission"] = mpCIELABcolourFbo->getColorTexture(2);
    perImageCB["gAlbedo"] = mpCIELABcolourFbo->getColorTexture(1);
    perImageCB["gPositionNormalFwidth"] = pPositionNormalFwidthTexture;
    perImageCB["gPrevIllum"] = mpFilteredPastFbo->getColorTexture(0);
    perImageCB["gPrevMoments"] = mpPrevReprojFbo->getColorTexture(1);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gPrevLinearZAndNormal"] = pPrevLinearZTexture;
    perImageCB["gPrevHistoryLength"] = mpPrevReprojFbo->getColorTexture(2);

    // Setup variables for our reprojection pass
    perImageCB["gAlpha"] = mAlpha;
    perImageCB["gMomentsAlpha"] = mMomentsAlpha;

    mpReprojection->execute(pRenderContext, mpCurReprojFbo);
}

void customDenoiser::computeFilteredMoments(RenderContext* pRenderContext)
{
    auto perImageCB = mpFilterMoments["PerImageCB"];

    perImageCB["gIllumination"] = mpCurReprojFbo->getColorTexture(0);
    perImageCB["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);
    perImageCB["gMoments"] = mpCurReprojFbo->getColorTexture(1);

    perImageCB["gPhiColor"] = mPhiColor;
    perImageCB["gPhiNormal"] = mPhiNormal;

    mpFilterMoments->execute(pRenderContext, mpPingPongFbo[0]);
}

void customDenoiser::computeAtrousDecomposition(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture)
{
    auto perImageCB = mpAtrous["PerImageCB"];

    perImageCB["gAlbedo"] = pAlbedoTexture;
    perImageCB["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gLinearZAndNormal"] = mpLinearZAndNormalFbo->getColorTexture(0);

    perImageCB["gPhiColor"] = mPhiColor;
    perImageCB["gPhiNormal"] = mPhiNormal;

    for (int i = 0; i < mFilterIterations; i++)
    {
        Fbo::SharedPtr curTargetFbo = mpPingPongFbo[1];

        perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
        perImageCB["gStepSize"] = 1 << i;

        mpAtrous->execute(pRenderContext, curTargetFbo);

        // store the filtered color for the feedback path
        if (i == std::min(mFeedbackTap, mFilterIterations - 1))
        {
            pRenderContext->blit(curTargetFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
        }

        //What happens if the illumination from the final wavelet is propagated back? aka mFeedbackTap == mFilterIterations
        //Answer: You loose specular/reflection detail when the filter iterations are high but you get noise when it is low

        std::swap(mpPingPongFbo[0], mpPingPongFbo[1]);
    }

    if (mFeedbackTap < 0)
    {
        pRenderContext->blit(mpCurReprojFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
    }
}

void customDenoiser::computeFreeze(RenderContext* pRenderContext)
{
    auto perImageCB = mpFreezeLuminance["PerImageCB"];
    // Framebuffers needed for remodulation and freezing
    perImageCB["gIllumination"] = mpPingPongFbo[0]->getColorTexture(0);
    perImageCB["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    perImageCB["gFrozenIllumination"] = mpFreezeFbo[0]->getColorTexture(0);
    // Freezing threshold parameters
    perImageCB["gHistoryThreshold"] = mHistoryThreshold;

    mpFreezeLuminance->execute(pRenderContext, mpFreezeFbo[1]);

    // Save the frozen illumination in preparation for the next frame
    pRenderContext->blit(mpFreezeFbo[1]->getColorTexture(0)->getSRV(), mpFreezeFbo[0]->getRenderTargetView(0));
    // use swap???
}

void customDenoiser::computeEigenvector(RenderContext* pRenderContext)
{
    // Structure Tensor pass
    mpStructureTensor["gSampler"] = mpSampler;
    mpStructureTensor["gSrcTex"] = mpFinalFbo->getColorTexture(0);
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

void customDenoiser::computeOABilateral(RenderContext* pRenderContext)
{
    // Orientation-aligned bilateral filter pass along the eigenvector gradients
    auto perFrameOABF = mpOABilateralGradient["PerFrameCB"];
    perFrameOABF["gSampler"] = mpSampler;
    perFrameOABF["gtwoSigmaD2"] = 2.0f * mSigmaD * mSigmaD;
    perFrameOABF["gtwoSigmaR2"] = 2.0f * mSigmaR * mSigmaR;
    perFrameOABF["gRadius"] = ceilf(mSigmaD * mSigmaDRgradPrecision);
    mpOABilateralGradient["gEigenvectorTex"] = mpEigenvectorFbo->getColorTexture(1); // Eigenvectors
    mpOABilateralGradient["gSrcTex"] = mpFinalFbo->getColorTexture(0); // LAB colour image

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

void customDenoiser::computeOrthogonalFDoG(RenderContext* pRenderContext)
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

void customDenoiser::computeLICFDoG(RenderContext* pRenderContext)
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

float customDenoiser::getCoefficient(float sigma, float x)
{
    float sigmaSquared = sigma * sigma;
    float p = -(x * x) / (2 * sigmaSquared);
    float e = std::exp(p);

    float a = 2 * (float)M_PI * sigmaSquared;
    return e / a;
}

Buffer::SharedPtr customDenoiser::updateKernel(float sigma)
{
    uint32_t kernelRadius = (uint32_t)std::max(1.0f, floor(sigma * 2.45f));
    float sum = 0;
    std::vector<float> weights(kernelRadius + 1);
    for (uint32_t i = 0; i <= kernelRadius; i++)
    {
        weights[i] = getCoefficient(sigma, (float)i);
        sum += (i == 0) ? weights[i] : 2 * weights[i];
    }

    Buffer::SharedPtr pBuf = Buffer::createTyped<float>((kernelRadius * 2) + 1, Resource::BindFlags::ShaderResource);

    for (uint32_t i = 0; i <= kernelRadius; i++)
    {
        float w = weights[i] / sum;
        pBuf->setElement(kernelRadius + i, w);
        pBuf->setElement(kernelRadius - i, w);
    }

    return pBuf;
}

void customDenoiser::renderUI(Gui::Widgets& widget)
{
    int dirty = 0;
    dirty |= (int)widget.checkbox(mFilterEnabled ? "Filter enabled" : "Filter disabled", mFilterEnabled);

    // Custom denoiser parameters
    widget.text("");
    widget.text("Luminance freezing parameters:");
    dirty |= (int)widget.var("History Threshold", mHistoryThreshold, 0.0f, 32.0f, 1.0f);

    // SVGF user parameters
    widget.text("");
    widget.text("Number of filter iterations.  Which");
    widget.text("    iteration feeds into future frames?");
    dirty |= (int)widget.var("Iterations", mFilterIterations, 2, 10, 1);
    dirty |= (int)widget.var("Feedback", mFeedbackTap, -1, mFilterIterations - 2, 1);

    widget.text("");
    widget.text("Contol edge stopping on bilateral fitler");
    dirty |= (int)widget.var("For Color", mPhiColor, 0.0f, 10000.0f, 0.01f);
    dirty |= (int)widget.var("For Normal", mPhiNormal, 0.001f, 1000.0f, 0.2f);

    widget.text("");
    widget.text("How much history should be used?");
    widget.text("    (alpha; 0 = full reuse; 1 = no reuse)");
    dirty |= (int)widget.var("Alpha", mAlpha, 0.0f, 1.0f, 0.001f);
    dirty |= (int)widget.var("Moments Alpha", mMomentsAlpha, 0.0f, 1.0f, 0.001f);

    // FDoG user parameters
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
    dirty |= (int)widget.var("Quantiser Step", mQuantizerStep, 1, 16, 1);
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
    dirty |= (int)widget.var("Colour Quantiser Step:", mColourQuantiseStep, 1, 16, 1);
    widget.text("Minimum Quantisation Sharpness:");
    dirty |= (int)widget.var("Lambda Delta", mLambdaDelta, 0.0f, 1.0f, 0.01f);
    widget.text("Maximum Quantisation Sharpness:");
    dirty |= (int)widget.var("Omega Delta", mOmegaDelta, 0.0f, 1.0f, 0.01f);
    widget.text("Minimum Quantisation Gradient:");
    dirty |= (int)widget.var("Lambda Phi", mLambdaPhi, 0.0f, 1.0f, 0.01f);
    widget.text("Maximum Quantisation Gradient:");
    dirty |= (int)widget.var("Omega Phi", mOmegaPhi, 0.0f, 1.0f, 0.01f);

    if (dirty) mBuffersNeedClear = true;
}


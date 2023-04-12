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
#include "KuwaharaFilter.h"
#define _USE_MATH_DEFINES
#include <math.h>


namespace
{
    const char kDesc[] = "Perform Anisotropic Kuwahara Filtering";

    // Input and Output buffer names
    const char kSrc[] = "source";
    const char kDst[] = "destination";

    // Names of valid entries in the parameter dictionary.
    const char kKernelRadius[] = "kernelRadius";
    const char kFilterIterations[] = "filterIterations";
    const char kSharpness[] = "sharpness";
    const char kHardness[] = "hardness";
    const char kAlpha[] = "alpha";
    const char kGamma[] = "gamma";
    const char kZeta[] = "zeta";
    const char kManualZeta[] = "manualZeta";

    // Shader source files
    const char kStructureTensorShader[] = "RenderPasses/KuwaharaFilter/StructureTensor.ps.slang";
    const char kGaussianShader[] = "RenderPasses/KuwaharaFilter/GaussianBlur.ps.slang";
    const char kKuwaharaShader[] = "RenderPasses/KuwaharaFilter/KuwaharaFilter.ps.slang";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("KuwaharaFilter", kDesc, KuwaharaFilter::create);
}

KuwaharaFilter::KuwaharaFilter(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kKernelRadius) mKernelRadius = value;
        else if (key == kFilterIterations) mFilterIterations = value;
        else if (key == kSharpness) mSharpness = value;
        else if (key == kHardness) mHardness = value;
        else if (key == kAlpha) mAlpha = value;
        else if (key == kGamma) mGamma = value;
        else if (key == kZeta) mZeta = value;
        else if (key == kManualZeta) mManualZeta = value;
        else logWarning("Unknown field '" + key + "' in Kuwahara Filter dictionary");
    }

    mpStructureTensor = FullScreenPass::create(kStructureTensorShader);
    mpKuwahara = FullScreenPass::create(kKuwaharaShader);

    // Setup two Gaussian passes, one for the horizontal pass and one for the vertical pass.
    Program::DefineList defines;
    defines.add("_KERNEL_WIDTH", std::to_string(mGaussianKernelWidth));
    defines.add("_HORIZONTAL_BLUR");
    mpHorizontalBlur = FullScreenPass::create(kGaussianShader, defines);
    defines.remove("_HORIZONTAL_BLUR");
    defines.add("_VERTICAL_BLUR");
    defines.add("_ANISOTROPY");
    mpVerticalBlur = FullScreenPass::create(kGaussianShader, defines);

    // Make the programs share the shader vars
    mpVerticalBlur->setVars(mpHorizontalBlur->getVars());

    // Update Gaussian kernel weights
    updateKernel();

    assert(mpStructureTensor && mpKuwahara && mpHorizontalBlur && mpVerticalBlur);

    // Create and setup sampler.
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point).setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpSampler = Sampler::create(samplerDesc);
}

KuwaharaFilter::SharedPtr KuwaharaFilter::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new KuwaharaFilter(dict));
}

std::string KuwaharaFilter::getDesc() { return kDesc; }

Dictionary KuwaharaFilter::getScriptingDictionary()
{
    Dictionary dict;
    dict[kKernelRadius] = mKernelRadius;
    dict[kFilterIterations] = mFilterIterations;
    dict[kSharpness] = mSharpness;
    dict[kHardness] = mHardness;
    dict[kAlpha] = mAlpha;
    dict[kGamma] = mGamma;
    dict[kZeta] = mZeta;
    dict[kManualZeta] = mManualZeta;
    return dict;
}

RenderPassReflection KuwaharaFilter::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput(kDst, "output filtered image").format(ResourceFormat::RGBA16Float);
    //Use the version above for HDR output.
    reflector.addOutput(kDst, "output filtered image");
    reflector.addInput(kSrc, "input image to be filtered");
    return reflector;
}

void KuwaharaFilter::compile(RenderContext* pContext, const CompileData& compileData)
{
    // Allocate and setup internal Screen-size FrameBufferObjects with 1 RGBA16Float buffer
    Fbo::Desc desc;
    desc.setColorTarget(0, Falcor::ResourceFormat::RGBA16Float);
    mpStructureTensorFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpTmpGaussianFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);
    mpKuwaharaFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);

    // This FBO only needs to store 2 components so it uses a smaller resource format with only RG components.
    desc.setColorTarget(0, Falcor::ResourceFormat::RG16Float);
    mpGaussianFbo = Fbo::create2D(compileData.defaultTexDims.x, compileData.defaultTexDims.y, desc);

    mFBOsNeedClear = true;
}

void KuwaharaFilter::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested external resources
    // Setup the external resources
    Texture::SharedPtr pSrc = renderData[kSrc]->asTexture();
    Texture::SharedPtr pDst = renderData[kDst]->asTexture();

    // If the screen size changes, the render graph will be recompiled so clear the old FBOs. 
    if (mFBOsNeedClear)
    {
        clearBuffers(pRenderContext);
        mFBOsNeedClear = false;
    }

    assert(mpKuwaharaFbo && mpKuwaharaFbo->getWidth() == pSrc->getWidth() && mpKuwaharaFbo->getHeight() == pSrc->getHeight());

    //mpKuwaharaFbo->attachColorTarget(renderData[kDst]->asTexture(), 0);

    // Structural Tensor pass
    mpStructureTensor["gSampler"] = mpSampler;
    mpStructureTensor["gSrcTex"] = pSrc;
    mpStructureTensor->execute(pRenderContext, mpStructureTensorFbo);
    //mpStructureTensor->execute(pRenderContext, mpKuwaharaFbo);
    
    // Horizontal Gaussian pass
    mpHorizontalBlur["gSampler"] = mpSampler;
    mpHorizontalBlur["gSrcTex"] = mpStructureTensorFbo->getColorTexture(0);
    mpHorizontalBlur->execute(pRenderContext, mpTmpGaussianFbo);
    //mpHorizontalBlur->execute(pRenderContext, mpKuwaharaFbo);
    
    // Vertical Gaussian pass
    mpVerticalBlur["gSrcTex"] = mpTmpGaussianFbo->getColorTexture(0);
    mpVerticalBlur->execute(pRenderContext, mpGaussianFbo);
    //mpVerticalBlur->execute(pRenderContext, mpKuwaharaFbo);

    for (int l = 0; l < mFilterIterations; l++)
    {
        //Kuwahara Filter pass
        auto perFrameCB = mpKuwahara["PerFrameCB"];
        perFrameCB["gSampler"] = mpSampler;
        perFrameCB["gRadius"] = mKernelRadius;
        perFrameCB["gHardness"] = mHardness;
        perFrameCB["gSharpness"] = mSharpness;
        perFrameCB["gAlpha"] = mAlpha;
        perFrameCB["gZeta"] = mZeta / mKernelRadius; // Zeta is the value mZeta divided by the kernel radius.
        perFrameCB["gEta"] = calulateEta();
        mpKuwahara["gTensorTex"] = mpGaussianFbo->getColorTexture(0);
        if (l > 0) {
            mpKuwahara["gSrcTex"] = mpKuwaharaFbo->getColorTexture(0);
        }
        else
        {
            mpKuwahara["gSrcTex"] = pSrc;
        }
        mpKuwahara->execute(pRenderContext, mpKuwaharaFbo);
    }
    
    // Blit into the output texture.
    pRenderContext->blit(mpKuwaharaFbo->getColorTexture(0)->getSRV(), pDst->getRTV());
    //pRenderContext->blit(pSrc->getSRV(), pDst->getRTV());
}

void KuwaharaFilter::clearBuffers(RenderContext* pRenderContext)
{
    pRenderContext->clearFbo(mpStructureTensorFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpTmpGaussianFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpGaussianFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->clearFbo(mpKuwaharaFbo.get(), float4(0), 1.0f, 0);
}

float KuwaharaFilter::calulateEta()
{
    // Eta = (Zeta + cos(Gamma * Pi)) / (sin(Gamma * Pi)^2) -- The Zero Crossing function. 
    float eta = ((mZeta / mKernelRadius) + cos(mGamma * (float)M_PI)) / (sin(mGamma * (float)M_PI) * sin(mGamma * (float)M_PI));
    return eta;
}

void KuwaharaFilter::renderUI(Gui::Widgets& widget)
{
    int dirty = 0;

    widget.text("");
    widget.text("Kuwahara Filter kernel radius size:");
    dirty |= (int)widget.var("Radius", (int&)mKernelRadius, 1, 10, 1);
    widget.text("Kuwahara Filter Iterations:");
    dirty |= (int)widget.var("Iterations", mFilterIterations, 1, 10, 1);

    widget.text("");
    widget.text("Kuwahara Filter Hardness:");
    dirty |= (int)widget.var("Hardness", mHardness, 1.0f, 100.0f, 1.0f);
    widget.text("Kuwahara Filter Sharpness:");
    dirty |= (int)widget.var("Sharpness", mSharpness, 1.0f, 18.0f, 1.0f);

    widget.text("");
    widget.text("Control the eccentricity of the anisotropy of the filter:");
    dirty |= (int)widget.var("Alpha", mAlpha, 0.01f, 2.0f, 0.1f);

    widget.text("");
    widget.text("Control the overlap of the weighting function at");
    widget.text("   the filter sides:");
    dirty |= (int)widget.var("Gamma", mGamma, 0.125f, 0.25f, 0.001f);
    dirty |= (int)widget.checkbox(mManualZeta ? "Manual zeta enabled" : "Manual zeta disabled", mManualZeta);
    if (mManualZeta)
    {
        widget.text("Control the overlap of the weighting function at");
        widget.text("   the filter origin:");
        dirty |= (int)widget.var("Zeta", mZeta, 0.01f, 3.0f, 0.1f);
    }
    else {
        mZeta = 2.0f; // Set zeta to the default value of 2.0f if manual zeta is not enabled. 
    }

    if (dirty) {
        mFBOsNeedClear = true;
    }
}

float getCoefficient(float x)
{
    const float sigmaSquared = 4.0f; // Sigma is always set to 2.0f
    float p = -(x * x) / (2 * sigmaSquared);
    float e = std::exp(p);

    float a = 2 * (float)M_PI * sigmaSquared;
    return e / a;
}

void KuwaharaFilter::updateKernel()
{
    uint32_t center = mGaussianKernelWidth / 2;
    float sum = 0;
    std::vector<float> weights(center + 1);
    for (uint32_t i = 0; i <= center; i++)
    {
        weights[i] = getCoefficient((float)i);
        sum += (i == 0) ? weights[i] : 2 * weights[i];
    }

    Buffer::SharedPtr pBuf = Buffer::createTyped<float>(mGaussianKernelWidth, Resource::BindFlags::ShaderResource);

    for (uint32_t i = 0; i <= center; i++)
    {
        float w = weights[i] / sum;
        pBuf->setElement(center + i, w);
        pBuf->setElement(center - i, w);
    }

    mpHorizontalBlur["weights"] = pBuf;
}

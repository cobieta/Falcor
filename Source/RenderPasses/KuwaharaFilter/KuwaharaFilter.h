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
#include "FalcorExperimental.h"


using namespace Falcor;

class KuwaharaFilter : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<KuwaharaFilter>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override {}
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }


private:
    KuwaharaFilter(const Dictionary& dict);
    void updateKernel();
    float calulateEta();

    bool mFBOsNeedClear = false;
    void clearBuffers(RenderContext* pRenderContext);

    // Kuwahara Filter parameters
    uint32_t mKernelRadius = 6;
    int32_t mFilterIterations = 1;
    float mSharpness = 8.0f;
    float mHardness = 8.0f;
    float mAlpha = 1.0f; // When Alpha is set to 1.0, the minor radius of the eclipse >= half of the radius.
    float mGamma = 0.1875f; // Equivalent to 3/(2 * Number of sectors), in this case number of sectors is 8 so 3/16
    float mZeta = 2.0f; // Default value is 2. 
    bool mManualZeta = false; // Controls whether the user has manual control over zeta or it is the default value of 2 / kernel radius

    const uint32_t mGaussianKernelWidth = 15; // Used to control the smoothing of the structure tensor, not user controlled

    // Kuwahara passes
    FullScreenPass::SharedPtr mpStructureTensor;
    FullScreenPass::SharedPtr mpHorizontalBlur;
    FullScreenPass::SharedPtr mpVerticalBlur;
    FullScreenPass::SharedPtr mpKuwahara;

    // Intermediate framebuffers
    Fbo::SharedPtr mpStructureTensorFbo;
    Fbo::SharedPtr mpGaussianFbo;
    Fbo::SharedPtr mpTmpGaussianFbo;
    Fbo::SharedPtr mpKuwaharaFbo;
    Sampler::SharedPtr mpSampler;
};

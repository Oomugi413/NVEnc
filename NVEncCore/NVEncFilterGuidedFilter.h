// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once

#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamGuidedFilter : public NVEncFilterParam {
public:
    VppGuidedfilter guidedfilter;

    NVEncFilterParamGuidedFilter() : guidedfilter() {};
    virtual ~NVEncFilterParamGuidedFilter() {};
    virtual tstring print() const override;
};

class NVEncFilterGuidedFilter : public NVEncFilter {
public:
    NVEncFilterGuidedFilter();
    virtual ~NVEncFilterGuidedFilter();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;
private:
    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamGuidedFilter> prm);
    RGY_ERR allocIntermediate(const RGYFrameInfo& frameOut);
    RGY_ERR procPlane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, int radius, float eps, cudaStream_t stream);
    RGY_ERR procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, int radius, float eps, bool chroma, cudaStream_t stream);

    std::unique_ptr<CUMemBuf> m_bufA;
    std::unique_ptr<CUMemBuf> m_bufB;
    int m_abAllocW;
    int m_abAllocH;
};

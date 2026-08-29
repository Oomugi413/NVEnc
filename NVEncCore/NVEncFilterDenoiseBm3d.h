// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2026 rigaya
// Copyright (c) 2015 Sampsa Sarjanoja
// Copyright (c) 2015-2016 mawen1250
// Copyright (c) 2021 HuangZhangming
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

#include <array>
#include "NVEncFilter.h"
#include "rgy_prm.h"

class NVEncFilterParamDenoiseBm3d : public NVEncFilterParam {
public:
    VppDenoiseBm3d bm3d;

    NVEncFilterParamDenoiseBm3d() : bm3d() {};
    virtual ~NVEncFilterParamDenoiseBm3d() {};
    virtual tstring print() const override { return bm3d.print(); };
};

class NVEncFilterDenoiseBm3d : public NVEncFilter {
public:
    NVEncFilterDenoiseBm3d();
    virtual ~NVEncFilterDenoiseBm3d();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    virtual void resetTemporalState() override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    RGY_ERR ensureScratch(int width, int height);
    RGY_ERR ensureRingBuffers(int planeIdx, int width, int height);
    RGY_ERR procPlane(int planeIdx, RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane, cudaStream_t stream);
    RGY_ERR pushNoisyToRing(int planeIdx, const RGYFrameInfo *pInputPlane, cudaStream_t stream);
    RGY_ERR pushBasicToRing(int planeIdx, cudaStream_t stream);

    std::unique_ptr<CUMemBuf> m_bufSimilarCoords;
    std::unique_ptr<CUMemBuf> m_bufBlockCounts;
    std::unique_ptr<CUMemBuf> m_bufAccumulator;
    std::unique_ptr<CUMemBuf> m_bufWeightMap;
    std::unique_ptr<CUMemBuf> m_bufBasicEstimate;
    std::unique_ptr<CUMemBuf> m_bufSimilarFrameIdx;
    std::array<std::unique_ptr<CUMemBuf>, 3> m_pastNoisyRing;
    std::array<std::unique_ptr<CUMemBuf>, 3> m_pastBasicRing;
    std::array<int, 3> m_ringW;
    std::array<int, 3> m_ringH;
    std::array<int, 3> m_ringNoisyPitch;
    std::array<int, 3> m_ringBasicPitch;
    int m_ringRadius;
    int m_ringSlotCursor;
    int m_ringFilled;
    int m_scratchW;
    int m_scratchH;
    int m_scratchBlockStep;
    int m_scratchGroupSize;
    int m_accPitch;
    int m_wmapPitch;
    int m_basicPitch;
};

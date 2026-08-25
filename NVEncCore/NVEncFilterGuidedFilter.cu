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

#include "convert_csp.h"
#include "NVEncFilterGuidedFilter.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int GUIDEDFILTER_BLOCK_X = 32;
static const int GUIDEDFILTER_BLOCK_Y = 8;

__device__ __forceinline__ int guidedfilter_clamp_index(const int value, const int upper) {
    return min(max(value, 0), upper);
}

// 自己guided filterの第1段: 入力Iから局所線形係数a,bを求める。
template<typename Type, int bit_depth>
__global__ void kernel_guidedfilter_calc_ab(
    float *__restrict__ pA,
    float *__restrict__ pB,
    const int abPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height,
    const int radius, const float eps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float maxValue = (float)((1u << bit_depth) - 1u);
    const float invMax = 1.0f / maxValue;

    // bit深度によらず分散が安定するよう、[0, 1]へ正規化してbox sumを求める。
    float sumI = 0.0f;
    float sumII = 0.0f;
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        const int sy = guidedfilter_clamp_index(y + dy, height - 1);
        const Type *row = (const Type *)(pSrc + sy * srcPitch);
        for (int dx = -radius; dx <= radius; dx++) {
            const int sx = guidedfilter_clamp_index(x + dx, width - 1);
            const float value = (float)row[sx] * invMax;
            sumI += value;
            sumII += value * value;
            count++;
        }
    }
    const float invCount = 1.0f / (float)count;
    const float meanI = sumI * invCount;
    const float meanII = sumII * invCount;
    const float varI = meanII - meanI * meanI;
    const float a = varI / (varI + eps);
    const float b = (1.0f - a) * meanI;

    pA[y * abPitch + x] = a;
    pB[y * abPitch + x] = b;
}

// 自己guided filterの第2段: 局所平均したa,bと入力Iから出力qを求める。
template<typename Type, int bit_depth>
__global__ void kernel_guidedfilter_calc_q(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height,
    const float *__restrict__ pA,
    const float *__restrict__ pB,
    const int abPitch,
    const int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float maxValue = (float)((1u << bit_depth) - 1u);
    const float invMax = 1.0f / maxValue;

    float sumA = 0.0f;
    float sumB = 0.0f;
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        const int sy = guidedfilter_clamp_index(y + dy, height - 1);
        for (int dx = -radius; dx <= radius; dx++) {
            const int sx = guidedfilter_clamp_index(x + dx, width - 1);
            sumA += pA[sy * abPitch + sx];
            sumB += pB[sy * abPitch + sx];
            count++;
        }
    }
    const float invCount = 1.0f / (float)count;
    const float meanA = sumA * invCount;
    const float meanB = sumB * invCount;

    const Type *srcRow = (const Type *)(pSrc + y * srcPitch);
    const float input = (float)srcRow[x] * invMax;
    const float output = meanA * input + meanB;
    const int outputValue = (int)fminf(fmaxf(output * maxValue + 0.5f, 0.0f), maxValue);

    Type *dstRow = (Type *)(pDst + y * dstPitch);
    dstRow[x] = (Type)outputValue;
}

template<typename Type, int bit_depth>
static RGY_ERR guidedfilter_plane(
    RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    float *pA, float *pB, const int abPitch,
    const int radius, const float eps, cudaStream_t stream) {
    const dim3 blockSize(GUIDEDFILTER_BLOCK_X, GUIDEDFILTER_BLOCK_Y);
    const dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));

    kernel_guidedfilter_calc_ab<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        pA, pB, abPitch,
        (const uint8_t *)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pOutputPlane->width, pOutputPlane->height, radius, eps);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }

    kernel_guidedfilter_calc_q<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (uint8_t *)pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        (const uint8_t *)pInputPlane->ptr[0], pInputPlane->pitch[0],
        pOutputPlane->width, pOutputPlane->height,
        pA, pB, abPitch, radius);
    return err_to_rgy(cudaGetLastError());
}

NVEncFilterGuidedFilter::NVEncFilterGuidedFilter() :
    NVEncFilter(),
    m_bufA(),
    m_bufB(),
    m_abAllocW(0),
    m_abAllocH(0) {
    m_name = _T("guidedfilter");
}

NVEncFilterGuidedFilter::~NVEncFilterGuidedFilter() {
    close();
}

tstring NVEncFilterParamGuidedFilter::print() const {
    return guidedfilter.print();
}

RGY_ERR NVEncFilterGuidedFilter::checkParam(const std::shared_ptr<NVEncFilterParamGuidedFilter> prm) {
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->guidedfilter.radius < 1 || 32 < prm->guidedfilter.radius) {
        prm->guidedfilter.radius = clamp(prm->guidedfilter.radius, 1, 32);
        AddMessage(RGY_LOG_WARN, _T("radius should be in range of %d - %d.\n"), 1, 32);
    }
    if (prm->guidedfilter.eps < 0.0001f || 1.0f < prm->guidedfilter.eps) {
        prm->guidedfilter.eps = clamp(prm->guidedfilter.eps, 0.0001f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("eps should be in range of %.4f - %.1f.\n"), 0.0001f, 1.0f);
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterGuidedFilter::allocIntermediate(const RGYFrameInfo& frameOut) {
    if (m_bufA && m_bufB && frameOut.width <= m_abAllocW && frameOut.height <= m_abAllocH) {
        return RGY_ERR_NONE;
    }

    const size_t bufferBytes = (size_t)frameOut.width * (size_t)frameOut.height * sizeof(float);
    auto bufA = std::make_unique<CUMemBuf>(bufferBytes);
    auto sts = bufA->alloc();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate intermediate a buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    auto bufB = std::make_unique<CUMemBuf>(bufferBytes);
    sts = bufB->alloc();
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate intermediate b buffer: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    m_bufA = std::move(bufA);
    m_bufB = std::move(bufB);
    m_abAllocW = frameOut.width;
    m_abAllocH = frameOut.height;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterGuidedFilter::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamGuidedFilter>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate output buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    sts = allocIntermediate(prm->frameOut);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterGuidedFilter::procPlane(
    RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    const int radius, const float eps, cudaStream_t stream) {
    if (pOutputPlane->width > m_abAllocW || pOutputPlane->height > m_abAllocH) {
        AddMessage(RGY_LOG_ERROR, _T("plane size %dx%d exceeds allocated intermediate buffer %dx%d.\n"),
            pOutputPlane->width, pOutputPlane->height, m_abAllocW, m_abAllocH);
        return RGY_ERR_INVALID_PARAM;
    }

    auto pA = (float *)m_bufA->ptr;
    auto pB = (float *)m_bufB->ptr;
    switch (RGY_CSP_DATA_TYPE[pInputPlane->csp]) {
    case RGY_DATA_TYPE_U8:
        return guidedfilter_plane<uint8_t, 8>(pOutputPlane, pInputPlane, pA, pB, m_abAllocW, radius, eps, stream);
    case RGY_DATA_TYPE_U16:
        return guidedfilter_plane<uint16_t, 16>(pOutputPlane, pInputPlane, pA, pB, m_abAllocW, radius, eps, stream);
    default:
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pInputPlane->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
}

RGY_ERR NVEncFilterGuidedFilter::procFrame(
    RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    const int radius, const float eps, const bool chroma, cudaStream_t stream) {
    // alphaは色差処理指定にかかわらず保持する。
    const int numColorPlanes = RGY_CSP_PLANES[pOutputFrame->csp] - (rgy_csp_has_alpha(pOutputFrame->csp) ? 1 : 0);
    for (int i = 0; i < numColorPlanes; i++) {
        auto planeDst = getPlane(pOutputFrame, (RGY_PLANE)i);
        const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        RGY_ERR sts = RGY_ERR_NONE;
        if (i > 0 && !chroma) {
            sts = copyPlaneAsync(&planeDst, &planeSrc, stream);
        } else {
            sts = procPlane(&planeDst, &planeSrc, radius, eps, stream);
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("Failed to guidedfilter frame plane %d %s: %s\n"),
                i, RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
            return sts;
        }
    }
    return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
}

RGY_ERR NVEncFilterGuidedFilter::run_filter(
    const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    RGY_ERR sts = RGY_ERR_NONE;
    if (pInputFrame->ptr[0] == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        auto pOutFrame = m_frameBuf[m_nFrameIdx].get();
        ppOutputFrames[0] = &pOutFrame->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (interlaced(*pInputFrame)) {
        return filter_as_interlaced_pair(pInputFrame, ppOutputFrames[0], stream);
    }
    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("guidedfilter only supports device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("guidedfilter does not support csp conversion.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamGuidedFilter>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    sts = procFrame(ppOutputFrames[0], pInputFrame,
        prm->guidedfilter.radius, prm->guidedfilter.eps, prm->guidedfilter.chroma, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at guidedfilter(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterGuidedFilter::close() {
    m_frameBuf.clear();
    m_bufA.reset();
    m_bufB.reset();
    m_abAllocW = 0;
    m_abAllocH = 0;
}

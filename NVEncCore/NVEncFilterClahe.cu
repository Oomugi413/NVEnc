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

#include <algorithm>
#include <cstdint>
#include "convert_csp.h"
#include "NVEncFilterClahe.h"
#include "rgy_prm.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int CLAHE_BLOCK_X = 32;
static const int CLAHE_BLOCK_Y = 8;

template<int bin_bit_depth, int storage_bit_depth>
__global__ void kernel_clahe_cdf(
    uint16_t *__restrict__ pTransform,
    uint32_t *__restrict__ pHist,
    const int width, const int height,
    const int tilesX, const int tilesY,
    const float slope) {
    static constexpr int bins = 1 << bin_bit_depth;
    static constexpr uint32_t maxValue = (1u << storage_bit_depth) - 1u;

    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    uint32_t *hist = pHist + (ty * tilesX + tx) * bins;
    uint16_t *xfrm = pTransform + (ty * tilesX + tx) * bins;

    const int tileX0 = (tx       * width)  / tilesX;
    const int tileX1 = ((tx + 1) * width)  / tilesX;
    const int tileY0 = (ty       * height) / tilesY;
    const int tileY1 = ((ty + 1) * height) / tilesY;
    const int tilePixels = (tileX1 - tileX0) * (tileY1 - tileY0);

    // 画素を含まないタイルでは0除算を避け、恒等変換表を作る。
    if (tilePixels == 0) {
        for (int i = 0; i < bins; i++) {
            xfrm[i] = (uint16_t)(((uint32_t)i * maxValue + (bins - 1) / 2) / (bins - 1));
        }
        return;
    }

    int clipLimit = (int)(slope * (float)tilePixels * (1.0f / (float)bins));
    if (clipLimit < 1) clipLimit = 1;

    uint32_t excess = 0u;
    for (int i = 0; i < bins; i++) {
        if (hist[i] > (uint32_t)clipLimit) {
            excess += hist[i] - (uint32_t)clipLimit;
            hist[i] = (uint32_t)clipLimit;
        }
    }

    // 余りは元実装と同様に捨て、切り詰め分を全binへ均等に戻す。
    const uint32_t redist = excess / bins;
    for (int i = 0; i < bins; i++) {
        hist[i] += redist;
    }

    uint32_t cum = 0u;
    const float maxVal = (float)maxValue;
    const float scale = maxVal / (float)tilePixels;
    for (int i = 0; i < bins; i++) {
        cum += hist[i];
        float value = (float)cum * scale + 0.5f;
        if (value < 0.0f) value = 0.0f;
        if (value > maxVal) value = maxVal;
        xfrm[i] = (uint16_t)value;
    }
}

template<typename Type, int storage_bit_depth, int bin_bit_depth>
__global__ void kernel_clahe_hist(
    uint32_t *__restrict__ pHist,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height,
    const int tilesX, const int tilesY) {
    static constexpr int bins = 1 << bin_bit_depth;
    static constexpr int numSubHist = (bin_bit_depth > 8) ? 4 : 8;
    __shared__ uint32_t subHist[numSubHist][bins];

    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    const int lid = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockSize = blockDim.x * blockDim.y;
    const int subId = lid & (numSubHist - 1);

    // shared memory上のヒストグラムを複製してatomic競合を抑える。
    for (int i = lid; i < numSubHist * bins; i += blockSize) {
        ((uint32_t *)subHist)[i] = 0u;
    }
    __syncthreads();

    const int tileX0 = (tx       * width)  / tilesX;
    const int tileX1 = ((tx + 1) * width)  / tilesX;
    const int tileY0 = (ty       * height) / tilesY;
    const int tileY1 = ((ty + 1) * height) / tilesY;

    // pitchはbyte単位なので、行頭を求めてから画素型へ変換する。
    for (int py = tileY0 + threadIdx.y; py < tileY1; py += blockDim.y) {
        const Type *row = (const Type *)(pSrc + py * srcPitch);
        for (int px = tileX0 + threadIdx.x; px < tileX1; px += blockDim.x) {
            int value = (int)row[px];
            if (storage_bit_depth > bin_bit_depth) {
                value >>= ((storage_bit_depth > bin_bit_depth) ? (storage_bit_depth - bin_bit_depth) : 0);
            }
            if (value > bins - 1) value = bins - 1;
            atomicAdd(&subHist[subId][value], 1u);
        }
    }
    __syncthreads();

    uint32_t *hist = pHist + (ty * tilesX + tx) * bins;
    for (int bin = lid; bin < bins; bin += blockSize) {
        uint32_t sum = 0u;
        for (int sub = 0; sub < numSubHist; sub++) {
            sum += subHist[sub][bin];
        }
        hist[bin] = sum;
    }
}

template<typename Type, int storage_bit_depth, int bin_bit_depth>
__global__ void kernel_clahe_apply(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height,
    const uint16_t *__restrict__ pTransform,
    const int tilesX, const int tilesY) {
    static constexpr int bins = 1 << bin_bit_depth;
    static constexpr float maxValue = (float)((1u << storage_bit_depth) - 1u);

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const Type *srcRow = (const Type *)(pSrc + y * srcPitch);
    Type *dstRow = (Type *)(pDst + y * dstPitch);

    int bin = (int)srcRow[x];
    if (storage_bit_depth > bin_bit_depth) {
        bin >>= ((storage_bit_depth > bin_bit_depth) ? (storage_bit_depth - bin_bit_depth) : 0);
    }
    if (bin > bins - 1) bin = bins - 1;

    // タイル中心を格子点とし、近傍4タイルの変換値をbilinear補間する。
    const float tileW = (float)width / (float)tilesX;
    const float tileH = (float)height / (float)tilesY;
    const float fx = ((float)x + 0.5f) / tileW - 0.5f;
    const float fy = ((float)y + 0.5f) / tileH - 0.5f;
    int tx0 = (int)floorf(fx);
    int ty0 = (int)floorf(fy);
    tx0 = (tx0 < 0) ? 0 : ((tx0 >= tilesX) ? tilesX - 1 : tx0);
    ty0 = (ty0 < 0) ? 0 : ((ty0 >= tilesY) ? tilesY - 1 : ty0);
    const int tx1 = (tx0 + 1 >= tilesX) ? tilesX - 1 : tx0 + 1;
    const int ty1 = (ty0 + 1 >= tilesY) ? tilesY - 1 : ty0 + 1;
    float u = fx - (float)tx0;
    float v = fy - (float)ty0;
    u = (u < 0.0f) ? 0.0f : ((u > 1.0f) ? 1.0f : u);
    v = (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);

    const float t00 = (float)pTransform[(ty0 * tilesX + tx0) * bins + bin];
    const float t10 = (float)pTransform[(ty0 * tilesX + tx1) * bins + bin];
    const float t01 = (float)pTransform[(ty1 * tilesX + tx0) * bins + bin];
    const float t11 = (float)pTransform[(ty1 * tilesX + tx1) * bins + bin];
    const float top = t00 + (t10 - t00) * u;
    const float bottom = t01 + (t11 - t01) * u;
    float value = top + (bottom - top) * v;
    if (value < 0.0f) value = 0.0f;
    if (value > maxValue) value = maxValue;
    dstRow[x] = (Type)(value + 0.5f);
}

template<typename Type, int storage_bit_depth, int bin_bit_depth>
static RGY_ERR clahe_plane(
    RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    uint32_t *pHist, uint16_t *pTransform,
    const int tilesX, const int tilesY, const float slope, cudaStream_t stream) {
    dim3 blockSize(CLAHE_BLOCK_X, CLAHE_BLOCK_Y);

    // 3パスを同一streamへ投入し、CUDAのstream順序保証で依存関係を保つ。
    dim3 histGrid(tilesX, tilesY);
    kernel_clahe_hist<Type, storage_bit_depth, bin_bit_depth><<<histGrid, blockSize, 0, stream>>>(
        pHist, pInputPlane->ptr[0], pInputPlane->pitch[0],
        pInputPlane->width, pInputPlane->height, tilesX, tilesY);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    dim3 cdfGrid(tilesX, tilesY);
    kernel_clahe_cdf<bin_bit_depth, storage_bit_depth><<<cdfGrid, dim3(1, 1), 0, stream>>>(
        pTransform, pHist, pInputPlane->width, pInputPlane->height, tilesX, tilesY, slope);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    dim3 applyGrid(divCeil(pOutputPlane->width, (int)blockSize.x), divCeil(pOutputPlane->height, (int)blockSize.y));
    kernel_clahe_apply<Type, storage_bit_depth, bin_bit_depth><<<applyGrid, blockSize, 0, stream>>>(
        pOutputPlane->ptr[0], pOutputPlane->pitch[0],
        pInputPlane->ptr[0], pInputPlane->pitch[0],
        pOutputPlane->width, pOutputPlane->height, pTransform, tilesX, tilesY);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    return RGY_ERR_NONE;
}

template<typename Type, int storage_bit_depth>
static RGY_ERR clahe_plane_dispatch(
    RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    uint32_t *pHist, uint16_t *pTransform,
    const int histBitdepth, const int tilesX, const int tilesY, const float slope, cudaStream_t stream) {
    switch (std::min(histBitdepth, 10)) {
    case 8: return clahe_plane<Type, storage_bit_depth, 8>(pOutputPlane, pInputPlane, pHist, pTransform, tilesX, tilesY, slope, stream);
    case 9: return clahe_plane<Type, storage_bit_depth, 9>(pOutputPlane, pInputPlane, pHist, pTransform, tilesX, tilesY, slope, stream);
    case 10: return clahe_plane<Type, storage_bit_depth, 10>(pOutputPlane, pInputPlane, pHist, pTransform, tilesX, tilesY, slope, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

NVEncFilterClahe::NVEncFilterClahe() :
    m_histBuf(), m_transformBuf(), m_tilesX(0), m_tilesY(0), m_binBitdepth(0) {
    m_name = _T("clahe");
}

NVEncFilterClahe::~NVEncFilterClahe() {
    close();
}

RGY_ERR NVEncFilterClahe::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamClahe>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->clahe.tiles_x < 2 || 32 < prm->clahe.tiles_x) {
        prm->clahe.tiles_x = clamp(prm->clahe.tiles_x, 2, 32);
        AddMessage(RGY_LOG_WARN, _T("tiles_x should be in range of %d - %d.\n"), 2, 32);
    }
    if (prm->clahe.tiles_y < 2 || 32 < prm->clahe.tiles_y) {
        prm->clahe.tiles_y = clamp(prm->clahe.tiles_y, 2, 32);
        AddMessage(RGY_LOG_WARN, _T("tiles_y should be in range of %d - %d.\n"), 2, 32);
    }
    if (prm->clahe.slope < 1.0f || 40.0f < prm->clahe.slope) {
        prm->clahe.slope = clamp(prm->clahe.slope, 1.0f, 40.0f);
        AddMessage(RGY_LOG_WARN, _T("slope should be in range of %.1f - %.1f.\n"), 1.0f, 40.0f);
    }

    const int storageBitdepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    if (prm->histBitdepth < 8 || storageBitdepth < prm->histBitdepth) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported bit depth combination: histogram %d-bit, storage %d-bit.\n"),
            prm->histBitdepth, storageBitdepth);
        return RGY_ERR_UNSUPPORTED;
    }
    if (RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U8
        && RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }

    auto sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[prm->frameOut.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const int tilesX = prm->clahe.tiles_x;
    const int tilesY = prm->clahe.tiles_y;
    const int binBitdepth = std::min(prm->histBitdepth, 10);
    const int bins = 1 << binBitdepth;
    if (!m_histBuf || !m_transformBuf || tilesX > m_tilesX || tilesY > m_tilesY || binBitdepth > m_binBitdepth) {
        const size_t histBytes = (size_t)tilesX * (size_t)tilesY * bins * sizeof(uint32_t);
        const size_t transformBytes = (size_t)tilesX * (size_t)tilesY * bins * sizeof(uint16_t);
        auto histBuf = std::unique_ptr<CUMemBuf>(new CUMemBuf(histBytes));
        auto transformBuf = std::unique_ptr<CUMemBuf>(new CUMemBuf(transformBytes));
        if ((sts = histBuf->alloc()) != RGY_ERR_NONE || (sts = transformBuf->alloc()) != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate hist / transform buffers: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
        m_histBuf = std::move(histBuf);
        m_transformBuf = std::move(transformBuf);
        m_tilesX = tilesX;
        m_tilesY = tilesY;
        m_binBitdepth = binBitdepth;
    }

    prm->frameOut.picstruct = prm->frameIn.picstruct;
    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

tstring NVEncFilterParamClahe::print() const {
    return clahe.print();
}

RGY_ERR NVEncFilterClahe::procPlane(
    RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    int storageBitdepth, int histBitdepth, int tilesX, int tilesY, float slope, cudaStream_t stream) {
    switch (storageBitdepth) {
    case 8:  return clahe_plane_dispatch<uint8_t,   8>(pOutputPlane, pInputPlane, (uint32_t *)m_histBuf->ptr, (uint16_t *)m_transformBuf->ptr, histBitdepth, tilesX, tilesY, slope, stream);
    case 9:  return clahe_plane_dispatch<uint16_t,  9>(pOutputPlane, pInputPlane, (uint32_t *)m_histBuf->ptr, (uint16_t *)m_transformBuf->ptr, histBitdepth, tilesX, tilesY, slope, stream);
    case 10: return clahe_plane_dispatch<uint16_t, 10>(pOutputPlane, pInputPlane, (uint32_t *)m_histBuf->ptr, (uint16_t *)m_transformBuf->ptr, histBitdepth, tilesX, tilesY, slope, stream);
    case 12: return clahe_plane_dispatch<uint16_t, 12>(pOutputPlane, pInputPlane, (uint32_t *)m_histBuf->ptr, (uint16_t *)m_transformBuf->ptr, histBitdepth, tilesX, tilesY, slope, stream);
    case 14: return clahe_plane_dispatch<uint16_t, 14>(pOutputPlane, pInputPlane, (uint32_t *)m_histBuf->ptr, (uint16_t *)m_transformBuf->ptr, histBitdepth, tilesX, tilesY, slope, stream);
    case 16: return clahe_plane_dispatch<uint16_t, 16>(pOutputPlane, pInputPlane, (uint32_t *)m_histBuf->ptr, (uint16_t *)m_transformBuf->ptr, histBitdepth, tilesX, tilesY, slope, stream);
    default: return RGY_ERR_UNSUPPORTED;
    }
}

RGY_ERR NVEncFilterClahe::run_filter(
    const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
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
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamClahe>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }

    const auto planeSrcY = getPlane(pInputFrame, RGY_PLANE_Y);
    auto planeDstY = getPlane(ppOutputFrames[0], RGY_PLANE_Y);
    sts = procPlane(&planeDstY, &planeSrcY,
        RGY_CSP_BIT_DEPTH[pInputFrame->csp], prm->histBitdepth,
        prm->clahe.tiles_x, prm->clahe.tiles_y, prm->clahe.slope, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at clahe(%s): %s.\n"),
            RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
        return sts;
    }

    // CLAHEは輝度のみ処理し、色差とalphaはそのまま渡す。
    for (int i = 1; i < RGY_CSP_PLANES[pInputFrame->csp]; i++) {
        const auto planeSrc = getPlane(pInputFrame, (RGY_PLANE)i);
        auto planeDst = getPlane(ppOutputFrames[0], (RGY_PLANE)i);
        sts = copyPlaneAsync(&planeDst, &planeSrc, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return RGY_ERR_NONE;
}

void NVEncFilterClahe::close() {
    m_frameBuf.clear();
    m_histBuf.reset();
    m_transformBuf.reset();
    m_tilesX = 0;
    m_tilesY = 0;
    m_binBitdepth = 0;
}

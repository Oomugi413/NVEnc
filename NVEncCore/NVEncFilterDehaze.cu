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
#include <cmath>
#include "convert_csp.h"
#include "NVEncFilterDehaze.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static const int DEHAZE_BLOCK_X = 32;
static const int DEHAZE_BLOCK_Y = 8;

// 矩形minの横方向パス。整数minだけなので2次元版とbit-exactになる。
template<typename Type, int bit_depth>
__global__ void kernel_dehaze_min_horizontal(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int width, const int height, const int patchRadius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const Type *srcRow = (const Type *)(pSrc + y * srcPitch);
    Type minValue = (Type)((1 << bit_depth) - 1);
    for (int dx = -patchRadius; dx <= patchRadius; dx++) {
        minValue = min(minValue, srcRow[clamp(x + dx, 0, width - 1)]);
    }
    Type *dstRow = (Type *)(pDst + y * dstPitch);
    dstRow[x] = minValue;
}

// 横minの結果を縦に走査し、透過率の算出と復元を同時に行う。
template<typename Type, int bit_depth>
__global__ void kernel_dehaze(
    uint8_t *__restrict__ pDst, const int dstPitch, const int width, const int height,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const uint8_t *__restrict__ pMinHorizontal, const int minHorizontalPitch,
    const int patchRadius, const float omega, const float tFloor, const float atmosphericLight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    Type minValue = (Type)((1 << bit_depth) - 1);
    for (int dy = -patchRadius; dy <= patchRadius; dy++) {
        const int sy = clamp(y + dy, 0, height - 1);
        const Type *row = (const Type *)(pMinHorizontal + sy * minHorizontalPitch);
        minValue = min(minValue, row[x]);
    }

    const float maxValue = (float)((1 << bit_depth) - 1);
    const float invMax = 1.0f / maxValue;
    const float invA = 1.0f / atmosphericLight;
    const float dark = (float)minValue * invMax * invA;
    const float transmission = fmaxf(1.0f - omega * fminf(dark, 1.0f), tFloor);
    const Type *srcRow = (const Type *)(pSrc + y * srcPitch);
    const float input = (float)srcRow[x] * invMax;
    const float restored = clamp((input - atmosphericLight) / transmission + atmosphericLight, 0.0f, 1.0f);
    Type *dstRow = (Type *)(pDst + y * dstPitch);
    dstRow[x] = (Type)(restored * maxValue + 0.5f);
}

template<typename Type, int bit_depth>
static RGY_ERR dehaze_plane(RGYFrameInfo *pOutputPlane, const RGYFrameInfo *pInputPlane,
    RGYFrameInfo *pMinHorizontal, const VppDehaze& prm, cudaStream_t stream) {
    const dim3 blockSize(DEHAZE_BLOCK_X, DEHAZE_BLOCK_Y);
    const dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));

    kernel_dehaze_min_horizontal<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        pMinHorizontal->ptr[0], pMinHorizontal->pitch[0],
        pInputPlane->ptr[0], pInputPlane->pitch[0],
        pInputPlane->width, pInputPlane->height, prm.patch_radius);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);

    kernel_dehaze<Type, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        pOutputPlane->ptr[0], pOutputPlane->pitch[0], pOutputPlane->width, pOutputPlane->height,
        pInputPlane->ptr[0], pInputPlane->pitch[0],
        pMinHorizontal->ptr[0], pMinHorizontal->pitch[0],
        prm.patch_radius, prm.omega, prm.t_floor, prm.atm_light);
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;
}

NVEncFilterDehaze::NVEncFilterDehaze() : m_minHorizontal() {
    m_name = _T("dehaze");
}

NVEncFilterDehaze::~NVEncFilterDehaze() {
    close();
}

RGY_ERR NVEncFilterDehaze::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDehaze>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U8
        && RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U16) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for dehaze: %s.\n"), RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (!std::isfinite(prm->dehaze.omega) || !std::isfinite(prm->dehaze.t_floor) || !std::isfinite(prm->dehaze.atm_light)) {
        AddMessage(RGY_LOG_ERROR, _T("dehaze parameters must be finite.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->dehaze.patch_radius < 1 || 15 < prm->dehaze.patch_radius) {
        prm->dehaze.patch_radius = clamp(prm->dehaze.patch_radius, 1, 15);
        AddMessage(RGY_LOG_WARN, _T("patch_radius should be in range of %d - %d.\n"), 1, 15);
    }
    if (prm->dehaze.omega < 0.5f || 1.0f < prm->dehaze.omega) {
        prm->dehaze.omega = clamp(prm->dehaze.omega, 0.5f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("omega should be in range of %.2f - %.2f.\n"), 0.5f, 1.0f);
    }
    if (prm->dehaze.t_floor < 0.01f || 0.5f < prm->dehaze.t_floor) {
        prm->dehaze.t_floor = clamp(prm->dehaze.t_floor, 0.01f, 0.5f);
        AddMessage(RGY_LOG_WARN, _T("t_floor should be in range of %.2f - %.2f.\n"), 0.01f, 0.5f);
    }
    if (prm->dehaze.atm_light < 0.1f || 1.0f < prm->dehaze.atm_light) {
        prm->dehaze.atm_light = clamp(prm->dehaze.atm_light, 0.1f, 1.0f);
        AddMessage(RGY_LOG_WARN, _T("atm_light should be in range of %.2f - %.2f.\n"), 0.1f, 1.0f);
    }

    auto sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    const auto frameY = getPlane(&prm->frameOut, RGY_PLANE_Y);
    const auto minCsp = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8) ? RGY_CSP_Y16 : RGY_CSP_Y8;
    if (!m_minHorizontal
        || m_minHorizontal->frame.width != frameY.width
        || m_minHorizontal->frame.height != frameY.height
        || m_minHorizontal->frame.csp != minCsp) {
        m_minHorizontal = std::make_unique<CUFrameBuf>(frameY.width, frameY.height, minCsp);
        m_minHorizontal->releasePtr();
        sts = m_minHorizontal->alloc();
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate dehaze horizontal-min buffer: %s.\n"), get_err_mes(sts));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }

    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

tstring NVEncFilterParamDehaze::print() const {
    return dehaze.print();
}

RGY_ERR NVEncFilterDehaze::procFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDehaze>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;

    auto planeDst = getPlane(pOutputFrame, RGY_PLANE_Y);
    const auto planeSrc = getPlane(pInputFrame, RGY_PLANE_Y);
    auto planeMinHorizontal = getPlane(&m_minHorizontal->frame, RGY_PLANE_Y);
    RGY_ERR sts = RGY_ERR_NONE;
    switch (RGY_CSP_DATA_TYPE[pInputFrame->csp]) {
    case RGY_DATA_TYPE_U8:
        sts = dehaze_plane<uint8_t, 8>(&planeDst, &planeSrc, &planeMinHorizontal, prm->dehaze, stream);
        break;
    case RGY_DATA_TYPE_U16:
        sts = dehaze_plane<uint16_t, 16>(&planeDst, &planeSrc, &planeMinHorizontal, prm->dehaze, stream);
        break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for dehaze: %s.\n"), RGY_CSP_NAMES[pInputFrame->csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (sts != RGY_ERR_NONE) return sts;

    const int copyPlanes = std::min<int>(RGY_CSP_PLANES[pInputFrame->csp], RGY_CSP_PLANES[rgy_csp_no_alpha(pInputFrame->csp)]);
    for (int i = 1; i < copyPlanes; i++) {
        const auto plane = (RGY_PLANE)i;
        auto chromaDst = getPlane(pOutputFrame, plane);
        const auto chromaSrc = getPlane(pInputFrame, plane);
        sts = copyPlaneAsync(&chromaDst, &chromaSrc, stream);
        if (sts != RGY_ERR_NONE) return sts;
    }
    return copyPlaneAlphaAsync(pOutputFrame, pInputFrame, stream);
}

RGY_ERR NVEncFilterDehaze::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames,
    int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) return RGY_ERR_NONE;

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

    const auto sts = procFrame(ppOutputFrames[0], pInputFrame, stream);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("error at dehaze(%s): %s.\n"), RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(sts));
    }
    return sts;
}

void NVEncFilterDehaze::close() {
    m_frameBuf.clear();
    m_minHorizontal.reset();
}

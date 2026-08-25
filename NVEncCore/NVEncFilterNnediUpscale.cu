// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include "NVEncFilterNnediUpscale.h"
#include "NVEncFilterTransform.h"
#include "rgy_cuda_util_kernel.h"
#include <climits>

#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

namespace {

template<typename Type>
static __device__ __forceinline__ float nnediUpscaleRead(
    const uint8_t *__restrict__ pSrc, const int srcPitch, const int srcW, const int srcH,
    const int x, const int y) {
    const int sx = min(max(x, 0), srcW - 1);
    const int sy = min(max(y, 0), srcH - 1);
    return (float)(*(const Type *)(pSrc + (size_t)sy * srcPitch + (size_t)sx * sizeof(Type)));
}

template<typename Type, bool shiftCubic>
__global__ void kernel_nnedi_upscale_shift_cuda(
    uint8_t *__restrict__ pDst, const int dstPitch,
    const uint8_t *__restrict__ pSrc, const int srcPitch,
    const int srcW, const int srcH, const int bitDepth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= srcW || y >= srcH) {
        return;
    }

    float v;
    if (shiftCubic) {
        float col[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int xx = x - 2 + k;
            col[k] = -0.0625f * nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, xx, y - 2)
                   +  0.5625f * nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, xx, y - 1)
                   +  0.5625f * nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, xx, y    )
                   -  0.0625f * nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, xx, y + 1);
        }
        v = -0.0625f * col[0] + 0.5625f * col[1] + 0.5625f * col[2] - 0.0625f * col[3];
    } else {
        v = 0.25f * (
            nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, x,     y)
            + nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, x - 1, y)
            + nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, x,     y - 1)
            + nnediUpscaleRead<Type>(pSrc, srcPitch, srcW, srcH, x - 1, y - 1));
    }

    const float maxValue = (float)((1 << bitDepth) - 1);
    *(Type *)(pDst + (size_t)y * dstPitch + (size_t)x * sizeof(Type))
        = (Type)fminf(fmaxf(v + 0.5f, 0.0f), maxValue);
}

template<typename Type, bool shiftCubic>
RGY_ERR launchNnediUpscaleShift(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
    const int bitDepth, cudaStream_t stream) {
    const dim3 blockSize(32, 8);
    const dim3 gridSize(divCeil(pSrc->width, (int)blockSize.x), divCeil(pSrc->height, (int)blockSize.y));
    kernel_nnedi_upscale_shift_cuda<Type, shiftCubic><<<gridSize, blockSize, 0, stream>>>(
        pDst->ptr[0], pDst->pitch[0], pSrc->ptr[0], pSrc->pitch[0],
        pSrc->width, pSrc->height, bitDepth);
    const auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return err_to_rgy(cudaerr);
    }
    CUDA_DEBUG_SYNC_ERR;
    return RGY_ERR_NONE;
}

} // namespace

NVEncFilterParamNnediUpscale::NVEncFilterParamNnediUpscale() :
    nnediUpscale(),
    compute_capability(std::make_pair(0, 0)),
    hModule(NULL),
    timebase() {
}

NVEncFilterNnediUpscale::NVEncFilterNnediUpscale() :
    NVEncFilter(), m_passV(), m_passH(), m_transposeToH(), m_transposeBack() {
    m_name = _T("nnedi-upscale");
}

NVEncFilterNnediUpscale::~NVEncFilterNnediUpscale() {
    close();
}

RGY_ERR NVEncFilterNnediUpscale::makeNnedi(std::unique_ptr<NVEncFilterNnedi>& target,
    const RGYFrameInfo& frameIn, const NVEncFilterParamNnediUpscale *prm, const TCHAR *label) {
    auto param = std::make_shared<NVEncFilterParamNnedi>();
    param->frameIn = frameIn;
    param->frameOut = frameIn;
    param->baseFps = prm->baseFps;
    param->timebase = prm->timebase;
    param->compute_capability = prm->compute_capability;
    param->hModule = prm->hModule;
    param->bOutOverwrite = false;
    param->nnedi.enable = true;
    // プログレッシブ画像の既存行を上フィールドとして扱い、空いた行を補間する。
    param->nnedi.field = VPP_NNEDI_FIELD_TOP;
    param->nnedi.doubleHeight = true;
    // doubleHeightでは全プレーンを同じ倍率で処理する。
    param->nnedi.processPlane = { true, true, true, true };
    param->nnedi.nsize = prm->nnediUpscale.nnedi.nsize;
    param->nnedi.nns = prm->nnediUpscale.nnedi.nns;
    param->nnedi.quality = prm->nnediUpscale.nnedi.quality;
    param->nnedi.prescreen = prm->nnediUpscale.nnedi.prescreen;
    param->nnedi.errortype = prm->nnediUpscale.nnedi.errortype;
    param->nnedi.clamp = prm->nnediUpscale.nnedi.clamp;
    param->nnedi.weightfile = prm->nnediUpscale.nnedi.weightfile;
    target = std::make_unique<NVEncFilterNnedi>();
    const auto err = target->init(param, m_pLog);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialise the %s doubling pass: %s.\n"),
            label, get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterNnediUpscale::makeTranspose(std::unique_ptr<NVEncFilterTransform>& target,
    const RGYFrameInfo& frameIn, const NVEncFilterParamNnediUpscale *prm, const TCHAR *label) {
    auto param = std::make_shared<NVEncFilterParamTransform>();
    param->frameIn = frameIn;
    param->frameOut = frameIn;
    param->baseFps = prm->baseFps;
    param->bOutOverwrite = false;
    param->trans.transpose = true;
    target = std::make_unique<NVEncFilterTransform>();
    const auto err = target->init(param, m_pLog);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to initialise the %s transpose: %s.\n"),
            label, get_err_mes(err));
    }
    return err;
}

RGY_ERR NVEncFilterNnediUpscale::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnediUpscale>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameIn.height <= 0 || prm->frameIn.width <= 0
        || prm->frameIn.height > INT_MAX / 2 || prm->frameIn.width > INT_MAX / 2) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    // 4:2:2は転置すると4:4:0相当になり、同じCSPの中間フレームでは表現できない。
    if (RGY_CSP_CHROMA_FORMAT[prm->frameIn.csp] == RGY_CHROMAFMT_YUV422) {
        AddMessage(RGY_LOG_ERROR, _T("nnedi-upscale does not support 4:2:2 input.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    // CUDA版transformとNNEDIの双方で扱えるplanar形式に限定する。
    switch (prm->frameIn.csp) {
    case RGY_CSP_Y8:
    case RGY_CSP_Y16:
    case RGY_CSP_YV12:
    case RGY_CSP_YV12_16:
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_16:
        break;
    default:
        AddMessage(RGY_LOG_ERROR, _T("nnedi-upscale does not support csp %s.\n"), RGY_CSP_NAMES[prm->frameIn.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->frameIn.picstruct & RGY_PICSTRUCT_INTERLACED) {
        AddMessage(RGY_LOG_ERROR, _T("nnedi-upscale requires progressive input; deinterlace first.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    // 各軸をフィールドとして2分するため、縦横とも偶数を要求する。
    if ((prm->frameIn.width & 1) || (prm->frameIn.height & 1)) {
        AddMessage(RGY_LOG_ERROR, _T("nnedi-upscale needs even width and height (got %dx%d).\n"),
            prm->frameIn.width, prm->frameIn.height);
        return RGY_ERR_INVALID_PARAM;
    }

    const auto width = prm->frameIn.width;
    const auto height = prm->frameIn.height;

    auto mid = prm->frameIn;
    mid.width = width;
    mid.height = height * 2;

    auto midT = prm->frameIn;
    midT.width = height * 2;
    midT.height = width;

    auto midH = prm->frameIn;
    midH.width = height * 2;
    midH.height = width * 2;

    auto sts = makeNnedi(m_passV, prm->frameIn, prm.get(), _T("vertical"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = makeTranspose(m_transposeToH, mid, prm.get(), _T("first"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = makeNnedi(m_passH, midT, prm.get(), _T("horizontal"));
    if (sts != RGY_ERR_NONE) return sts;
    sts = makeTranspose(m_transposeBack, midH, prm.get(), _T("second"));
    if (sts != RGY_ERR_NONE) return sts;

    prm->frameOut = prm->frameIn;
    prm->frameOut.width = width * 2;
    prm->frameOut.height = height * 2;
    prm->frameOut.picstruct = RGY_PICSTRUCT_FRAME;

    sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory: %s.\n"), get_err_mes(sts));
        return RGY_ERR_MEMORY_ALLOC;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }

    m_pathThrough &= ~FILTER_PATHTHROUGH_PICSTRUCT;
    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNnediUpscale::halfPixel(RGYFrameInfo *pDst, const RGYFrameInfo *pSrc,
    cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamNnediUpscale>(m_param);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[pSrc->csp];
    const int nPlanes = RGY_CSP_PLANES[pSrc->csp];
    for (int i = 0; i < nPlanes; i++) {
        auto planeDst = getPlane(pDst, (RGY_PLANE)i);
        const auto planeSrc = getPlane(pSrc, (RGY_PLANE)i);
        RGY_ERR err;
        if (bitDepth > 8) {
            err = prm->nnediUpscale.shiftCubic
                ? launchNnediUpscaleShift<uint16_t, true>(&planeDst, &planeSrc, bitDepth, stream)
                : launchNnediUpscaleShift<uint16_t, false>(&planeDst, &planeSrc, bitDepth, stream);
        } else {
            err = prm->nnediUpscale.shiftCubic
                ? launchNnediUpscaleShift<uint8_t, true>(&planeDst, &planeSrc, bitDepth, stream)
                : launchNnediUpscaleShift<uint8_t, false>(&planeDst, &planeSrc, bitDepth, stream);
        }
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at kernel_nnedi_upscale_shift_cuda: %s.\n"), get_err_mes(err));
            return err;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterNnediUpscale::run_filter(const RGYFrameInfo *pInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    *pOutputFrameNum = 0;
    if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[0]->frame;
    }
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;

    const auto memcpyKind = getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    // 拡大処理では入力のフィールド情報に関係なく各行をプログレッシブとして扱う。
    auto srcFrame = *pInputFrame;
    srcFrame.picstruct = RGY_PICSTRUCT_FRAME;

    int count = 0;
    RGYFrameInfo *outV[2] = { nullptr, nullptr };
    auto sts = m_passV->filter(&srcFrame, outV, &count, stream);
    if (sts != RGY_ERR_NONE || count != 1 || outV[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the vertical doubling failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }

    count = 0;
    RGYFrameInfo *outTransposedV[1] = { nullptr };
    sts = m_transposeToH->filter(outV[0], outTransposedV, &count, stream);
    if (sts != RGY_ERR_NONE || count != 1 || outTransposedV[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the first transpose failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }

    auto midFrame = *outTransposedV[0];
    midFrame.picstruct = RGY_PICSTRUCT_FRAME;
    count = 0;
    RGYFrameInfo *outH[2] = { nullptr, nullptr };
    sts = m_passH->filter(&midFrame, outH, &count, stream);
    if (sts != RGY_ERR_NONE || count != 1 || outH[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the horizontal doubling failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }

    count = 0;
    RGYFrameInfo *outTransposedH[1] = { nullptr };
    sts = m_transposeBack->filter(outH[0], outTransposedH, &count, stream);
    if (sts != RGY_ERR_NONE || count != 1 || outTransposedH[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("the second transpose failed: %s.\n"), get_err_mes(sts));
        return sts != RGY_ERR_NONE ? sts : RGY_ERR_UNKNOWN;
    }

    // 各パスが各軸に残す半画素ずれを最後にまとめて補正する。
    sts = halfPixel(ppOutputFrames[0], outTransposedH[0], stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }

    copyFramePropWithoutRes(ppOutputFrames[0], pInputFrame);
    ppOutputFrames[0]->picstruct = RGY_PICSTRUCT_FRAME;
    return RGY_ERR_NONE;
}

void NVEncFilterNnediUpscale::close() {
    m_passV.reset();
    m_passH.reset();
    m_transposeToH.reset();
    m_transposeBack.reset();
    m_frameBuf.clear();
}

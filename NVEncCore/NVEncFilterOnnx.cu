// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterOnnx.h"
#include "rgy_cuda_util_kernel.h"

template<typename Type, int bit_depth>
__global__ void kernel_onnx_pack_luma(float *__restrict__ dst,
    const uint8_t *__restrict__ src, const int srcPitch, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const auto srcRow = (const Type *)(src + (size_t)y * srcPitch);
        dst[(size_t)y * width + x] = srcRow[x] * (1.0f / ((1 << bit_depth) - 1));
    }
}

template<typename Type, int bit_depth>
__global__ void kernel_onnx_unpack_luma(uint8_t *__restrict__ dst, const int dstPitch,
    const float *__restrict__ src, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        constexpr int maxValue = (1 << bit_depth) - 1;
        const int value = (int)(src[(size_t)y * width + x] * maxValue + 0.5f);
        auto dstRow = (Type *)(dst + (size_t)y * dstPitch);
        dstRow[x] = (Type)max(0, min(value, maxValue));
    }
}

RGY_ERR run_onnx_pack_luma(float *dst, const RGYFrameInfo *src, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(src->width, (int)block.x), divCeil(src->height, (int)block.y));
    switch (src->csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12:    kernel_onnx_pack_luma<uint8_t,  8><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_09: kernel_onnx_pack_luma<uint16_t, 9><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_10: kernel_onnx_pack_luma<uint16_t,10><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_12: kernel_onnx_pack_luma<uint16_t,12><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_YV12_14: kernel_onnx_pack_luma<uint16_t,14><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    case RGY_CSP_P010:
    case RGY_CSP_YV12_16: kernel_onnx_pack_luma<uint16_t,16><<<grid, block, 0, stream>>>(dst, src->ptr[0], src->pitch[0], src->width, src->height); break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    return err_to_rgy(cudaGetLastError());
}

RGY_ERR run_onnx_unpack_luma(RGYFrameInfo *dst, const float *src, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid(divCeil(dst->width, (int)block.x), divCeil(dst->height, (int)block.y));
    switch (dst->csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_YV12:    kernel_onnx_unpack_luma<uint8_t,  8><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_09: kernel_onnx_unpack_luma<uint16_t, 9><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_10: kernel_onnx_unpack_luma<uint16_t,10><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_12: kernel_onnx_unpack_luma<uint16_t,12><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_YV12_14: kernel_onnx_unpack_luma<uint16_t,14><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    case RGY_CSP_P010:
    case RGY_CSP_YV12_16: kernel_onnx_unpack_luma<uint16_t,16><<<grid, block, 0, stream>>>(dst->ptr[0], dst->pitch[0], src, dst->width, dst->height); break;
    default: return RGY_ERR_UNSUPPORTED;
    }
    return err_to_rgy(cudaGetLastError());
}

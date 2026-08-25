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

#include <algorithm>
#include <climits>
#include <cmath>
#include "NVEncFilterDenoiseBm3d.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util_kernel.h"

static constexpr int BM3D_BLOCK_SIZE = 8;
static constexpr int BM3D_MAX_GROUP_SIZE_BASIC = 16;
static constexpr int BM3D_MAX_GROUP_SIZE_WIENER = 32;
static constexpr int BM3D_MAX_GROUP_SIZE_TEMPORAL = 16;

static constexpr float BM3D_DCT_C3A = 0.83146961230254523707878837761791f;
static constexpr float BM3D_DCT_C3B = 0.55557023301960222474283081394853f;
static constexpr float BM3D_DCT_C1A = 0.98078528040323044912618223613424f;
static constexpr float BM3D_DCT_C1B = 0.19509032201612826784828486847702f;
static constexpr float BM3D_DCT_S2C3A = 0.54119610014619698439972320536639f;
static constexpr float BM3D_DCT_S2C3B = 1.3065629648763765278566431734272f;
static constexpr float BM3D_DCT_NORM_2D = 0.125f;
static constexpr float BM3D_DCT_SQRT2 = 1.4142135623730950488016887242097f;
static constexpr float BM3D_HAAR_INV_SQRT2 = 0.70710678118654752440084436210485f;

__device__ __constant__ float c_bm3dKaiser[BM3D_BLOCK_SIZE * BM3D_BLOCK_SIZE] = {
    0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f,
    0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f,
    0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f,
    0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f,
    0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f,
    0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f,
    0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f,
    0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f
};

template<typename Type>
__device__ __forceinline__ float bm3dReadPixel(const uint8_t *src, int pitch, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return (float)((const Type *)(src + (size_t)y * pitch))[x];
}

__device__ __forceinline__ float bm3dReadPixelF32(const float *src, int pitch, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return ((const float *)((const uint8_t *)src + (size_t)y * pitch))[x];
}

__device__ __forceinline__ void bm3dDct8(float v[8]) {
    float s[8], t[2], tmp;
    s[0] = v[0] + v[7]; s[1] = v[1] + v[6]; s[2] = v[2] + v[5]; s[3] = v[3] + v[4];
    s[4] = v[3] - v[4]; s[5] = v[2] - v[5]; s[6] = v[1] - v[6]; s[7] = v[0] - v[7];
    const float u0 = s[0] + s[3];
    const float u1 = s[1] + s[2];
    const float u2 = s[1] - s[2];
    const float u3 = s[0] - s[3];
    tmp = BM3D_DCT_C3A * (s[4] + s[7]);
    const float u4 = tmp + (BM3D_DCT_C3B - BM3D_DCT_C3A) * s[7];
    const float u7 = tmp - (BM3D_DCT_C3A + BM3D_DCT_C3B) * s[4];
    tmp = BM3D_DCT_C1A * (s[5] + s[6]);
    const float u5 = tmp + (BM3D_DCT_C1B - BM3D_DCT_C1A) * s[6];
    const float u6 = tmp - (BM3D_DCT_C1A + BM3D_DCT_C1B) * s[5];
    v[0] = u0 + u1;
    v[4] = u0 - u1;
    tmp = BM3D_DCT_S2C3A * (u2 + u3);
    v[2] = tmp + (BM3D_DCT_S2C3B - BM3D_DCT_S2C3A) * u3;
    v[6] = tmp - (BM3D_DCT_S2C3A + BM3D_DCT_S2C3B) * u2;
    t[0] = u4 + u6;
    t[1] = u5 + u7;
    v[3] = (u7 - u5) * BM3D_DCT_SQRT2;
    v[5] = (u4 - u6) * BM3D_DCT_SQRT2;
    v[1] = t[0] + t[1];
    v[7] = t[1] - t[0];
}

__device__ __forceinline__ void bm3dIdct8(float v[8]) {
    float r[8], s4[2], tmp;
    s4[0] = v[1] - v[7];
    const float st5 = v[3] * BM3D_DCT_SQRT2;
    const float st6 = v[5] * BM3D_DCT_SQRT2;
    s4[1] = v[1] + v[7];
    r[0] = v[0] + v[4]; r[1] = v[0] - v[4];
    tmp = BM3D_DCT_S2C3A * (v[2] + v[6]);
    r[2] = tmp - (BM3D_DCT_S2C3A + BM3D_DCT_S2C3B) * v[6];
    r[3] = tmp + (BM3D_DCT_S2C3B - BM3D_DCT_S2C3A) * v[2];
    r[4] = s4[0] + st6; r[5] = s4[1] - st5; r[6] = s4[0] - st6; r[7] = st5 + s4[1];
    float w[8];
    w[0] = r[0] + r[3]; w[1] = r[1] + r[2]; w[2] = r[1] - r[2]; w[3] = r[0] - r[3];
    tmp = BM3D_DCT_C3A * (r[4] + r[7]);
    w[4] = tmp - (BM3D_DCT_C3A + BM3D_DCT_C3B) * r[7];
    w[7] = tmp + (BM3D_DCT_C3B - BM3D_DCT_C3A) * r[4];
    tmp = BM3D_DCT_C1A * (r[5] + r[6]);
    w[5] = tmp - (BM3D_DCT_C1A + BM3D_DCT_C1B) * r[6];
    w[6] = tmp + (BM3D_DCT_C1B - BM3D_DCT_C1A) * r[5];
    v[0] = w[0] + w[7]; v[1] = w[1] + w[6]; v[2] = w[2] + w[5]; v[3] = w[3] + w[4];
    v[4] = w[3] - w[4]; v[5] = w[2] - w[5]; v[6] = w[1] - w[6]; v[7] = w[0] - w[7];
}

__device__ __forceinline__ void bm3dDct2d(float block[8][8]) {
#pragma unroll
    for (int j = 0; j < 8; j++) bm3dDct8(block[j]);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        float col[8];
#pragma unroll
        for (int j = 0; j < 8; j++) col[j] = block[j][i];
        bm3dDct8(col);
#pragma unroll
        for (int j = 0; j < 8; j++) block[j][i] = col[j] * BM3D_DCT_NORM_2D;
    }
}

__device__ __forceinline__ void bm3dIdct2d(float block[8][8]) {
#pragma unroll
    for (int j = 0; j < 8; j++) bm3dIdct8(block[j]);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        float col[8];
#pragma unroll
        for (int j = 0; j < 8; j++) col[j] = block[j][i];
        bm3dIdct8(col);
#pragma unroll
        for (int j = 0; j < 8; j++) block[j][i] = col[j] * BM3D_DCT_NORM_2D;
    }
}

__device__ __forceinline__ void bm3dHaar8(float x[8], float y[8]) {
    int count = 8;
#pragma unroll
    for (int j = 0; j < 3; j++) {
        const int prevCount = count;
        count >>= 1;
        for (int i = 0; i < count; i++) {
            const int i2 = i << 1;
            y[i] = (x[i2] + x[i2 + 1]) * BM3D_HAAR_INV_SQRT2;
            y[i + count] = (x[i2] - x[i2 + 1]) * BM3D_HAAR_INV_SQRT2;
        }
        for (int i = 0; i < prevCount; i++) x[i] = y[i];
    }
}

__device__ __forceinline__ void bm3dIHaar8(float x[8], float y[8]) {
    int count = 1;
#pragma unroll
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < count; i++) {
            const int i2 = i << 1;
            y[i2] = (x[i] + x[i + count]) * BM3D_HAAR_INV_SQRT2;
            y[i2 + 1] = (x[i] - x[i + count]) * BM3D_HAAR_INV_SQRT2;
        }
        count <<= 1;
        for (int i = 0; i < count; i++) x[i] = y[i];
    }
}

template<typename Type, bool temporal>
__global__ void kernelBm3dMatch(const uint8_t *src, int srcPitch,
    const uint8_t *noisyRing, int noisyRingPitch, int noisyRingSlotStride,
    int ringCursor, int ringRadius, int ringFilled,
    int width, int height, int refCountX, int refCountY,
    short *similarCoords, uint8_t *similarFrameIdx, uint8_t *blockCounts,
    int blockStep, int bmRange, int groupSize, int distThreshold) {
    const int rgx = blockIdx.x * blockDim.x + threadIdx.x;
    const int rgy = blockIdx.y * blockDim.y + threadIdx.y;
    if (rgx >= refCountX || rgy >= refCountY) return;
    const int rx = rgx * blockStep;
    const int ry = rgy * blockStep;
    const int refId = rgy * refCountX + rgx;
    float ref[8][8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int i = 0; i < 8; i++) ref[j][i] = bm3dReadPixel<Type>(src, srcPitch, rx + i, ry + j, width, height);
    }
    int distances[BM3D_MAX_GROUP_SIZE_BASIC];
    short positionsX[BM3D_MAX_GROUP_SIZE_BASIC];
    short positionsY[BM3D_MAX_GROUP_SIZE_BASIC];
    uint8_t positionsF[BM3D_MAX_GROUP_SIZE_BASIC];
#pragma unroll
    for (int n = 0; n < BM3D_MAX_GROUP_SIZE_BASIC; n++) {
        distances[n] = INT_MAX; positionsX[n] = 0; positionsY[n] = 0; positionsF[n] = 0;
    }
    int count = 0;
    const int groupCap = min(groupSize, BM3D_MAX_GROUP_SIZE_BASIC);
    const int temporalCount = temporal ? ringFilled : 0;
    for (int back = 0; back <= temporalCount; back++) {
        const int slot = (back > 0) ? (ringCursor + ringRadius - back) % ringRadius : 0;
        const uint8_t *frameSrc = (back > 0) ? noisyRing + (size_t)slot * noisyRingSlotStride : src;
        const int framePitch = (back > 0) ? noisyRingPitch : srcPitch;
        for (int wy = -bmRange; wy <= bmRange; wy++) {
            for (int wx = -bmRange; wx <= bmRange; wx++) {
                int distance = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) {
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float diff = ref[j][i] - bm3dReadPixel<Type>(frameSrc, framePitch, rx + wx + i, ry + wy + j, width, height);
                        distance += (int)(diff * diff);
                    }
                }
                if (distance > distThreshold) continue;
                for (int n = 0; n < groupCap; n++) {
                    if (distance < distances[n]) {
                        for (int k = groupCap - 1; k > n; k--) {
                            distances[k] = distances[k - 1]; positionsX[k] = positionsX[k - 1];
                            positionsY[k] = positionsY[k - 1]; positionsF[k] = positionsF[k - 1];
                        }
                        distances[n] = distance; positionsX[n] = (short)wx; positionsY[n] = (short)wy; positionsF[n] = (uint8_t)back;
                        if (count < groupCap) count++;
                        break;
                    }
                }
            }
        }
    }
    blockCounts[refId] = (uint8_t)count;
    const int base = refId * groupCap * 2;
    for (int n = 0; n < count; n++) {
        similarCoords[base + n * 2] = positionsX[n];
        similarCoords[base + n * 2 + 1] = positionsY[n];
        if (temporal) similarFrameIdx[refId * groupCap + n] = positionsF[n];
    }
}

template<typename Type, bool temporal>
__global__ void kernelBm3dBasic(const uint8_t *src, int srcPitch,
    const uint8_t *noisyRing, int noisyRingPitch, int noisyRingSlotStride,
    int ringCursor, int ringRadius,
    int width, int height, int refCountX, int refCountY,
    const short *similarCoords, const uint8_t *similarFrameIdx, const uint8_t *blockCounts,
    float *accumulator, int accPitch, float *weightMap, int weightPitch,
    int blockStep, int groupSize, float sigmaScaled, float tau) {
    const int rgx = blockIdx.x * blockDim.x + threadIdx.x;
    const int rgy = blockIdx.y * blockDim.y + threadIdx.y;
    if (rgx >= refCountX || rgy >= refCountY) return;
    const int rx = rgx * blockStep;
    const int ry = rgy * blockStep;
    const int refId = rgy * refCountX + rgx;
    const int groupCap = min(groupSize, BM3D_MAX_GROUP_SIZE_BASIC);
    const int blockCount = (int)blockCounts[refId];
    if (blockCount == 0) return;
    float stack[BM3D_MAX_GROUP_SIZE_BASIC][8][8];
    const int base = refId * groupCap * 2;
    for (int n = 0; n < blockCount; n++) {
        const int sx = (int)similarCoords[base + n * 2];
        const int sy = (int)similarCoords[base + n * 2 + 1];
        const int back = temporal ? (int)similarFrameIdx[refId * groupCap + n] : 0;
        const int slot = (back > 0) ? (ringCursor + ringRadius - back) % ringRadius : 0;
        const uint8_t *frameSrc = (back > 0) ? noisyRing + (size_t)slot * noisyRingSlotStride : src;
        const int framePitch = (back > 0) ? noisyRingPitch : srcPitch;
#pragma unroll
        for (int j = 0; j < 8; j++) {
#pragma unroll
            for (int i = 0; i < 8; i++) stack[n][j][i] = bm3dReadPixel<Type>(frameSrc, framePitch, rx + sx + i, ry + sy + j, width, height);
        }
        bm3dDct2d(stack[n]);
    }
    int retained = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int left = blockCount;
            int slabIndex = 0;
            while (left > 0) {
                float slab[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                float transformed[8];
                const int take = min(left, 8);
                for (int n = 0; n < take; n++) slab[n] = stack[slabIndex * 8 + n][j][i];
                bm3dHaar8(slab, transformed);
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    if (fabsf(transformed[n]) <= tau) transformed[n] = 0.0f; else retained++;
                }
                bm3dIHaar8(transformed, slab);
                for (int n = 0; n < take; n++) stack[slabIndex * 8 + n][j][i] = slab[n];
                slabIndex++;
                left -= 8;
            }
        }
    }
    const float groupWeight = (retained >= 1) ? 1.0f / (sigmaScaled * sigmaScaled * (float)retained) : 1.0f;
    for (int n = 0; n < blockCount; n++) {
        bm3dIdct2d(stack[n]);
        if (temporal && similarFrameIdx[refId * groupCap + n] != 0) continue;
        const int sx = (int)similarCoords[base + n * 2];
        const int sy = (int)similarCoords[base + n * 2 + 1];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int py = ry + sy + j;
            if (py < 0 || py >= height) continue;
            float *accRow = (float *)((uint8_t *)accumulator + (size_t)py * accPitch);
            float *weightRow = (float *)((uint8_t *)weightMap + (size_t)py * weightPitch);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int px = rx + sx + i;
                if (px < 0 || px >= width) continue;
                const float pixelWeight = groupWeight * c_bm3dKaiser[j * 8 + i];
                atomicAdd(&accRow[px], stack[n][j][i] * pixelWeight);
                atomicAdd(&weightRow[px], pixelWeight);
            }
        }
    }
}

template<typename Type>
__global__ void kernelBm3dNormalizeF32(float *dst, int dstPitch, const uint8_t *src, int srcPitch,
    int width, int height, const float *accumulator, int accPitch, const float *weightMap, int weightPitch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float *accRow = (const float *)((const uint8_t *)accumulator + (size_t)y * accPitch);
    const float *weightRow = (const float *)((const uint8_t *)weightMap + (size_t)y * weightPitch);
    float *dstRow = (float *)((uint8_t *)dst + (size_t)y * dstPitch);
    const float weight = weightRow[x];
    dstRow[x] = (weight <= 0.0f) ? (float)((const Type *)(src + (size_t)y * srcPitch))[x] : accRow[x] / weight;
}

template<bool temporal>
__global__ void kernelBm3dMatchBasic(const float *basic, int basicPitch,
    const float *basicRing, int basicRingPitch, int basicRingSlotStride,
    int ringCursor, int ringRadius, int ringFilled,
    int width, int height, int refCountX, int refCountY,
    short *similarCoords, uint8_t *similarFrameIdx, uint8_t *blockCounts,
    int blockStep, int bmRange, int groupSize, float distThreshold) {
    const int rgx = blockIdx.x * blockDim.x + threadIdx.x;
    const int rgy = blockIdx.y * blockDim.y + threadIdx.y;
    if (rgx >= refCountX || rgy >= refCountY) return;
    const int rx = rgx * blockStep;
    const int ry = rgy * blockStep;
    const int refId = rgy * refCountX + rgx;
    float ref[8][8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int i = 0; i < 8; i++) ref[j][i] = bm3dReadPixelF32(basic, basicPitch, rx + i, ry + j, width, height);
    }
    float distances[BM3D_MAX_GROUP_SIZE_WIENER];
    short positionsX[BM3D_MAX_GROUP_SIZE_WIENER];
    short positionsY[BM3D_MAX_GROUP_SIZE_WIENER];
    uint8_t positionsF[BM3D_MAX_GROUP_SIZE_WIENER];
#pragma unroll
    for (int n = 0; n < BM3D_MAX_GROUP_SIZE_WIENER; n++) {
        distances[n] = 1.0e30f; positionsX[n] = 0; positionsY[n] = 0; positionsF[n] = 0;
    }
    int count = 0;
    const int groupCap = min(groupSize, BM3D_MAX_GROUP_SIZE_WIENER);
    const int temporalCount = temporal ? ringFilled : 0;
    for (int back = 0; back <= temporalCount; back++) {
        const int slot = (back > 0) ? (ringCursor + ringRadius - back) % ringRadius : 0;
        const float *frameSrc = (back > 0)
            ? (const float *)((const uint8_t *)basicRing + (size_t)slot * basicRingSlotStride)
            : basic;
        const int framePitch = (back > 0) ? basicRingPitch : basicPitch;
        for (int wy = -bmRange; wy <= bmRange; wy++) {
            for (int wx = -bmRange; wx <= bmRange; wx++) {
                float distance = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; j++) {
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float diff = ref[j][i] - bm3dReadPixelF32(frameSrc, framePitch, rx + wx + i, ry + wy + j, width, height);
                        distance += diff * diff;
                    }
                }
                if (distance > distThreshold) continue;
                for (int n = 0; n < groupCap; n++) {
                    if (distance < distances[n]) {
                        for (int k = groupCap - 1; k > n; k--) {
                            distances[k] = distances[k - 1]; positionsX[k] = positionsX[k - 1];
                            positionsY[k] = positionsY[k - 1]; positionsF[k] = positionsF[k - 1];
                        }
                        distances[n] = distance; positionsX[n] = (short)wx; positionsY[n] = (short)wy; positionsF[n] = (uint8_t)back;
                        if (count < groupCap) count++;
                        break;
                    }
                }
            }
        }
    }
    blockCounts[refId] = (uint8_t)count;
    const int base = refId * groupCap * 2;
    for (int n = 0; n < count; n++) {
        similarCoords[base + n * 2] = positionsX[n];
        similarCoords[base + n * 2 + 1] = positionsY[n];
        if (temporal) similarFrameIdx[refId * groupCap + n] = positionsF[n];
    }
}

template<typename Type, bool temporal>
__global__ void kernelBm3dWiener(const uint8_t *src, int srcPitch,
    const uint8_t *noisyRing, int noisyRingPitch, int noisyRingSlotStride,
    const float *basic, int basicPitch,
    const float *basicRing, int basicRingPitch, int basicRingSlotStride,
    int ringCursor, int ringRadius,
    int width, int height, int refCountX, int refCountY,
    const short *similarCoords, const uint8_t *similarFrameIdx, const uint8_t *blockCounts,
    float *accumulator, int accPitch, float *weightMap, int weightPitch,
    int blockStep, int groupSize, float sigmaScaled) {
    const int rgx = blockIdx.x * blockDim.x + threadIdx.x;
    const int rgy = blockIdx.y * blockDim.y + threadIdx.y;
    if (rgx >= refCountX || rgy >= refCountY) return;
    const int rx = rgx * blockStep;
    const int ry = rgy * blockStep;
    const int refId = rgy * refCountX + rgx;
    const int groupCap = min(groupSize, BM3D_MAX_GROUP_SIZE_WIENER);
    const int blockCount = (int)blockCounts[refId];
    if (blockCount == 0) return;
    float noiseStack[BM3D_MAX_GROUP_SIZE_WIENER][8][8];
    float basicStack[BM3D_MAX_GROUP_SIZE_WIENER][8][8];
    const int base = refId * groupCap * 2;
    for (int n = 0; n < blockCount; n++) {
        const int sx = (int)similarCoords[base + n * 2];
        const int sy = (int)similarCoords[base + n * 2 + 1];
        const int back = temporal ? (int)similarFrameIdx[refId * groupCap + n] : 0;
        const int slot = (back > 0) ? (ringCursor + ringRadius - back) % ringRadius : 0;
        const uint8_t *noiseSrc = (back > 0) ? noisyRing + (size_t)slot * noisyRingSlotStride : src;
        const int noisePitch = (back > 0) ? noisyRingPitch : srcPitch;
        const float *basicSrc = (back > 0)
            ? (const float *)((const uint8_t *)basicRing + (size_t)slot * basicRingSlotStride)
            : basic;
        const int basicSrcPitch = (back > 0) ? basicRingPitch : basicPitch;
#pragma unroll
        for (int j = 0; j < 8; j++) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                noiseStack[n][j][i] = bm3dReadPixel<Type>(noiseSrc, noisePitch, rx + sx + i, ry + sy + j, width, height);
                basicStack[n][j][i] = bm3dReadPixelF32(basicSrc, basicSrcPitch, rx + sx + i, ry + sy + j, width, height);
            }
        }
        bm3dDct2d(noiseStack[n]);
        bm3dDct2d(basicStack[n]);
    }
    const float sigmaSq = sigmaScaled * sigmaScaled;
    float sumSqrWeights = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int left = blockCount;
            int slabIndex = 0;
            while (left > 0) {
                float noiseSlab[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                float basicSlab[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                float transformedNoise[8], transformedBasic[8];
                const int take = min(left, 8);
                for (int n = 0; n < take; n++) {
                    noiseSlab[n] = noiseStack[slabIndex * 8 + n][j][i];
                    basicSlab[n] = basicStack[slabIndex * 8 + n][j][i];
                }
                bm3dHaar8(noiseSlab, transformedNoise);
                bm3dHaar8(basicSlab, transformedBasic);
                float filtered[8];
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    const float basicSq = transformedBasic[n] * transformedBasic[n];
                    const float weight = basicSq / (basicSq + sigmaSq);
                    sumSqrWeights += weight * weight;
                    filtered[n] = weight * transformedNoise[n];
                }
                float output[8];
                bm3dIHaar8(filtered, output);
                for (int n = 0; n < take; n++) noiseStack[slabIndex * 8 + n][j][i] = output[n];
                slabIndex++;
                left -= 8;
            }
        }
    }
    const float groupWeight = (sumSqrWeights > 1.0e-12f) ? 1.0f / (sigmaSq * sumSqrWeights) : 1.0f;
    for (int n = 0; n < blockCount; n++) {
        bm3dIdct2d(noiseStack[n]);
        if (temporal && similarFrameIdx[refId * groupCap + n] != 0) continue;
        const int sx = (int)similarCoords[base + n * 2];
        const int sy = (int)similarCoords[base + n * 2 + 1];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int py = ry + sy + j;
            if (py < 0 || py >= height) continue;
            float *accRow = (float *)((uint8_t *)accumulator + (size_t)py * accPitch);
            float *weightRow = (float *)((uint8_t *)weightMap + (size_t)py * weightPitch);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int px = rx + sx + i;
                if (px < 0 || px >= width) continue;
                const float pixelWeight = groupWeight * c_bm3dKaiser[j * 8 + i];
                atomicAdd(&accRow[px], noiseStack[n][j][i] * pixelWeight);
                atomicAdd(&weightRow[px], pixelWeight);
            }
        }
    }
}

template<typename Type>
__global__ void kernelBm3dNormalize(uint8_t *dst, int dstPitch, const uint8_t *src, int srcPitch,
    int width, int height, const float *accumulator, int accPitch, const float *weightMap, int weightPitch,
    float pixelMax) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float *accRow = (const float *)((const uint8_t *)accumulator + (size_t)y * accPitch);
    const float *weightRow = (const float *)((const uint8_t *)weightMap + (size_t)y * weightPitch);
    Type *dstRow = (Type *)(dst + (size_t)y * dstPitch);
    const float weight = weightRow[x];
    if (weight <= 0.0f) {
        dstRow[x] = ((const Type *)(src + (size_t)y * srcPitch))[x];
    } else {
        dstRow[x] = (Type)(fminf(fmaxf(accRow[x] / weight, 0.0f), pixelMax) + 0.5f);
    }
}

template<typename Type, bool temporal>
static RGY_ERR bm3dProcessPlaneCuda(RGYFrameInfo *output, const RGYFrameInfo *input,
    short *similarCoords, uint8_t *similarFrameIdx, uint8_t *blockCounts,
    float *accumulator, int accPitch, float *weightMap, int weightPitch, float *basic, int basicPitch,
    const uint8_t *noisyRing, int noisyRingPitch, const float *basicRing, int basicRingPitch,
    int ringCursor, int ringRadius, int ringFilled,
    int blockStep, int groupSize, int bmRange, float sigmaScaled, int distThreshold, int bitDepth,
    cudaStream_t stream) {
    const int width = input->width;
    const int height = input->height;
    const int refCountX = divCeil(width, blockStep);
    const int refCountY = divCeil(height, blockStep);
    const int noisySlotStride = noisyRingPitch * height;
    const int basicSlotStride = basicRingPitch * height;
    const dim3 blockRef(8, 8);
    const dim3 gridRef(divCeil(refCountX, (int)blockRef.x), divCeil(refCountY, (int)blockRef.y));
    const dim3 blockPixel(32, 8);
    const dim3 gridPixel(divCeil(width, (int)blockPixel.x), divCeil(height, (int)blockPixel.y));
    const size_t floatBytes = (size_t)accPitch * height;
    auto cudaerr = cudaMemsetAsync(accumulator, 0, floatBytes, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    cudaerr = cudaMemsetAsync(weightMap, 0, (size_t)weightPitch * height, stream);
    if (cudaerr != cudaSuccess) return err_to_rgy(cudaerr);
    kernelBm3dMatch<Type, temporal><<<gridRef, blockRef, 0, stream>>>(
        input->ptr[0], input->pitch[0], noisyRing, noisyRingPitch, noisySlotStride,
        ringCursor, ringRadius, ringFilled, width, height, refCountX, refCountY,
        similarCoords, similarFrameIdx, blockCounts, blockStep, bmRange, min(groupSize, BM3D_MAX_GROUP_SIZE_BASIC), distThreshold);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    kernelBm3dBasic<Type, temporal><<<gridRef, blockRef, 0, stream>>>(
        input->ptr[0], input->pitch[0], noisyRing, noisyRingPitch, noisySlotStride,
        ringCursor, ringRadius, width, height, refCountX, refCountY,
        similarCoords, similarFrameIdx, blockCounts, accumulator, accPitch, weightMap, weightPitch,
        blockStep, min(groupSize, BM3D_MAX_GROUP_SIZE_BASIC), sigmaScaled, 2.7f * sigmaScaled);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    kernelBm3dNormalizeF32<Type><<<gridPixel, blockPixel, 0, stream>>>(
        basic, basicPitch, input->ptr[0], input->pitch[0], width, height,
        accumulator, accPitch, weightMap, weightPitch);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    if ((cudaerr = cudaMemsetAsync(accumulator, 0, floatBytes, stream)) != cudaSuccess) return err_to_rgy(cudaerr);
    if ((cudaerr = cudaMemsetAsync(weightMap, 0, (size_t)weightPitch * height, stream)) != cudaSuccess) return err_to_rgy(cudaerr);
    kernelBm3dMatchBasic<temporal><<<gridRef, blockRef, 0, stream>>>(
        basic, basicPitch, basicRing, basicRingPitch, basicSlotStride,
        ringCursor, ringRadius, ringFilled, width, height, refCountX, refCountY,
        similarCoords, similarFrameIdx, blockCounts, blockStep, bmRange, groupSize,
        (float)distThreshold / 6.25f);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    kernelBm3dWiener<Type, temporal><<<gridRef, blockRef, 0, stream>>>(
        input->ptr[0], input->pitch[0], noisyRing, noisyRingPitch, noisySlotStride,
        basic, basicPitch, basicRing, basicRingPitch, basicSlotStride,
        ringCursor, ringRadius, width, height, refCountX, refCountY,
        similarCoords, similarFrameIdx, blockCounts, accumulator, accPitch, weightMap, weightPitch,
        blockStep, groupSize, sigmaScaled);
    if ((cudaerr = cudaGetLastError()) != cudaSuccess) return err_to_rgy(cudaerr);
    kernelBm3dNormalize<Type><<<gridPixel, blockPixel, 0, stream>>>(
        output->ptr[0], output->pitch[0], input->ptr[0], input->pitch[0], width, height,
        accumulator, accPitch, weightMap, weightPitch, (float)((1 << bitDepth) - 1));
    return err_to_rgy(cudaGetLastError());
}

NVEncFilterDenoiseBm3d::NVEncFilterDenoiseBm3d() :
    m_bufSimilarCoords(), m_bufBlockCounts(),
    m_bufAccumulator(), m_bufWeightMap(), m_bufBasicEstimate(), m_bufSimilarFrameIdx(),
    m_pastNoisyRing(), m_pastBasicRing(),
    m_ringW({ 0, 0, 0 }), m_ringH({ 0, 0, 0 }),
    m_ringNoisyPitch({ 0, 0, 0 }), m_ringBasicPitch({ 0, 0, 0 }),
    m_ringRadius(0), m_ringSlotCursor(0), m_ringFilled(0),
    m_scratchW(0), m_scratchH(0), m_scratchBlockStep(0), m_scratchGroupSize(0),
    m_accPitch(0), m_wmapPitch(0), m_basicPitch(0) {
    m_name = _T("bm3d");
}

NVEncFilterDenoiseBm3d::~NVEncFilterDenoiseBm3d() {
    close();
}

void NVEncFilterDenoiseBm3d::resetTemporalState() {
    m_ringSlotCursor = 0;
    m_ringFilled = 0;
}

RGY_ERR NVEncFilterDenoiseBm3d::ensureScratch(int width, int height) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseBm3d>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    const int blockStep = std::min(prm->bm3d.block_step, BM3D_BLOCK_SIZE);
    const int groupSize = std::min(prm->bm3d.group_size, BM3D_MAX_GROUP_SIZE_WIENER);
    const bool sameLayout = blockStep == m_scratchBlockStep && groupSize == m_scratchGroupSize;
    if (m_bufSimilarCoords && m_bufBlockCounts && m_bufAccumulator && m_bufWeightMap && m_bufBasicEstimate
        && (m_ringRadius == 0 || m_bufSimilarFrameIdx)
        && width <= m_scratchW && height <= m_scratchH && sameLayout) {
        m_accPitch = width * (int)sizeof(float);
        m_wmapPitch = width * (int)sizeof(float);
        m_basicPitch = width * (int)sizeof(float);
        return RGY_ERR_NONE;
    }
    // 色差面へ移っても最大確保寸法を維持し、フレーム内の再確保を避ける。
    const int allocWidth = sameLayout ? std::max(width, m_scratchW) : width;
    const int allocHeight = sameLayout ? std::max(height, m_scratchH) : height;
    const size_t refCount = (size_t)divCeil(allocWidth, blockStep) * divCeil(allocHeight, blockStep);
    const size_t coordBytes = refCount * (size_t)groupSize * 2 * sizeof(int16_t);
    const size_t countBytes = refCount * sizeof(uint8_t);
    const size_t floatBytes = (size_t)allocWidth * allocHeight * sizeof(float);
    auto allocBuffer = [](size_t bytes) -> std::unique_ptr<CUMemBuf> {
        auto buffer = std::make_unique<CUMemBuf>(bytes);
        return (buffer->alloc() == RGY_ERR_NONE) ? std::move(buffer) : nullptr;
    };
    auto similarCoords = allocBuffer(coordBytes);
    auto blockCounts = allocBuffer(countBytes);
    auto accumulator = allocBuffer(floatBytes);
    auto weightMap = allocBuffer(floatBytes);
    auto basicEstimate = allocBuffer(floatBytes);
    auto similarFrameIdx = (m_ringRadius > 0)
        ? allocBuffer(refCount * BM3D_MAX_GROUP_SIZE_TEMPORAL * sizeof(uint8_t))
        : nullptr;
    if (!similarCoords || !blockCounts || !accumulator || !weightMap || !basicEstimate
        || (m_ringRadius > 0 && !similarFrameIdx)) {
        AddMessage(RGY_LOG_ERROR, _T("BM3D用作業バッファの確保に失敗しました。\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    m_bufSimilarCoords = std::move(similarCoords);
    m_bufBlockCounts = std::move(blockCounts);
    m_bufAccumulator = std::move(accumulator);
    m_bufWeightMap = std::move(weightMap);
    m_bufBasicEstimate = std::move(basicEstimate);
    m_bufSimilarFrameIdx = std::move(similarFrameIdx);
    m_scratchW = allocWidth;
    m_scratchH = allocHeight;
    m_scratchBlockStep = blockStep;
    m_scratchGroupSize = groupSize;
    m_accPitch = width * (int)sizeof(float);
    m_wmapPitch = width * (int)sizeof(float);
    m_basicPitch = width * (int)sizeof(float);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseBm3d::ensureRingBuffers(int planeIdx, int width, int height) {
    if (planeIdx < 0 || planeIdx >= 3) return RGY_ERR_INVALID_PARAM;
    if (m_ringRadius <= 0) return RGY_ERR_NONE;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseBm3d>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    const int bytesPerSample = (RGY_CSP_BIT_DEPTH[prm->frameOut.csp] > 8) ? 2 : 1;
    const int noisyPitch = width * bytesPerSample;
    const int basicPitch = width * (int)sizeof(float);
    if (m_pastNoisyRing[planeIdx] && m_pastBasicRing[planeIdx]
        && m_ringW[planeIdx] == width && m_ringH[planeIdx] == height
        && m_ringNoisyPitch[planeIdx] == noisyPitch && m_ringBasicPitch[planeIdx] == basicPitch) {
        return RGY_ERR_NONE;
    }
    auto noisy = std::make_unique<CUMemBuf>((size_t)noisyPitch * height * m_ringRadius);
    auto basic = std::make_unique<CUMemBuf>((size_t)basicPitch * height * m_ringRadius);
    if (noisy->alloc() != RGY_ERR_NONE || basic->alloc() != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("BM3D時間方向履歴バッファの確保に失敗しました (plane %d)。\n"), planeIdx);
        return RGY_ERR_MEMORY_ALLOC;
    }
    m_pastNoisyRing[planeIdx] = std::move(noisy);
    m_pastBasicRing[planeIdx] = std::move(basic);
    m_ringW[planeIdx] = width;
    m_ringH[planeIdx] = height;
    m_ringNoisyPitch[planeIdx] = noisyPitch;
    m_ringBasicPitch[planeIdx] = basicPitch;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseBm3d::pushNoisyToRing(int planeIdx, const RGYFrameInfo *input, cudaStream_t stream) {
    if (m_ringRadius <= 0 || !m_pastNoisyRing[planeIdx]) return RGY_ERR_NONE;
    const int slot = m_ringSlotCursor;
    uint8_t *dst = (uint8_t *)m_pastNoisyRing[planeIdx]->ptr
        + (size_t)slot * m_ringNoisyPitch[planeIdx] * input->height;
    const size_t rowBytes = (size_t)input->width * ((RGY_CSP_BIT_DEPTH[m_param->frameOut.csp] > 8) ? 2 : 1);
    return err_to_rgy(cudaMemcpy2DAsync(dst, m_ringNoisyPitch[planeIdx],
        input->ptr[0], input->pitch[0], rowBytes, input->height, cudaMemcpyDeviceToDevice, stream));
}

RGY_ERR NVEncFilterDenoiseBm3d::pushBasicToRing(int planeIdx, cudaStream_t stream) {
    if (m_ringRadius <= 0 || !m_pastBasicRing[planeIdx] || !m_bufBasicEstimate) return RGY_ERR_NONE;
    const int slot = m_ringSlotCursor;
    uint8_t *dst = (uint8_t *)m_pastBasicRing[planeIdx]->ptr
        + (size_t)slot * m_ringBasicPitch[planeIdx] * m_ringH[planeIdx];
    return err_to_rgy(cudaMemcpyAsync(dst, m_bufBasicEstimate->ptr,
        (size_t)m_ringBasicPitch[planeIdx] * m_ringH[planeIdx], cudaMemcpyDeviceToDevice, stream));
}

RGY_ERR NVEncFilterDenoiseBm3d::procPlane(int planeIdx, RGYFrameInfo *output, const RGYFrameInfo *input, cudaStream_t stream) {
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseBm3d>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    auto sts = ensureScratch(input->width, input->height);
    if (sts != RGY_ERR_NONE) return sts;
    const bool temporal = m_ringRadius > 0;
    if (temporal) {
        sts = ensureRingBuffers(planeIdx, input->width, input->height);
        if (sts != RGY_ERR_NONE) return sts;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const float pixelScale = (float)(1 << (bitDepth - 8));
    const float sigmaScaled = prm->bm3d.sigma * pixelScale;
    const int distThreshold = (int)(2500.0f * pixelScale * pixelScale);
    const int groupSize = temporal
        ? std::min(prm->bm3d.group_size, BM3D_MAX_GROUP_SIZE_TEMPORAL)
        : std::min(prm->bm3d.group_size, BM3D_MAX_GROUP_SIZE_WIENER);
    const uint8_t *noisyRing = temporal ? (const uint8_t *)m_pastNoisyRing[planeIdx]->ptr : nullptr;
    const float *basicRing = temporal ? (const float *)m_pastBasicRing[planeIdx]->ptr : nullptr;
    const int noisyPitch = temporal ? m_ringNoisyPitch[planeIdx] : 0;
    const int basicRingPitch = temporal ? m_ringBasicPitch[planeIdx] : 0;
    if (bitDepth > 8) {
        if (temporal) {
            sts = bm3dProcessPlaneCuda<uint16_t, true>(output, input,
                (short *)m_bufSimilarCoords->ptr, (uint8_t *)m_bufSimilarFrameIdx->ptr, (uint8_t *)m_bufBlockCounts->ptr,
                (float *)m_bufAccumulator->ptr, m_accPitch, (float *)m_bufWeightMap->ptr, m_wmapPitch,
                (float *)m_bufBasicEstimate->ptr, m_basicPitch,
                noisyRing, noisyPitch, basicRing, basicRingPitch, m_ringSlotCursor, m_ringRadius, m_ringFilled,
                m_scratchBlockStep, groupSize, prm->bm3d.bm_range, sigmaScaled, distThreshold, bitDepth, stream);
        } else {
            sts = bm3dProcessPlaneCuda<uint16_t, false>(output, input,
                (short *)m_bufSimilarCoords->ptr, nullptr, (uint8_t *)m_bufBlockCounts->ptr,
                (float *)m_bufAccumulator->ptr, m_accPitch, (float *)m_bufWeightMap->ptr, m_wmapPitch,
                (float *)m_bufBasicEstimate->ptr, m_basicPitch,
                nullptr, 0, nullptr, 0, 0, 0, 0,
                m_scratchBlockStep, groupSize, prm->bm3d.bm_range, sigmaScaled, distThreshold, bitDepth, stream);
        }
    } else {
        if (temporal) {
            sts = bm3dProcessPlaneCuda<uint8_t, true>(output, input,
                (short *)m_bufSimilarCoords->ptr, (uint8_t *)m_bufSimilarFrameIdx->ptr, (uint8_t *)m_bufBlockCounts->ptr,
                (float *)m_bufAccumulator->ptr, m_accPitch, (float *)m_bufWeightMap->ptr, m_wmapPitch,
                (float *)m_bufBasicEstimate->ptr, m_basicPitch,
                noisyRing, noisyPitch, basicRing, basicRingPitch, m_ringSlotCursor, m_ringRadius, m_ringFilled,
                m_scratchBlockStep, groupSize, prm->bm3d.bm_range, sigmaScaled, distThreshold, bitDepth, stream);
        } else {
            sts = bm3dProcessPlaneCuda<uint8_t, false>(output, input,
                (short *)m_bufSimilarCoords->ptr, nullptr, (uint8_t *)m_bufBlockCounts->ptr,
                (float *)m_bufAccumulator->ptr, m_accPitch, (float *)m_bufWeightMap->ptr, m_wmapPitch,
                (float *)m_bufBasicEstimate->ptr, m_basicPitch,
                nullptr, 0, nullptr, 0, 0, 0, 0,
                m_scratchBlockStep, groupSize, prm->bm3d.bm_range, sigmaScaled, distThreshold, bitDepth, stream);
        }
    }
    return sts;
}

RGY_ERR NVEncFilterDenoiseBm3d::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseBm3d>(pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("パラメータ型が不正です。\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->frameOut.height <= 0 || prm->frameOut.width <= 0) {
        AddMessage(RGY_LOG_ERROR, _T("解像度が不正です。\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    const int bitDepth = RGY_CSP_BIT_DEPTH[prm->frameOut.csp];
    const int planeCount = RGY_CSP_PLANES[prm->frameOut.csp];
    if ((RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U8
            && RGY_CSP_DATA_TYPE[prm->frameOut.csp] != RGY_DATA_TYPE_U16)
        || bitDepth > 12 || (planeCount != 1 && planeCount != 3)
        || rgy_chromafmt_is_rgb(RGY_CSP_CHROMA_FORMAT[prm->frameOut.csp])) {
        AddMessage(RGY_LOG_ERROR, _T("BM3Dは12bit以下のplanar YUVのみ対応しています: %s。\n"),
            RGY_CSP_NAMES[prm->frameOut.csp]);
        return RGY_ERR_UNSUPPORTED;
    }
    if (!std::isfinite(prm->bm3d.sigma)) {
        AddMessage(RGY_LOG_ERROR, _T("sigmaには有限値を指定してください。\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->bm3d.sigma != 0.0f && (prm->bm3d.sigma < 0.5f || prm->bm3d.sigma > 100.0f)) {
        prm->bm3d.sigma = clamp(prm->bm3d.sigma, 0.5f, 100.0f);
        AddMessage(RGY_LOG_WARN, _T("sigmaを%.2f～%.2fの範囲に補正しました。\n"), 0.5f, 100.0f);
    }
    if (prm->bm3d.block_step < 1 || prm->bm3d.block_step > BM3D_BLOCK_SIZE) {
        prm->bm3d.block_step = clamp(prm->bm3d.block_step, 1, BM3D_BLOCK_SIZE);
        AddMessage(RGY_LOG_WARN, _T("block_stepを%d～%dの範囲に補正しました。\n"), 1, BM3D_BLOCK_SIZE);
    }
    if (prm->bm3d.group_size < 1 || prm->bm3d.group_size > BM3D_MAX_GROUP_SIZE_WIENER) {
        prm->bm3d.group_size = clamp(prm->bm3d.group_size, 1, BM3D_MAX_GROUP_SIZE_WIENER);
        AddMessage(RGY_LOG_WARN, _T("group_sizeを%d～%dの範囲に補正しました。\n"), 1, BM3D_MAX_GROUP_SIZE_WIENER);
    }
    if (prm->bm3d.bm_range < 1 || prm->bm3d.bm_range > 32) {
        prm->bm3d.bm_range = clamp(prm->bm3d.bm_range, 1, 32);
        AddMessage(RGY_LOG_WARN, _T("bm_rangeを%d～%dの範囲に補正しました。\n"), 1, 32);
    }
    if (prm->bm3d.radius < 0 || prm->bm3d.radius > 4) {
        prm->bm3d.radius = clamp(prm->bm3d.radius, 0, 4);
        AddMessage(RGY_LOG_WARN, _T("radiusを%d～%dの範囲に補正しました。\n"), 0, 4);
    }
    if (prm->bm3d.radius > 0 && prm->bm3d.group_size > BM3D_MAX_GROUP_SIZE_TEMPORAL) {
        AddMessage(RGY_LOG_WARN, _T("時間方向BM3Dではgroup_sizeを%dに補正します。\n"), BM3D_MAX_GROUP_SIZE_TEMPORAL);
        prm->bm3d.group_size = BM3D_MAX_GROUP_SIZE_TEMPORAL;
    }

    auto prev = std::dynamic_pointer_cast<NVEncFilterParamDenoiseBm3d>(m_param);
    const bool resetRing = !prev
        || prev->bm3d.radius != prm->bm3d.radius
        || prev->bm3d.chroma != prm->bm3d.chroma
        || cmpFrameInfoCspResolution(&prev->frameOut, &prm->frameOut);
    if (resetRing) {
        for (int plane = 0; plane < 3; plane++) {
            m_pastNoisyRing[plane].reset();
            m_pastBasicRing[plane].reset();
            m_ringW[plane] = 0;
            m_ringH[plane] = 0;
            m_ringNoisyPitch[plane] = 0;
            m_ringBasicPitch[plane] = 0;
        }
        resetTemporalState();
    }
    m_ringRadius = prm->bm3d.radius;
    if (prev && (prev->bm3d.block_step != prm->bm3d.block_step
        || prev->bm3d.group_size != prm->bm3d.group_size
        || prev->bm3d.radius != prm->bm3d.radius)) {
        m_bufSimilarCoords.reset();
        m_bufBlockCounts.reset();
        m_bufAccumulator.reset();
        m_bufWeightMap.reset();
        m_bufBasicEstimate.reset();
        m_bufSimilarFrameIdx.reset();
        m_scratchW = 0;
        m_scratchH = 0;
        m_scratchBlockStep = 0;
        m_scratchGroupSize = 0;
    }
    auto sts = AllocFrameBuf(prm->frameOut, 1);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("出力フレームの確保に失敗しました: %s。\n"), get_err_mes(sts));
        return sts;
    }
    for (int plane = 0; plane < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; plane++) {
        prm->frameOut.pitch[plane] = m_frameBuf[0]->frame.pitch[plane];
    }
    setFilterInfo(prm->print());
    m_param = prm;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterDenoiseBm3d::run_filter(const RGYFrameInfo *pInputFrame,
    RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) {
    if (pInputFrame->ptr[0] == nullptr) {
        *pOutputFrameNum = 0;
        ppOutputFrames[0] = nullptr;
        return RGY_ERR_NONE;
    }
    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        ppOutputFrames[0] = &m_frameBuf[m_nFrameIdx]->frame;
        m_nFrameIdx = (m_nFrameIdx + 1) % m_frameBuf.size();
    }
    ppOutputFrames[0]->picstruct = pInputFrame->picstruct;
    if (getCudaMemcpyKind(pInputFrame->mem_type, ppOutputFrames[0]->mem_type) != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("BM3DはGPUメモリ上のフレームのみ対応しています。\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (m_param->frameOut.csp != m_param->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("入出力の色空間が一致しません。\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamDenoiseBm3d>(m_param);
    if (!prm) return RGY_ERR_INVALID_PARAM;
    if (prm->bm3d.sigma == 0.0f) {
        return copyFrameAsync(ppOutputFrames[0], pInputFrame, stream);
    }
    const int planeCount = RGY_CSP_PLANES[pInputFrame->csp];
    for (int plane = 0; plane < planeCount; plane++) {
        auto outputPlane = getPlane(ppOutputFrames[0], (RGY_PLANE)plane);
        auto inputPlane = getPlane(pInputFrame, (RGY_PLANE)plane);
        RGY_ERR sts = RGY_ERR_NONE;
        if (plane == 0 || prm->bm3d.chroma) {
            sts = procPlane(plane, &outputPlane, &inputPlane, stream);
            if (sts == RGY_ERR_NONE && m_ringRadius > 0) {
                sts = pushNoisyToRing(plane, &inputPlane, stream);
            }
            if (sts == RGY_ERR_NONE && m_ringRadius > 0) {
                sts = pushBasicToRing(plane, stream);
            }
        } else {
            sts = copyPlaneAsync(&outputPlane, &inputPlane, stream);
        }
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("BM3D処理に失敗しました (%s, plane %d): %s。\n"),
                RGY_CSP_NAMES[pInputFrame->csp], plane, get_err_mes(sts));
            return sts;
        }
    }
    if (m_ringRadius > 0) {
        m_ringSlotCursor = (m_ringSlotCursor + 1) % m_ringRadius;
        m_ringFilled = std::min(m_ringFilled + 1, m_ringRadius);
    }
    return RGY_ERR_NONE;
}

void NVEncFilterDenoiseBm3d::close() {
    m_frameBuf.clear();
    m_bufSimilarCoords.reset();
    m_bufBlockCounts.reset();
    m_bufAccumulator.reset();
    m_bufWeightMap.reset();
    m_bufBasicEstimate.reset();
    m_bufSimilarFrameIdx.reset();
    for (int plane = 0; plane < 3; plane++) {
        m_pastNoisyRing[plane].reset();
        m_pastBasicRing[plane].reset();
        m_ringW[plane] = 0;
        m_ringH[plane] = 0;
        m_ringNoisyPitch[plane] = 0;
        m_ringBasicPitch[plane] = 0;
    }
    m_ringRadius = 0;
    resetTemporalState();
    m_scratchW = 0;
    m_scratchH = 0;
    m_scratchBlockStep = 0;
    m_scratchGroupSize = 0;
    m_accPitch = 0;
    m_wmapPitch = 0;
    m_basicPitch = 0;
}

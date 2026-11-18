/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include "ATenAtomic.cuh"
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>

#    include "Common.h"
#    include "Rasterization.h"
#    include "RasterizeToPixels3DGSDevice.cuh"
#    include "Utils.cuh"
#    include "Dispatch.h"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

#    if USE_ROCM
// ---------------------------------------------------------------------------
// MI300-class (wave64, gfx9) optimized backward for 64-thread tiles
// (tile_size == 8). One block is exactly one wavefront, so:
//   * the per-gaussian data (id / xy+opacity / conic) is held in registers and
//     broadcast to the wave with __shfl instead of staging it in shared memory
//     (only the per-channel colors still use LDS),
//   * the per-gaussian gradient reductions use DPP row/bcast ops
//     (v_mov_b32_dpp), which fold into the consuming add and need no LDS,
//   * the CDIM-wide v_colors reduction leaves element e in lane e % 64, so the
//     final atomics are wave-coalesced instead of serialized on one lane.
// ---------------------------------------------------------------------------
// 32-lane targets have no DPP broadcast and never launch the kernel that uses it.
template<int CTRL>
__device__ __forceinline__ float dpp_mov(float v)
{
    if constexpr(GSPLAT_WARP_SIZE != 64)
    {
        return v;
    }
    else
    {
        return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), CTRL, 0xF, 0xF, true));
    }
}

template<int CTRL>
__device__ __forceinline__ int32_t dpp_mov(int32_t v)
{
    if constexpr(GSPLAT_WARP_SIZE != 64)
    {
        return v;
    }
    else
    {
        return __builtin_amdgcn_mov_dpp(v, CTRL, 0xF, 0xF, true);
    }
}

// DPP controls: ROW_SHR(n) = 0x110 + n, ROW_BCAST15 = 0x142, ROW_BCAST31 = 0x143.
// After the last step lane 63 holds the reduction of the whole wave.
__device__ __forceinline__ float dpp_wave64_sum(float val)
{
    float tmp = val + dpp_mov<0x118>(val);
    tmp       = tmp + dpp_mov<0x114>(tmp);
    tmp       = tmp + dpp_mov<0x112>(tmp);
    tmp       = tmp + dpp_mov<0x111>(tmp);
    tmp       = tmp + dpp_mov<0x142>(tmp);
    tmp       = tmp + dpp_mov<0x143>(tmp);
    return tmp;
}

__device__ __forceinline__ int32_t dpp_wave64_max(int32_t val)
{
    int32_t tmp = max(val, dpp_mov<0x118>(val));
    tmp         = max(tmp, dpp_mov<0x114>(tmp));
    tmp         = max(tmp, dpp_mov<0x112>(tmp));
    tmp         = max(tmp, dpp_mov<0x111>(tmp));
    tmp         = max(tmp, dpp_mov<0x142>(tmp));
    tmp         = max(tmp, dpp_mov<0x143>(tmp));
    return __shfl(tmp, 63);
}

// Wave sum, result broadcast to every lane.
__device__ __forceinline__ void dpp_sum_all(float &val)
{
    val = __shfl(dpp_wave64_sum(val), 63);
}

__device__ __forceinline__ void dpp_sum_all(vec2 &val)
{
    dpp_sum_all(val.x);
    dpp_sum_all(val.y);
}

__device__ __forceinline__ void dpp_sum_all(vec3 &val)
{
    dpp_sum_all(val.x);
    dpp_sum_all(val.y);
    dpp_sum_all(val.z);
}

// Wave sum of each of the `numel` elements of `val`; the reduced element e ends
// up in val[e / 64] of lane (e % 64), ready for a wave-coalesced store.
template<uint32_t numel>
__device__ __forceinline__ void dpp_sum_spread(float (&val)[numel], const uint32_t lane)
{
#        pragma unroll
    for(uint32_t e = 0; e < numel; ++e)
    {
        const float sum = __shfl(dpp_wave64_sum(val[e]), 63);
        if(lane == e % 64)
        {
            val[e / 64] = sum;
        }
    }
}

template<uint32_t CDIM, typename scalar_t>
__launch_bounds__(64) __global__ void rasterize_bs64_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const int64_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t block_offset,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,       // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,   // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,       // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,        // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,    // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [..., N] or [nnz]
    const uint32_t max_batch_size
)
{
    auto block              = cg::this_thread_block();
    auto linear_block_index = block.group_index().x + block_offset;
    dim3 block_index(
        linear_block_index / (tile_width * tile_height),
        (linear_block_index / tile_width) % tile_height,
        linear_block_index % tile_width
    );

    uint32_t image_id = block_index.x;
    uint32_t tile_id  = block_index.y * tile_width + block_index.z;
    uint32_t i        = block_index.y * tile_size + block.thread_index().y;
    uint32_t j        = block_index.z * tile_size + block.thread_index().x;

    tile_offsets    += image_id * tile_height * tile_width;
    render_alphas   += image_id * image_height * image_width;
    last_ids        += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if(masks != nullptr && !masks[tile_id])
    {
        return;
    }

    const float px       = (float)j + 0.5f;
    const float py       = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int64_t num_intersections = range_end - range_start;
    const int64_t num_batches       = (num_intersections + max_batch_size - 1) / max_batch_size;

    // Only the per-channel colors live in shared memory: [max_batch_size * CDIM]
    extern __shared__ int s[];
    float *rgbs_batch = (float *)s;

    // this is the T AFTER the last gaussian in this pixel
    float T_final           = 1.0f - render_alphas[pix_id];
    float T                 = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM]      = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#        pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);

    const int32_t warp_bin_final = dpp_wave64_max(bin_final);

    // this thread's gaussian of the current batch (broadcast to the wave via __shfl)
    int32_t g_reg = 0;
    vec3 xy_opacity_reg;
    vec3 conic_reg;
    for(int64_t b = 0; b < num_batches; ++b)
    {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // batch_end_offset is relative to this tile's first intersection.
        const int64_t batch_end_offset = num_intersections - 1 - (int64_t)max_batch_size * b;
        const int64_t remaining        = batch_end_offset + 1;
        const uint32_t batch_size      = static_cast<uint32_t>(remaining < max_batch_size ? remaining : max_batch_size);
        const int64_t idx              = range_start + batch_end_offset - tr;
        if(tr < batch_size)
        {
            const int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            g_reg           = g;
            const vec2 xy   = means2d[g];
            const float opac = opacities[g];
            xy_opacity_reg   = {xy.x, xy.y, opac};
            conic_reg        = conics[g];
#        pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        const int64_t first_t64 = batch_end_offset > warp_bin_final ? batch_end_offset - warp_bin_final : 0;
        const uint32_t first_t  = first_t64 < batch_size ? static_cast<uint32_t>(first_t64) : batch_size;
        for(uint32_t t = first_t; t < batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end_offset - t > bin_final)
            {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;
            conic.x = __shfl(conic_reg.x, t);
            conic.y = __shfl(conic_reg.y, t);
            conic.z = __shfl(conic_reg.z, t);
            vec3 xy_opac;
            xy_opac.x = __shfl(xy_opacity_reg.x, t);
            xy_opac.y = __shfl(xy_opacity_reg.y, t);
            xy_opac.z = __shfl(xy_opacity_reg.z, t);

            if(valid)
            {
                opac                    = xy_opac.z;
                delta                   = {xy_opac.x - px, xy_opac.y - py};
                const GaussianWeight gw = eval_gaussian_weight(conic, delta.x, delta.y, opac);
                vis                     = gw.vis;
                alpha                   = gw.alpha;
                if(!gw.valid)
                {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid))
            {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            vec2 v_xy_abs_local     = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if(valid)
            {
                // compute the current T for this gaussian
                float ra         = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T               *= ra;
                // update v_rgb for this gaussian
                const float fac  = alpha * T;
#        pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#        pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if(backgrounds != nullptr)
                {
                    float accum = 0.f;
#        pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if(opac * vis <= MAX_ALPHA)
                {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local
                        = {0.5f * v_sigma * delta.x * delta.x,
                           v_sigma * delta.x * delta.y,
                           0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local
                        = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                           v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    if(v_means2d_abs != nullptr)
                    {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#        pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            dpp_sum_spread<CDIM>(v_rgb_local, tr); // element e is left in lane e % 64
            dpp_sum_all(v_conic_local);
            dpp_sum_all(v_xy_local);
            if(v_means2d_abs != nullptr)
            {
                dpp_sum_all(v_xy_abs_local);
            }
            dpp_sum_all(v_opacity_local);

            const int32_t g = __shfl(g_reg, t); // flatten index in [I * N] or [nnz]

            // wave-coalesced: lane l updates channel k + l
            float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#        pragma unroll
            for(uint32_t k = 0; k < CDIM; k += 64)
            {
                if(k + tr < CDIM)
                {
                    unsafeAtomicAdd(v_rgb_ptr + k + tr, v_rgb_local[k / 64]);
                }
            }

            if(tr == 0)
            {
                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                unsafeAtomicAdd(v_conic_ptr, v_conic_local.x);
                unsafeAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                unsafeAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                unsafeAtomicAdd(v_xy_ptr, v_xy_local.x);
                unsafeAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    unsafeAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    unsafeAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                unsafeAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

// Use the wave64-optimized kernel for 64-thread tiles when its (color-only)
// shared memory fits in the 64 KB LDS. Fills in the launch configuration.
template<uint32_t CDIM>
inline bool rocm_use_bs64_bwd(const uint32_t tile_size, uint32_t &max_batch_size, int64_t &shmem_size)
{
    if(tile_size * tile_size != 64 || device_warp_size() != 64)
    {
        return false;
    }
    const uint32_t batch = CDIM <= 32 ? 64u : 32u;
    const int64_t bytes  = static_cast<int64_t>(batch) * CDIM * sizeof(float);
    if(bytes > 64 * 1024)
    {
        return false;
    }
    max_batch_size = batch;
    shmem_size     = bytes;
    return true;
}

// ---------------------------------------------------------------------------
// 32-lane GPUs: backward for 16x16 tiles without atomics, which serialize on
// the gaussians that cover many tiles. As in the forward, a wavefront owns an
// 8x8 pixel block, two pixels per lane, and walks only the gaussians that reach
// it. Thread t adds up gaussian t's sums over the tile and stores them at the
// intersection's index before the sort, which intersect_tile leaves behind
// flatten_ids. There the intersections of a gaussian are contiguous, and
// gather_isect_grads_kernel adds them.
// ---------------------------------------------------------------------------

// Reduces across the wavefront; every lane receives the result.
template<typename T, typename Op>
__device__ __forceinline__ T wave_all_reduce(T val, Op op)
{
    using reduce_t = rocprim::warp_reduce<T, GSPLAT_WARP_SIZE, /*UseAllReduce=*/true>;
    typename reduce_t::storage_type unused;
    T result;
    reduce_t().reduce(val, result, unused, op);
    return result;
}

// Sums eight values over a 32-lane wavefront at once: lane k < 8 returns the total of
// v[k], with seven DPP exchanges and two lane permutes instead of eight reductions.
__device__ __forceinline__ float wave32_sum8(float v[8], const uint32_t lane)
{
#        define GSPLAT_DPP(x, ctrl) __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), ctrl, 0xF, 0xF, false))
    for(int k = 0; k < 4; ++k)
    {
        v[k] = ((lane & 4) ? v[k + 4] : v[k]) + GSPLAT_DPP((lane & 4) ? v[k] : v[k + 4], 0x141);
    }
    for(int k = 0; k < 2; ++k)
    {
        v[k] = ((lane & 2) ? v[k + 2] : v[k]) + GSPLAT_DPP((lane & 2) ? v[k] : v[k + 2], 0x1B);
    }
    v[0] = ((lane & 1) ? v[1] : v[0]) + GSPLAT_DPP((lane & 1) ? v[0] : v[1], 0xB1);
#        undef GSPLAT_DPP
    // add the other half of the row of 16 lanes, then the other row
    if constexpr(ROCPRIM_HAS_PERMLANE())
    {
        const int bits = __float_as_int(v[0]);
        const float half
            = v[0] + __int_as_float(__builtin_amdgcn_permlane16(bits, bits, 0xFEDCBA98, 0x76543210, false, false));
        const int half_bits = __float_as_int(half);
        return half
             + __int_as_float(__builtin_amdgcn_permlanex16(half_bits, half_bits, 0x76543210, 0xFEDCBA98, false, false));
    }
    else
    {
        const float half = v[0] + __shfl_xor(v[0], 8, 32);
        return half + __shfl_xor(half, 16, 32);
    }
}

bool rasterize_blocks_to_pixels_3dgs_bwd_fits(
    const at::Tensor &means2d, const at::Tensor &colors, const uint32_t tile_size, const at::Tensor &flatten_ids
)
{
    const int64_t n_isects = flatten_ids.numel(), n_gauss = means2d.numel() / 2;
    return n_isects > 0
        && tile_size == 16
        && colors.size(-1) <= 32
        && device_warp_size() == 32
        && flatten_ids.storage_offset() == 0
        && flatten_ids.storage().nbytes() == 2 * (n_isects + n_gauss) * sizeof(int32_t);
}

template<uint32_t CDIM, bool ABS>
__global__ void __launch_bounds__(128) rasterize_blocks_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const int64_t n_isects,
    // fwd inputs
    const vec2 *__restrict__ means2d,      // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,       // [..., N, 3] or [nnz, 3]
    const float *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const float *__restrict__ opacities,   // [..., N] or [nnz]
    const float *__restrict__ backgrounds, // [..., CDIM]
    const bool *__restrict__ masks,        // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects], then each one's index before the sort
    // fwd outputs
    const float *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,    // [..., image_height, image_width]
    // grad outputs
    const float *__restrict__ v_render_colors, // [..., image_height, image_width, CDIM]
    const float *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // sums per intersection: colors, conic, mean, opacity, absolute mean
    float *__restrict__ part // [n_isects, GRADS]
)
{
    constexpr uint32_t WAVE = 32, WAVES = 4, PIXELS = 2, GRADS = CDIM + (ABS ? 8 : 6);
    const uint32_t tr = threadIdx.x, wave = tr / WAVE, lane = tr % WAVE;
    const uint32_t n_tiles  = tile_width * tile_height;
    const uint32_t image_id = blockIdx.x / n_tiles, tile_id = blockIdx.x % n_tiles;
    // the first pixel of the wavefront's block
    const uint32_t bx = tile_id % tile_width * 16 + (wave & 1) * 8, by = tile_id / tile_width * 16 + (wave >> 1) * 8;

    tile_offsets    += image_id * n_tiles;
    render_alphas   += image_id * image_height * image_width;
    last_ids        += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }

    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end   = (image_id == I - 1) && (tile_id == n_tiles - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int32_t count       = static_cast<int32_t>(range_end - range_start);
    const int32_t *perm       = flatten_ids + n_isects;

    // this lane's pixels, as rasterize_to_pixels_3dgs_bwd_kernel keeps one
    const float px = (float)(bx + lane % 8) + 0.5f;
    float py[PIXELS], T_final[PIXELS], T[PIXELS], v_render_a[PIXELS];
    float buffer[PIXELS][CDIM] = {0.f}, v_render_c[PIXELS][CDIM];
    int32_t bin_final[PIXELS];
#        pragma unroll
    for(uint32_t p = 0; p < PIXELS; ++p)
    {
        const uint32_t i = by + lane / 8 + 4 * p, j = bx + lane % 8;
        const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);
        py[p]                = (float)i + 0.5f;
        T_final[p]           = 1.0f - render_alphas[pix_id];
        T[p]                 = T_final[p];
#        pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            v_render_c[p][k] = v_render_colors[pix_id * CDIM + k];
        }
        v_render_a[p] = v_render_alphas[pix_id];
        bin_final[p]  = (i < image_height && j < image_width) ? last_ids[pix_id] : -1;
    }

    // The gaussians behind every pixel's last one, and all of a masked tile, get zeros.
    __shared__ int32_t wave_bin_final[WAVES];
    const int32_t block_bin_final = wave_all_reduce(max(bin_final[0], bin_final[1]), rocprim::maximum<int32_t>());
    if(lane == 0)
    {
        wave_bin_final[wave] = block_bin_final;
    }
    __syncthreads();
    const bool masked = masks != nullptr && !masks[image_id * n_tiles + tile_id];
    const int32_t used
        = masked
            ? 0
            : min(count, 1 + max(max(wave_bin_final[0], wave_bin_final[1]), max(wave_bin_final[2], wave_bin_final[3])));
    for(int32_t idx = used + tr; idx < count; idx += WAVES * WAVE)
    {
        float *slot = part + (int64_t)perm[range_start + idx] * GRADS;
#        pragma unroll
        for(uint32_t k = 0; k < GRADS; ++k)
        {
            slot[k] = 0.f;
        }
    }

    __shared__ vec3 xy_opacity_batch[WAVE], conic_batch[WAVE];
    __shared__ float rgbs_batch[WAVE * CDIM], sums[WAVES * WAVE * GRADS];
    __shared__ uint32_t contributed[WAVES];
    const float x0 = (float)bx + 0.5f, y0 = (float)by + 0.5f;

    // batches of 32 gaussians from back to front; thread t fetches the t-th from the back
    for(int32_t batch_end = used - 1; batch_end >= 0; batch_end -= WAVE)
    {
        const uint32_t batch_size = min((int32_t)WAVE, batch_end + 1);
        if(tr < batch_size)
        {
            const int32_t g      = flatten_ids[range_start + batch_end - tr];
            const vec2 xy        = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr]      = conics[g];
#        pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        __syncthreads();

        uint32_t todo   = __builtin_amdgcn_readfirstlane((uint32_t)__ballot(
            lane < batch_size
            && gaussian_reaches_rect(conic_batch[lane], xy_opacity_batch[lane], x0, y0, x0 + 7.f, y0 + 7.f)
        ));
        uint32_t walked = 0;
        while(todo)
        {
            const uint32_t t  = __builtin_ctz(todo);
            todo             &= todo - 1;
            const vec3 conic = conic_batch[t], xy_opac = xy_opacity_batch[t];
            float g[GRADS] = {0.f};
            bool valid     = false;
#        pragma unroll
            for(uint32_t p = 0; p < PIXELS; ++p)
            {
                const vec2 delta        = {xy_opac.x - px, xy_opac.y - py[p]};
                const GaussianWeight gw = eval_gaussian_weight(conic, delta.x, delta.y, xy_opac.z);
                if(batch_end - (int32_t)t > bin_final[p] || !gw.valid)
                {
                    continue;
                }
                float v_rgb_local[CDIM] = {0.f};
                vec3 v_conic_local      = {0.f, 0.f, 0.f};
                vec2 v_xy_local         = {0.f, 0.f};
                vec2 v_xy_abs_local     = {0.f, 0.f};
                float v_opacity_local   = 0.f;
                rasterize_to_pixels_3dgs_blend_bwd<CDIM>(
                    conic,
                    delta,
                    xy_opac.z,
                    gw.vis,
                    gw.alpha,
                    rgbs_batch + t * CDIM,
                    v_render_c[p],
                    v_render_a[p],
                    T_final[p],
                    backgrounds,
                    ABS,
                    T[p],
                    buffer[p],
                    v_rgb_local,
                    v_conic_local,
                    v_xy_local,
                    v_xy_abs_local,
                    v_opacity_local
                );
#        pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    g[k] += v_rgb_local[k];
                }
                g[CDIM]     += v_conic_local.x;
                g[CDIM + 1] += v_conic_local.y;
                g[CDIM + 2] += v_conic_local.z;
                g[CDIM + 3] += v_xy_local.x;
                g[CDIM + 4] += v_xy_local.y;
                g[CDIM + 5] += v_opacity_local;
                if constexpr(ABS)
                {
                    g[CDIM + 6] += v_xy_abs_local.x;
                    g[CDIM + 7] += v_xy_abs_local.y;
                }
                valid = true;
            }
            // if no lane of the wavefront uses the gaussian, skip it
            if(!__ballot(valid))
            {
                continue;
            }
            walked                    |= 1u << t;
            float *slot                = sums + (wave * WAVE + t) * GRADS;
            // eight sums at a time, and a last one alone
            constexpr uint32_t EIGHTS  = GRADS % 8 == 1 ? GRADS - 1 : GRADS;
#        pragma unroll
            for(uint32_t base = 0; base < EIGHTS; base += 8)
            {
                float v[8];
#        pragma unroll
                for(uint32_t k = 0; k < 8; ++k)
                {
                    v[k] = base + k < EIGHTS ? g[base + k] : 0.f;
                }
                const float sum = wave32_sum8(v, lane);
                if(lane < 8 && base + lane < EIGHTS)
                {
                    slot[base + lane] = sum;
                }
            }
            if constexpr(EIGHTS < GRADS)
            {
                const float sum = wave_all_reduce(g[EIGHTS], rocprim::plus<float>());
                if(lane == 0)
                {
                    slot[EIGHTS] = sum;
                }
            }
        }
        if(lane == 0)
        {
            contributed[wave] = walked;
        }
        __syncthreads();

        if(tr < batch_size)
        {
            float total[GRADS] = {0.f};
#        pragma unroll
            for(uint32_t w = 0; w < WAVES; ++w)
            {
                if(contributed[w] >> tr & 1)
                {
#        pragma unroll
                    for(uint32_t k = 0; k < GRADS; ++k)
                    {
                        total[k] += sums[(w * WAVE + tr) * GRADS + k];
                    }
                }
            }
            float *slot = part + (int64_t)perm[range_start + batch_end - tr] * GRADS;
#        pragma unroll
            for(uint32_t k = 0; k < GRADS; ++k)
            {
                slot[k] = total[k];
            }
        }
    }
}

// Adds the sums of each gaussian's intersections, [cum[g - 1], cum[g]) before the sort.
// A lane owns a gaussian; a whole wavefront adds those of more than BIG tiles, for no
// lane to walk a long list.
template<uint32_t CDIM, bool ABS>
__global__ void gather_isect_grads_kernel(
    const int64_t n,
    const int64_t *__restrict__ cum, // [n]
    const float *__restrict__ part,  // [n_isects, GRADS]
    vec2 *__restrict__ v_means2d_abs,
    vec2 *__restrict__ v_means2d,
    vec3 *__restrict__ v_conics,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities
)
{
    constexpr uint32_t BIG = 8, GRADS = CDIM + (ABS ? 8 : 6);
    const int64_t g     = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x % GSPLAT_WARP_SIZE;
    const int64_t u0 = g < n && g > 0 ? cum[g - 1] : 0, u1 = g < n ? cum[g] : 0;
    float s[GRADS] = {0.f};
    uint64_t big   = __ballot(u1 - u0 > BIG);
    while(big)
    {
        const uint32_t owner  = __builtin_ctzll(big);
        big                  &= big - 1;
        const int64_t start = __shfl(u0, owner), end = __shfl(u1, owner);
        float acc[GRADS] = {0.f};
        for(int64_t u = start + lane; u < end; u += GSPLAT_WARP_SIZE)
        {
#        pragma unroll
            for(uint32_t k = 0; k < GRADS; ++k)
            {
                acc[k] += part[u * GRADS + k];
            }
        }
#        pragma unroll
        for(uint32_t k = 0; k < GRADS; ++k)
        {
            acc[k] = wave_all_reduce(acc[k], rocprim::plus<float>());
            if(lane == owner)
            {
                s[k] = acc[k];
            }
        }
    }
    if(g >= n)
    {
        return;
    }
    if(u1 - u0 <= BIG)
    {
        for(int64_t u = u0; u < u1; ++u)
        {
#        pragma unroll
            for(uint32_t k = 0; k < GRADS; ++k)
            {
                s[k] += part[u * GRADS + k];
            }
        }
    }
#        pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_colors[g * CDIM + k] = s[k];
    }
    v_conics[g]    = {s[CDIM], s[CDIM + 1], s[CDIM + 2]};
    v_means2d[g]   = {s[CDIM + 3], s[CDIM + 4]};
    v_opacities[g] = s[CDIM + 5];
    if constexpr(ABS)
    {
        v_means2d_abs[g] = {s[CDIM + 6], s[CDIM + 7]};
    }
}
#    endif

template<uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const int64_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t block_offset,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,       // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities // [..., N] or [nnz]
)
{
    auto block              = cg::this_thread_block();
    auto linear_block_index = block.group_index().x + block_offset;
    dim3 block_index(
        linear_block_index / (tile_width * tile_height),
        (linear_block_index / tile_width) % tile_height,
        linear_block_index % tile_width
    );

    uint32_t image_id = block_index.x;
    uint32_t tile_id  = block_index.y * tile_width + block_index.z;
    uint32_t i        = block_index.y * tile_size + block.thread_index().y;
    uint32_t j        = block_index.z * tile_size + block.thread_index().x;

    tile_offsets    += image_id * tile_height * tile_width;
    render_alphas   += image_id * image_height * image_width;
    last_ids        += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if(masks != nullptr && !masks[tile_id])
    {
        return;
    }

    const float px       = (float)j + 0.5f;
    const float py       = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size       = block.size();
    const int64_t num_intersections = range_end - range_start;
    const int64_t num_batches       = (num_intersections + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;                                            // [block_size]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);         // [block_size]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *rgbs_batch      = (float *)&conic_batch[block_size];                       // [block_size * CDIM]
#    if USE_ROCM
    using warp_reduce_float_t = rocprim::warp_reduce<float, GSPLAT_WARP_SIZE>;
    auto *warp_storage_base   = (typename warp_reduce_float_t::storage_type *)(rgbs_batch + block_size * CDIM);
#    endif

    // this is the T AFTER the last gaussian in this pixel
    float T_final           = 1.0f - render_alphas[pix_id];
    float T                 = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM]      = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#    pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
#    if USE_ROCM
    cg::thread_block_tile<GSPLAT_WARP_SIZE> warp = cg::tiled_partition<GSPLAT_WARP_SIZE>(block);
#    else
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
#    endif
#    if USE_ROCM
    const int32_t warp_bin_final = reduce_max_shuffle(bin_final);
#    else
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
#    endif
    for(int64_t b = 0; b < num_batches; ++b)
    {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch_end_offset is relative to this tile's first intersection.
        const int64_t batch_end_offset = num_intersections - 1 - block_size * b;
        const int64_t remaining        = batch_end_offset + 1;
        const uint32_t batch_size      = static_cast<uint32_t>(remaining < block_size ? remaining : block_size);
        const int64_t idx              = range_start + batch_end_offset - tr;
        if(idx >= range_start)
        {
            int32_t g            = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr]         = g;
            const vec2 xy        = means2d[g];
            const float opac     = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr]      = conics[g];
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        const int64_t first_t64 = batch_end_offset > warp_bin_final ? batch_end_offset - warp_bin_final : 0;
        const uint32_t first_t  = first_t64 < batch_size ? static_cast<uint32_t>(first_t64) : batch_size;
        for(uint32_t t = first_t; t < batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end_offset - t > bin_final)
            {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;

            if(valid)
            {
                conic                   = conic_batch[t];
                vec3 xy_opac            = xy_opacity_batch[t];
                opac                    = xy_opac.z;
                delta                   = {xy_opac.x - px, xy_opac.y - py};
                const GaussianWeight gw = eval_gaussian_weight(conic, delta.x, delta.y, opac);
                vis                     = gw.vis;
                alpha                   = gw.alpha;
                if(!gw.valid)
                {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid))
            {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            vec2 v_xy_abs_local     = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if(valid)
            {
                // compute the current T for this gaussian
                float ra         = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T               *= ra;
                // update v_rgb for this gaussian
                const float fac  = alpha * T;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if(backgrounds != nullptr)
                {
                    float accum = 0.f;
#    pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if(opac * vis <= MAX_ALPHA)
                {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local
                        = {0.5f * v_sigma * delta.x * delta.x,
                           v_sigma * delta.x * delta.y,
                           0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local
                        = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                           v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    if(v_means2d_abs != nullptr)
                    {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
#    if USE_ROCM
            rocprim_warpSum<CDIM, GSPLAT_WARP_SIZE>(v_rgb_local, warp_storage_base);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_conic_local, warp_storage_base);
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_local, warp_storage_base);
            if(v_means2d_abs != nullptr)
            {
                rocprim_warpSum<GSPLAT_WARP_SIZE>(v_xy_abs_local, warp_storage_base);
            }
            rocprim_warpSum<GSPLAT_WARP_SIZE>(v_opacity_local, warp_storage_base);
#    else
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if(v_means2d_abs != nullptr)
            {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
#    endif
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#    if USE_ROCM
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    unsafeAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                unsafeAtomicAdd(v_conic_ptr, v_conic_local.x);
                unsafeAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                unsafeAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                unsafeAtomicAdd(v_xy_ptr, v_xy_local.x);
                unsafeAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    unsafeAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    unsafeAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                unsafeAtomicAdd(v_opacities + g, v_opacity_local);
#    else
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    atomicAdd_system(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                atomicAdd_system(v_conic_ptr, v_conic_local.x);
                atomicAdd_system(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd_system(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                atomicAdd_system(v_xy_ptr, v_xy_local.x);
                atomicAdd_system(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    atomicAdd_system(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd_system(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                atomicAdd_system(v_opacities + g, v_opacity_local);
#    endif
            }
        }
    }
}

void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N           = packed ? 0 : means2d.size(-2);                        // number of gaussians
    uint32_t I           = render_alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads     = {tile_size, tile_size, 1};
    uint32_t n_tiles = I * tile_height * tile_width;
    dim3 grid        = {n_tiles, 1, 1};

    if(n_isects == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );

    auto launch_kernel = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

#    if USE_ROCM
        const uint32_t block_size = tile_size * tile_size;
        const uint32_t warps_per_block = (block_size + 31) / 32;
        std::size_t warp_scratch_bytes
            = warps_per_block * sizeof(typename rocprim::warp_reduce<float, GSPLAT_WARP_SIZE>::storage_type);
        int64_t shmem_size = tile_size
                           * tile_size
                           * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM)
                           + warp_scratch_bytes;
#    else
        int64_t shmem_size
            = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
#    endif

#    if USE_ROCM
        if constexpr(CDIM <= 32)
        {
            if(rasterize_blocks_to_pixels_3dgs_bwd_fits(means2d, colors, tile_size, flatten_ids))
            {
                const int64_t n_gauss = means2d.numel() / 2;
                auto launch_blocks    = [&]<bool ABS>()
                {
                    at::Tensor part     = at::empty({n_isects, CDIM + (ABS ? 8 : 6)}, means2d.options());
                    const int32_t *perm = flatten_ids.const_data_ptr<int32_t>() + n_isects;
                    rasterize_blocks_to_pixels_3dgs_bwd_kernel<CDIM, ABS><<<grid, 128, 0, GET_CURRENT_STREAM()>>>(
                        I,
                        n_isects,
                        reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                        reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                        colors.const_data_ptr<float>(),
                        opacities.const_data_ptr<float>(),
                        backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                        masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                        image_width,
                        image_height,
                        tile_width,
                        tile_height,
                        tile_offsets.const_data_ptr<int64_t>(),
                        flatten_ids.const_data_ptr<int32_t>(),
                        render_alphas.const_data_ptr<float>(),
                        last_ids.const_data_ptr<int32_t>(),
                        v_render_colors.const_data_ptr<float>(),
                        v_render_alphas.const_data_ptr<float>(),
                        part.data_ptr<float>()
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    gather_isect_grads_kernel<CDIM, ABS><<<(n_gauss + 255) / 256, 256, 0, GET_CURRENT_STREAM()>>>(
                        n_gauss,
                        reinterpret_cast<const int64_t *>(perm + n_isects),
                        part.const_data_ptr<float>(),
                        ABS ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>()) : nullptr,
                        reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                        reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                        v_colors.data_ptr<float>(),
                        v_opacities.data_ptr<float>()
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                };
                if(v_means2d_abs.has_value())
                {
                    launch_blocks.template operator()<true>();
                }
                else
                {
                    launch_blocks.template operator()<false>();
                }
                return;
            }
        }

        uint32_t bs64_max_batch_size = 0;
        if(rocm_use_bs64_bwd<CDIM>(tile_size, bs64_max_batch_size, shmem_size))
        {
            hipError_t bs64_err = hipFuncSetAttribute(
                reinterpret_cast<void *>(rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem_size)
            );
            if(bs64_err != hipSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size. HIP Error: ",
                    hipGetErrorString(bs64_err)
                );
            }
            rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float>
                <<<grid, threads, shmem_size, GET_CURRENT_STREAM()>>>(
                    I,
                    N,
                    n_isects,
                    packed,
                    reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                    reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                    masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                    image_width,
                    image_height,
                    tile_size,
                    tile_width,
                    tile_height,
                    0,
                    tile_offsets.const_data_ptr<int64_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    render_alphas.const_data_ptr<float>(),
                    last_ids.const_data_ptr<int32_t>(),
                    v_render_colors.const_data_ptr<float>(),
                    v_render_alphas.const_data_ptr<float>(),
                    v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                              : nullptr,
                    reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                    reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    bs64_max_batch_size
                );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
#    endif

#    ifndef USE_ROCM
        if(cudaFuncSetAttribute(
               rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }
#    else
        hipError_t err = hipFuncSetAttribute(
            reinterpret_cast<void *>(rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem_size)
        );
        if(err != hipSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes), try lowering tile_size. HIP Error: ",
                hipGetErrorString(err)
            );
        }
#    endif

        rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>
            <<<grid, threads, shmem_size, GET_CURRENT_STREAM()>>>(
                I,
                N,
                n_isects,
                packed,
                reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                colors.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                0,
                tile_offsets.const_data_ptr<int64_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
                render_alphas.const_data_ptr<float>(),
                last_ids.const_data_ptr<int32_t>(),
                v_render_colors.const_data_ptr<float>(),
                v_render_alphas.const_data_ptr<float>(),
                v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>()) : nullptr,
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

void launch_rasterize_to_pixels_3dgs_bwd_kernels(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N           = packed ? 0 : means2d.size(-2);
    uint32_t I           = render_alphas.numel() / (image_height * image_width);
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);
    uint32_t n_tiles     = I * tile_height * tile_width;

    dim3 threads = {tile_size, tile_size, 1};

    if(n_isects == 0)
    {
        return;
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );

    auto launch_kernels = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

#    if USE_ROCM
        const uint32_t block_size = tile_size * tile_size;
        const uint32_t warps_per_block = (block_size + 31) / 32;
        std::size_t warp_scratch_bytes
            = warps_per_block * sizeof(typename rocprim::warp_reduce<float, GSPLAT_WARP_SIZE>::storage_type);
        int64_t shmem_size = tile_size
                           * tile_size
                           * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM)
                           + warp_scratch_bytes;
#    else
        int64_t shmem_size
            = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
#    endif
#    if USE_ROCM
        uint32_t bs64_max_batch_size = 0;
        const bool use_bs64          = rocm_use_bs64_bwd<CDIM>(tile_size, bs64_max_batch_size, shmem_size);
#    endif

        for(const auto device_id: c10::irange(c10::cuda::device_count()))
        {
            C10_CUDA_CHECK(cudaSetDevice(device_id));
#    ifndef USE_ROCM
            if(cudaFuncSetAttribute(
                   rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   shmem_size
               )
               != cudaSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size."
                );
            }
#    else
            hipError_t err = hipFuncSetAttribute(
                use_bs64 ? reinterpret_cast<void *>(rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float>)
                         : reinterpret_cast<void *>(rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem_size)
            );
            if(err != hipSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size. HIP Error: ",
                    hipGetErrorString(err)
                );
            }
#    endif
            auto stream = GET_CURRENT_STREAM();

            int64_t block_offset, block_count;
            std::tie(block_offset, block_count) = chunk(n_tiles, device_id);
            if(block_count > 0)
            {
                dim3 grid = {static_cast<uint32_t>(block_count), 1, 1};
#    if USE_ROCM
                if(use_bs64)
                {
                    rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float><<<grid, threads, shmem_size, stream>>>(
                        I,
                        N,
                        n_isects,
                        packed,
                        reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                        reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                        colors.const_data_ptr<float>(),
                        opacities.const_data_ptr<float>(),
                        backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                        masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                        image_width,
                        image_height,
                        tile_size,
                        tile_width,
                        tile_height,
                        block_offset,
                        tile_offsets.const_data_ptr<int64_t>(),
                        flatten_ids.const_data_ptr<int32_t>(),
                        render_alphas.const_data_ptr<float>(),
                        last_ids.const_data_ptr<int32_t>(),
                        v_render_colors.const_data_ptr<float>(),
                        v_render_alphas.const_data_ptr<float>(),
                        v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                                  : nullptr,
                        reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                        reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                        v_colors.data_ptr<float>(),
                        v_opacities.data_ptr<float>(),
                        bs64_max_batch_size
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    continue;
                }
#    endif
                rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float><<<grid, threads, shmem_size, stream>>>(
                    I,
                    N,
                    n_isects,
                    packed,
                    reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                    reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                    masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                    image_width,
                    image_height,
                    tile_size,
                    tile_width,
                    tile_height,
                    block_offset,
                    tile_offsets.const_data_ptr<int64_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    render_alphas.const_data_ptr<float>(),
                    last_ids.const_data_ptr<int32_t>(),
                    v_render_colors.const_data_ptr<float>(),
                    v_render_alphas.const_data_ptr<float>(),
                    v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                              : nullptr,
                    reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                    reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>()
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernels));
    TORCH_CHECK(
        dispatched,
        "dispatch failed: no matching compile-time instantiation for runtime "
        "parameters"
    );

    merge_streams();
}
} // namespace gsplat

#endif

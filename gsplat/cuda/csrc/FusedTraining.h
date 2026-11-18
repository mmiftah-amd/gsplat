// SPDX-License-Identifier: Apache-2.0

// The kernels of gsplat/optimizers/fused_training.py, on ROCm.

#pragma once

#include <vector>

#include <ATen/core/Tensor.h>

namespace gsplat
{
#ifdef USE_ROCM
void l1_ssim(
    at::Tensor render,
    at::Tensor target,
    at::Tensor dm,
    at::optional<at::Tensor> v_loss,
    at::Tensor out,
    double l1_scale,
    double ssim_scale
);
at::Tensor fused_sh(
    int64_t degree,
    at::optional<at::Tensor> means,
    at::optional<at::Tensor> c2w,
    at::optional<at::Tensor> radii,
    std::vector<at::Tensor> sh,
    std::vector<at::Tensor> m,
    std::vector<at::Tensor> v,
    at::Tensor state,
    at::Tensor pend,
    at::Tensor table,
    int64_t t,
    double lr_scale,
    double b1,
    double b2,
    double eps
);
void fused_geometry(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> first,
    std::vector<at::Tensor> second,
    std::vector<double> rates,
    at::Tensor view,
    at::Tensor K,
    int64_t width,
    int64_t height,
    at::Tensor radii,
    at::Tensor conics,
    at::Tensor d_xy,
    at::Tensor d_conic,
    at::Tensor v_opacity,
    at::Tensor v_colors,
    at::Tensor state,
    at::Tensor pend,
    int64_t degree,
    double b1,
    double b2,
    double eps,
    double vscale,
    double opacity_reg,
    double scale_reg,
    at::optional<at::Tensor> noise,
    double noise_scale,
    double noise_t,
    double noise_k,
    int64_t camera_model,
    double eps2d,
    at::optional<at::Tensor> compensations
);
#endif
} // namespace gsplat

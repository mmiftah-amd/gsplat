// SPDX-License-Identifier: Apache-2.0

// The loss and optimizer step of gsplat/optimizers/fused_training.py on ROCm.
#include "Common.h"
#include "FusedTraining.h"
#include "Utils.cuh"
#include <ATen/Functions.h>

namespace gsplat
{
#ifdef USE_ROCM

// Basis k of the real SH of degree <= 3 at direction d, for a lane that owns one
// basis function.
inline __device__ float sh_basis3(const int k, const vec3 d)
{
    if(k == 0)
    {
        return 0.2820947917738781f;
    }
    const float r = rsqrtf(glm::dot(d, d)), x = d.x * r, y = d.y * r, z = d.z * r;
    const float xy = 2.f * x * y, c = x * x - y * y, z2 = z * z;
    const float e = -2.285228997322329f * z2 + 0.4570457994644658f;
    float b       = 0.f;
    b             = k == 1 ? -0.48860251190292f * y : b;
    b             = k == 2 ? 0.48860251190292f * z : b;
    b             = k == 3 ? -0.48860251190292f * x : b;
    b             = k == 4 ? 0.5462742152960395f * xy : b;
    b             = k == 5 ? -1.092548430592079f * z * y : b;
    b             = k == 6 ? 0.9461746957575601f * z2 - 0.3153915652525201f : b;
    b             = k == 7 ? -1.092548430592079f * z * x : b;
    b             = k == 8 ? 0.5462742152960395f * c : b;
    b             = k == 9 ? -0.5900435899266435f * (x * xy + y * c) : b;
    b             = k == 10 ? 1.445305721320277f * z * xy : b;
    b             = k == 11 ? e * y : b;
    b             = k == 12 ? z * (1.865881662950577f * z2 - 1.119528997770346f) : b;
    b             = k == 13 ? e * x : b;
    b             = k == 14 ? 1.445305721320277f * z * c : b;
    b             = k == 15 ? -0.5900435899266435f * (x * c - y * xy) : b;
    return b;
}

// Sum over the 16 lanes of a DPP row; every lane receives it.
inline __device__ float row16_sum(float v)
{
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x140, 0xF, 0xF, false));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x141, 0xF, 0xF, false));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x1B, 0xF, 0xF, false));
    return v + __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));
}

// Delayed lazy SH Adam. The geometry step only records a Gaussian's color gradient;
// the next render (or flush) applies that update and replays the zero-gradient steps
// since. This gives the parameters of an update at every step, with one load of the
// coefficients and moments per render.
//
// state[g] holds the last applied step << 6, the channels the last render's clamp
// passed << 3, and 0 or 1 + the SH degree of a pending update, whose color gradient
// is pend[g, 3:6] and direction pend[g, 0:3]. table[s] holds step s's bias-corrected
// step size and 1 / sqrt(1 - b2^s). shN has K - 1 <= 15 coefficients per gaussian.
struct SHState
{
    float *sh0, *m0, *v0, *shN, *mN, *vN;
    int K;
};

inline __device__ bool sh_catch_up(
    float c[3],
    float m[3],
    float v[3],
    const int k,
    const int st,
    const float *__restrict__ pend,
    const float2 *__restrict__ table,
    const int t,
    const float lr_scale,
    const float b1,
    const float b2,
    const float eps
)
{
    const int code     = st & 7;
    int applied        = st >> 6;
    const bool touched = code != 0 || applied < t;
    if(code)
    {
        const float basis = k < code * code ? sh_basis3(k, vec3(pend[0], pend[1], pend[2])) : 0.f;
        const float2 step = table[++applied];
        for(int ch = 0; ch < 3; ++ch)
        {
            const float grad  = basis * pend[3 + ch];
            m[ch]             = b1 * m[ch] + (1.f - b1) * grad;
            v[ch]             = b2 * v[ch] + (1.f - b2) * grad * grad;
            c[ch]            -= lr_scale * step.x * m[ch] / (sqrtf(v[ch]) * step.y + eps);
        }
    }
    for(int s = applied + 1; s <= t; ++s)
    {
        const float2 step = table[s];
        for(int ch = 0; ch < 3; ++ch)
        {
            m[ch] *= b1;
            v[ch] *= b2;
            c[ch] -= lr_scale * step.x * m[ch] / (sqrtf(v[ch]) * step.y + eps);
        }
    }
    return touched;
}

// Leaves a render's color gradient vc of a Gaussian for its next sh_catch_up().
inline __device__ void sh_pend(int32_t &st, float *__restrict__ pend, const float *__restrict__ vc, const int degree)
{
    pend[3]  = st & 8 ? vc[0] : 0.f;
    pend[4]  = st & 16 ? vc[1] : 0.f;
    pend[5]  = st & 32 ? vc[2] : 0.f;
    st      |= degree + 1;
}

// Brings the Gaussians the view sees (with radii, else all) to step t, and with radii
// writes the colors max(SH + 0.5, 0). 16 lanes per Gaussian; lane k < K owns coefficient
// k, in sh0 for k = 0, whose learning rate is lr_scale times that of shN.
__global__ void fused_sh_kernel(
    const int N,
    const int degree,
    const vec3 *__restrict__ means,
    const float *__restrict__ c2w,
    const int32_t *__restrict__ radii,
    const SHState s,
    int32_t *__restrict__ state,
    float *__restrict__ pend,
    const float2 *__restrict__ table,
    const int t,
    const float lr_scale,
    const float b1,
    const float b2,
    const float eps,
    float *__restrict__ colors
)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, g = tid / 16;
    const int k   = tid % 16;
    const bool in = g < N, flush = radii == nullptr;
    const bool live = in && k < s.K && (flush || (radii[2 * g] > 0 && radii[2 * g + 1] > 0));
    const int st    = live ? state[g] : 0;
    float col[3]    = {0.f, 0.f, 0.f};
    vec3 d(0.f);
    if(live)
    {
        const int64_t i0 = g * 3, iN = (g * (s.K - 1) + k - 1) * 3;
        float *cp = k ? s.shN + iN : s.sh0 + i0, *mp = k ? s.mN + iN : s.m0 + i0, *vp = k ? s.vN + iN : s.v0 + i0;
        float c[3], mm[3], vv[3];
        for(int ch = 0; ch < 3; ++ch)
        {
            c[ch]  = cp[ch];
            mm[ch] = mp[ch];
            vv[ch] = vp[ch];
        }
        if(sh_catch_up(c, mm, vv, k, st, pend + g * 6, table, t, k ? 1.f : lr_scale, b1, b2, eps))
        {
            for(int ch = 0; ch < 3; ++ch)
            {
                cp[ch] = c[ch];
                mp[ch] = mm[ch];
                vp[ch] = vv[ch];
            }
        }
        if(!flush)
        {
            d                 = means[g] - vec3(c2w[3], c2w[7], c2w[11]);
            const float basis = k < (degree + 1) * (degree + 1) ? sh_basis3(k, d) : 0.f;
            for(int ch = 0; ch < 3; ++ch)
            {
                col[ch] = basis * c[ch];
            }
        }
    }
    if(flush)
    {
        if(in && k == 0)
        {
            state[g] = t << 6 | (st & 0x38);
        }
        return;
    }
    for(int ch = 0; ch < 3; ++ch)
    {
        col[ch] = row16_sum(col[ch]);
    }
    if(k != 0 || !in)
    {
        return;
    }
    int passed = 0;
    for(int ch = 0; ch < 3; ++ch)
    {
        const float value   = (live ? col[ch] : 0.f) + 0.5f;
        colors[g * 3 + ch]  = max(value, 0.f);
        passed             |= (live && value >= 0.f) << ch;
    }
    if(!live)
    {
        return;
    }
    state[g]        = t << 6 | passed << 3;
    pend[g * 6]     = d.x;
    pend[g * 6 + 1] = d.y;
    pend[g * 6 + 2] = d.z;
}

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
)
{
    DEVICE_GUARD(table);
    const int64_t N = state.numel();
    TORCH_CHECK(sh.size() == 2 && m.size() == 2 && v.size() == 2, "sh, m and v must hold two tensors: sh0's and shN's");
    for(const auto *tensors: {&sh, &m, &v})
    {
        for(const at::Tensor &x: *tensors)
        {
            TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "sh, m and v must hold contiguous CUDA tensors");
        }
    }
    CHECK_INPUT(state);
    CHECK_INPUT(pend);
    CHECK_INPUT(table);
    TORCH_CHECK(sh[1].dim() == 3 && sh[1].size(1) <= 15, "shN must have shape [N, K - 1, 3], K <= 16");
    TORCH_CHECK(
        sh[0].numel() == 3 * N
            && m[0].numel() == 3 * N
            && v[0].numel() == 3 * N
            && sh[1].size(0) == N
            && m[1].sizes() == sh[1].sizes()
            && v[1].sizes() == sh[1].sizes()
            && pend.numel() == 6 * N,
        "sh0, shN, their moments, state and pend must be of the same gaussians"
    );
    TORCH_CHECK(table.size(0) > t, "table must hold step ", t);
    if(radii)
    {
        CHECK_INPUT(means.value());
        CHECK_INPUT(c2w.value());
        CHECK_INPUT(radii.value());
        TORCH_CHECK(
            means.value().numel() == 3 * N && radii.value().numel() == 2 * N && c2w.value().numel() >= 12,
            "means [N, 3], radii [1, N, 2] and c2w [3, 4] must fit the state of ",
            N,
            " gaussians"
        );
    }
    at::Tensor colors = radii ? at::empty({N, 3}, table.options()) : at::Tensor();
    if(N)
    {
        fused_sh_kernel<<<(N * 16 + 255) / 256, 256, 0, GET_CURRENT_STREAM()>>>(
            N,
            degree,
            radii ? (const vec3 *)means.value().data_ptr<float>() : nullptr,
            radii ? c2w.value().data_ptr<float>() : nullptr,
            radii ? radii.value().data_ptr<int32_t>() : nullptr,
            SHState{
                sh[0].data_ptr<float>(),
                m[0].data_ptr<float>(),
                v[0].data_ptr<float>(),
                sh[1].data_ptr<float>(),
                m[1].data_ptr<float>(),
                v[1].data_ptr<float>(),
                int(sh[1].size(1)) + 1
            },
            state.data_ptr<int32_t>(),
            pend.data_ptr<float>(),
            (const float2 *)table.data_ptr<float>(),
            t,
            lr_scale,
            b1,
            b2,
            eps,
            radii ? colors.data_ptr<float>() : nullptr
        );
    }
    return colors;
}

// For a render that takes its colors from fused_sh(): leaves their gradients for the next
// fused_sh(), of the Gaussians the view saw.
__global__ void fused_sh_grad_kernel(
    const int N,
    const int degree,
    const int32_t *__restrict__ radii,
    const float *__restrict__ v_colors,
    int32_t *__restrict__ state,
    float *__restrict__ pend
)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g < N && radii[2 * g] > 0 && radii[2 * g + 1] > 0)
    {
        sh_pend(state[g], pend + g * 6, v_colors + g * 3, degree);
    }
}

void fused_sh_grad(at::Tensor v_colors, at::Tensor radii, int64_t degree, at::Tensor state, at::Tensor pend)
{
    DEVICE_GUARD(state);
    const int64_t N = state.numel();
    CHECK_INPUT(v_colors);
    CHECK_INPUT(radii);
    CHECK_INPUT(state);
    CHECK_INPUT(pend);
    TORCH_CHECK(
        v_colors.numel() == 3 * N && radii.numel() == 2 * N && pend.numel() == 6 * N,
        "v_colors [N, 3], radii [1, N, 2] and pend [N, 6] must fit the state of ",
        N,
        " gaussians"
    );
    if(N)
    {
        fused_sh_grad_kernel<<<(N + 255) / 256, 256, 0, GET_CURRENT_STREAM()>>>(
            N,
            degree,
            radii.data_ptr<int32_t>(),
            v_colors.data_ptr<float>(),
            state.data_ptr<int32_t>(),
            pend.data_ptr<float>()
        );
    }
}

// A parameter tensor with its bias-corrected step size, and the D values of a Gaussian.
struct AdamTensor
{
    float *p, *m, *v;
    float lr;
};

template<int D>
struct AdamValues
{
    float p[D], m[D], v[D], grad[D];

    inline __device__ void load(const AdamTensor &t, const int g)
    {
        for(int c = 0; c < D; ++c)
        {
            p[c]    = t.p[g * D + c];
            m[c]    = t.m[g * D + c];
            v[c]    = t.v[g * D + c];
            grad[c] = 0.f;
        }
    }

    inline __device__ void step(
        const AdamTensor &t, const float b1, const float b2, const float eps, const float vscale
    )
    {
        for(int c = 0; c < D; ++c)
        {
            m[c]  = b1 * m[c] + (1.f - b1) * grad[c];
            v[c]  = b2 * v[c] + (1.f - b2) * grad[c] * grad[c];
            p[c] -= t.lr * m[c] / (sqrtf(v[c]) * vscale + eps);
        }
    }

    inline __device__ void store(const AdamTensor &t, const int g) const
    {
        for(int c = 0; c < D; ++c)
        {
            t.p[g * D + c] = p[c];
            t.m[g * D + c] = m[c];
            t.v[g * D + c] = v[c];
        }
    }
};

// Adam on means, quats, log-scales and opacity logits, with the projection's VJP (as
// projection_ewa_3dgs_fused_bwd_kernel, for one camera), the gradients of the sigmoid
// and the regularizers (per value), the record of the delayed SH update and MCMC's noise.
// Unlike rasterization(), the position gets no gradient through the view direction of the
// SH colors.
__global__ void fused_geometry_kernel(
    const int N,
    const AdamTensor means,
    const AdamTensor quats,
    const AdamTensor log_scales,
    const AdamTensor opacities,
    const float *__restrict__ view,
    const float *__restrict__ K,
    const int width,
    const int height,
    const int *__restrict__ radii,
    const vec3 *__restrict__ conics,
    const vec2 *__restrict__ d_xy,
    const vec3 *__restrict__ d_conic,
    const float *__restrict__ v_opacity,
    const float *__restrict__ v_colors,
    const int channels,
    int32_t *__restrict__ state,
    float *__restrict__ pend,
    const int degree,
    const float b1,
    const float b2,
    const float eps,
    const float vscale,
    const float opacity_reg,
    const float scale_reg,
    const vec3 *__restrict__ noise,
    const float noise_scale,
    const float noise_t,
    const float noise_k,
    const CameraModelType camera_model,
    const float eps2d,
    const float *__restrict__ compensations
)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= N)
    {
        return;
    }
    const bool seen = radii[g * 2] > 0 && radii[g * 2 + 1] > 0;
    // Load everything before storing: the compiler cannot tell these do not alias.
    AdamValues<3> mean_s, scale_s;
    AdamValues<4> quat_s;
    AdamValues<1> opacity_s;
    mean_s.load(means, g);
    quat_s.load(quats, g);
    scale_s.load(log_scales, g);
    opacity_s.load(opacities, g);
    // d sigmoid(x) / dx = sigmoid(x) (1 - sigmoid(x))
    const float sig = 1.f / (1.f + __expf(-opacity_s.p[0]));
    float d_opacity = 0.f;
    if(seen)
    {
        const vec3 mean(mean_s.p[0], mean_s.p[1], mean_s.p[2]);
        const vec3 scale(__expf(scale_s.p[0]), __expf(scale_s.p[1]), __expf(scale_s.p[2]));
        const vec4 quat(quat_s.p[0], quat_s.p[1], quat_s.p[2], quat_s.p[3]);
        const vec3 conic = conics[g], dc = d_conic[g];
        mat2 inv(conic.x, conic.y, conic.y, conic.z);
        mat2 d_inv(dc.x, dc.y * .5f, dc.y * .5f, dc.z), d_cov2(0.f);
        inverse_vjp(inv, d_inv, d_cov2);
        d_opacity = v_opacity[g];
        if(compensations)
        {
            // antialiased: the rasterizer saw the opacity times the compensation
            add_blur_vjp(eps2d, inv, compensations[g], d_opacity * sig, d_cov2);
            d_opacity *= compensations[g];
        }
        const mat3 R(view[0], view[4], view[8], view[1], view[5], view[9], view[2], view[6], view[10]);
        const vec3 t(view[3], view[7], view[11]);
        mat3 covar, covar_c, d_R(0.f), d_covar(0.f), d_covar_c(0.f);
        vec3 mean_c, d_mean_c(0.f), d_mean(0.f), d_t(0.f), d_scale(0.f);
        vec4 d_quat(0.f);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
        posW2C(R, t, mean, mean_c);
        covarW2C(R, covar, covar_c);
        // the gradients of the colors, and of the depth when it is rendered as a fourth channel
        const float *vc = v_colors + (int64_t)g * channels;
        if(channels > 3)
        {
            d_mean_c.z = vc[3];
        }
        const float fx = K[0], fy = K[4], cx = K[2], cy = K[5];
        const vec2 d_mean2d = d_xy[g];
        if(camera_model == CameraModelType::ORTHO)
        {
            ortho_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, width, height, d_cov2, d_mean2d, d_mean_c, d_covar_c);
        }
        else if(camera_model == CameraModelType::FISHEYE)
        {
            fisheye_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, width, height, d_cov2, d_mean2d, d_mean_c, d_covar_c);
        }
        else
        {
            persp_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, width, height, d_cov2, d_mean2d, d_mean_c, d_covar_c);
        }
        posW2C_VJP(R, t, mean, d_mean_c, d_R, d_t, d_mean);
        covarW2C_VJP(R, covar, d_covar_c, d_R, d_covar);
        quat_scale_to_covar_vjp(quat, scale, quat_to_rotmat(quat), d_covar, d_quat, d_scale);
        for(int c = 0; c < 3; ++c)
        {
            mean_s.grad[c]  = d_mean[c];
            scale_s.grad[c] = d_scale[c] * scale[c];
        }
        for(int c = 0; c < 4; ++c)
        {
            quat_s.grad[c] = d_quat[c];
        }
        sh_pend(state[g], pend + g * 6, vc, degree);
    }
    opacity_s.grad[0] = (d_opacity + opacity_reg) * (sig * (1.f - sig));
    for(int c = 0; c < 3; ++c)
    {
        scale_s.grad[c] += scale_reg * __expf(scale_s.p[c]);
    }
    mean_s.step(means, b1, b2, eps, vscale);
    quat_s.step(quats, b1, b2, eps, vscale);
    scale_s.step(log_scales, b1, b2, eps, vscale);
    opacity_s.step(opacities, b1, b2, eps, vscale);
    if(noise)
    {
        // inject_noise_to_position of the stepped Gaussian
        mat3 covar;
        const vec4 q(quat_s.p[0], quat_s.p[1], quat_s.p[2], quat_s.p[3]);
        const vec3 sc(__expf(scale_s.p[0]), __expf(scale_s.p[1]), __expf(scale_s.p[2]));
        quat_scale_to_covar_preci(q, sc, &covar, nullptr);
        const float opacity = 1.f / (1.f + __expf(-opacity_s.p[0]));
        const float gate    = 1.f / (1.f + __expf(noise_k * (opacity - noise_t)));
        const vec3 delta    = covar * (noise[g] * (gate * noise_scale));
        for(int c = 0; c < 3; ++c)
        {
            mean_s.p[c] += delta[c];
        }
    }
    mean_s.store(means, g);
    quat_s.store(quats, g);
    scale_s.store(log_scales, g);
    opacity_s.store(opacities, g);
}

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
)
{
    DEVICE_GUARD(view);
    const int N = params[0].size(0);
    auto tensor = [&](int i)
    {
        return AdamTensor{
            params[i].data_ptr<float>(), first[i].data_ptr<float>(), second[i].data_ptr<float>(), float(rates[i])
        };
    };
    if(N)
    {
        fused_geometry_kernel<<<(N + 127) / 128, 128, 0, GET_CURRENT_STREAM()>>>(
            N,
            tensor(0),
            tensor(1),
            tensor(2),
            tensor(3),
            view.data_ptr<float>(),
            K.data_ptr<float>(),
            width,
            height,
            radii.data_ptr<int>(),
            (const vec3 *)conics.data_ptr<float>(),
            (const vec2 *)d_xy.data_ptr<float>(),
            (const vec3 *)d_conic.data_ptr<float>(),
            v_opacity.data_ptr<float>(),
            v_colors.data_ptr<float>(),
            v_colors.size(-1),
            state.data_ptr<int32_t>(),
            pend.data_ptr<float>(),
            degree,
            b1,
            b2,
            eps,
            vscale,
            opacity_reg,
            scale_reg,
            noise ? (const vec3 *)noise.value().data_ptr<float>() : nullptr,
            noise_scale,
            noise_t,
            noise_k,
            static_cast<CameraModelType>(camera_model),
            eps2d,
            compensations ? compensations.value().data_ptr<float>() : nullptr
        );
    }
}

// The photometric loss on [1, H, W, 3] images: this kernel forward stores the SSIM's
// three derivatives, and backward filters them by the chain rule. 16x16 tiles with a
// 5 pixel halo are filtered in shared memory along x, then y, a channel at a time.
// The SSIM is derived from fused-ssim (https://github.com/rahul-goel/fused-ssim), with its
// "valid" padding:
//
//   Copyright (c) 2024 Rahul Goel. MIT License.
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this
//   software and associated documentation files (the "Software"), to deal in the Software
//   without restriction, including without limitation the rights to use, copy, modify, merge,
//   publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
//   to whom the Software is furnished to do so, subject to the following conditions: The above
//   copyright notice and this permission notice shall be included in all copies or substantial
//   portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
//   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//   THE USE OR OTHER DEALINGS IN THE SOFTWARE.
__constant__ float ssim_window[11]
    = {0.00102838012f,
       0.00759875821f,
       0.0360007733f,
       0.109360687f,
       0.213005528f,
       0.266011715f,
       0.213005528f,
       0.109360687f,
       0.0360007733f,
       0.00759875821f,
       0.00102838012f};

template<int N, typename Fetch>
inline __device__ void ssim_filter(Fetch fetch, float s[N])
{
    for(int k = 0; k < N; ++k)
    {
        s[k] = 0.f;
    }
#    pragma unroll
    for(int d = 0; d < 11; ++d)
    {
        for(int k = 0; k < N; ++k)
        {
            s[k] += fetch(d, k) * ssim_window[d];
        }
    }
}

template<bool BWD>
__global__ void __launch_bounds__(256) l1_ssim_kernel(
    const int H,
    const int W,
    const float l1_scale,
    const float ssim_scale,
    const float *__restrict__ render,
    const float *__restrict__ target,
    float *__restrict__ dm,
    const float *__restrict__ v_loss,
    float *__restrict__ out
)
{
    constexpr int T = 16, R = T + 10, N = BWD ? 3 : 5;
    __shared__ float tile[BWD ? 3 : 2][R][R], along_x[R][T][N];
    const int x0 = blockIdx.x * T, y0 = blockIdx.y * T, tx = threadIdx.x % T, ty = threadIdx.x / T;
    const int px = x0 + tx, py = y0 + ty;
    const int64_t own = (int64_t)py * W + px, plane = (int64_t)H * W;
    const bool inside = px < W && py < H, valid = py >= 5 && py < H - 5 && px >= 5 && px < W - 5;
    float v_render[3], l1 = 0.f, ssim = 0.f;
    for(int c = 0; c < 3; ++c)
    {
        for(int i = threadIdx.x; i < R * R; i += T * T)
        {
            const int ly = i / R, lx = i % R, gy = y0 + ly - 5, gx = x0 + lx - 5;
            const bool in   = gy >= 0 && gy < H && gx >= 0 && gx < W;
            const int64_t p = (int64_t)gy * W + gx;
            if constexpr(BWD)
            {
                for(int k = 0; k < 3; ++k)
                {
                    tile[k][ly][lx] = in ? dm[(c * 3 + k) * plane + p] : 0.f;
                }
            }
            else
            {
                tile[0][ly][lx] = in ? render[p * 3 + c] : 0.f;
                tile[1][ly][lx] = in ? target[p * 3 + c] : 0.f;
            }
        }
        __syncthreads();
        for(int i = threadIdx.x; i < R * T; i += T * T)
        {
            const int ly = i / T, lx = i % T;
            float s[N];
            ssim_filter<N>(
                [&](int d, int k)
                {
                    if constexpr(BWD)
                    {
                        return tile[k][ly][lx + d];
                    }
                    const float a = tile[0][ly][lx + d], b = tile[1][ly][lx + d];
                    return k == 0 ? a : k == 1 ? a * a : k == 2 ? b : k == 3 ? b * b : a * b;
                },
                s
            );
            for(int k = 0; k < N; ++k)
            {
                along_x[ly][lx][k] = s[k];
            }
        }
        __syncthreads();
        float s[N];
        ssim_filter<N>([&](int d, int k) { return along_x[ty + d][tx][k]; }, s);
        const float a = inside ? render[own * 3 + c] : 0.f, b = inside ? target[own * 3 + c] : 0.f;
        if constexpr(BWD)
        {
            const float diff = a - b, sign = (diff > 0.f) - (diff < 0.f);
            v_render[c] = v_loss[0] * (l1_scale * sign - ssim_scale * (s[0] + 2.f * a * s[1] + b * s[2]));
        }
        else
        {
            float d0 = 0.f, d1 = 0.f, d2 = 0.f;
            if(valid)
            {
                const float m1 = s[0], m2 = s[2], m11 = m1 * m1, m22 = m2 * m2;
                const float A = m11 + m22 + 0.0001f;
                const float B = fmaxf(0.f, s[1] - m11) + fmaxf(0.f, s[3] - m22) + 0.0009f;
                const float C = 2.f * m1 * m2 + 0.0001f, D = 2.f * (s[4] - m1 * m2) + 0.0009f;
                const float val  = (C * D) / (A * B);
                ssim            += 1.f - fmaxf(-1.f, fminf(1.f, val));
                if(val >= -1.f && val <= 1.f)
                {
                    d0 = (m2 * 2.f * D - m2 * 2.f * C) / (A * B)
                       + m1 * 2.f * C * D * (1.f / (A * B * B) - 1.f / (A * A * B));
                    d1 = -C * D / (A * B * B);
                    d2 = 2.f * C / (A * B);
                }
            }
            l1 += fabsf(a - b);
            if(inside)
            {
                dm[c * 3 * plane + own]       = d0;
                dm[(c * 3 + 1) * plane + own] = d1;
                dm[(c * 3 + 2) * plane + own] = d2;
            }
        }
        __syncthreads();
    }
    if(inside)
    {
        if constexpr(BWD)
        {
            for(int c = 0; c < 3; ++c)
            {
                out[own * 3 + c] = v_render[c];
            }
        }
        else
        {
            out[own * 2]     = l1 * l1_scale;
            out[own * 2 + 1] = ssim * ssim_scale;
        }
    }
}

// The forward without v_loss, which writes dm and each pixel's two loss terms to out.
void l1_ssim(
    at::Tensor render,
    at::Tensor target,
    at::Tensor dm,
    at::optional<at::Tensor> v_loss,
    at::Tensor out,
    double l1_scale,
    double ssim_scale
)
{
    DEVICE_GUARD(render);
    CHECK_INPUT(render);
    CHECK_INPUT(target);
    TORCH_CHECK(
        render.dim() == 4 && render.size(0) == 1 && render.size(3) == 3 && target.sizes() == render.sizes(),
        "render and target must have the same shape [1, H, W, 3], got ",
        render.sizes(),
        " and ",
        target.sizes()
    );
    const int H = render.size(1), W = render.size(2);
    CHECK_INPUT(dm);
    CHECK_INPUT(out);
    TORCH_CHECK(
        dm.numel() == 9ll * H * W && out.numel() == (v_loss ? 3ll : 2ll) * H * W,
        "dm must hold 9 values per pixel and out 2, or 3 with v_loss"
    );
    if(v_loss)
    {
        CHECK_INPUT(v_loss.value());
        TORCH_CHECK(v_loss.value().numel() == 1, "v_loss must be a scalar");
    }
    auto kernel = v_loss ? l1_ssim_kernel<true> : l1_ssim_kernel<false>;
    kernel<<<dim3((W + 15) / 16, (H + 15) / 16), 256, 0, GET_CURRENT_STREAM()>>>(
        H,
        W,
        l1_scale,
        ssim_scale,
        render.data_ptr<float>(),
        target.data_ptr<float>(),
        dm.data_ptr<float>(),
        v_loss ? v_loss.value().data_ptr<float>() : nullptr,
        out.data_ptr<float>()
    );
}

#endif
} // namespace gsplat

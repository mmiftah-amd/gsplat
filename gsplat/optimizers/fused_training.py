# SPDX-License-Identifier: Apache-2.0
"""Fused single-view training of 3DGS on ROCm GPUs with 32-lane wavefronts.

FusedTraining replaces the render, the photometric loss and the optimizer step of a
trainer that keeps the usual splats and torch.optim.Adam optimizers. The updates are
Adam's, in the optimizers' own state, so strategies and checkpoints work as usual, but
for one difference, as in the original 3DGS: positions get no gradient through the view
direction of the SH colors. SH updates wait until a Gaussian is rendered: flush() before
reading sh0 or shN. The strategies leave PyTorch's memory cache alone meanwhile: memory
that is released and mapped again can stay slow for thousands of steps.
"""

import math

import torch
from torch import Tensor

from ..cuda._backend import _C
from ..cuda._wrapper import _make_lazy_cuda_func, _make_lazy_cuda_obj
from ..exporter import encode_morton3_vec
from ..strategy import MCMCStrategy
from ..strategy.ops import _update_param_with_optimizer

GEOMETRY = ("means", "quats", "scales", "opacities")
TILE_SIZE, EPS2D = 16, 0.3  # rasterization()'s defaults


class _Render(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, fused, *args):
        ctx.set_materialize_grads(False)
        ctx.fused = fused
        return fused._forward(*args)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, v_image, v_alpha):
        ctx.fused._backward(v_image, v_alpha)
        return (None,) * 12


class _L1SSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, render, target, weights, ssim_lambda):
        _, H, W, _ = render.shape
        ctx.scales = 1 / (3 * H * W), 1 / (3 * (H - 10) * (W - 10))
        ctx.ssim_lambda = ssim_lambda
        pixels, dm = render.new_empty(H, W, 2), render.new_empty(9, H, W)
        _C.l1_ssim(render, target, dm, None, pixels, *ctx.scales)
        ctx.save_for_backward(render, target, dm)
        sums = pixels.sum((0, 1))
        ctx.mark_non_differentiable(sums)
        return (sums * weights).sum(), sums

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, v_loss, _):
        render, target, dm = ctx.saved_tensors
        v_render = render.new_empty(render.shape)
        l1, ssim = ctx.scales
        _C.l1_ssim(
            render,
            target,
            dm,
            v_loss,
            v_render,
            (1.0 - ctx.ssim_lambda) * l1,
            ctx.ssim_lambda * ssim,
        )
        return v_render, None, None, None


class _CachedViews:
    """The decoded training views on the GPU, in random order per pass."""

    def __init__(self, dataset, device):
        self.views = []
        for data in torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=4
        ):
            data["image"] = data["image"].to(torch.uint8)  # pixel values 0-255
            view = {k: torch.as_tensor(v)[None].to(device) for k, v in data.items()}
            view["camtoworld"].viewmat = torch.linalg.inv(
                view["camtoworld"]
            ).contiguous()
            self.views.append(view)

    def __iter__(self):
        return (self.views[i] for i in torch.randperm(len(self.views)).tolist())


class FusedTraining:
    def __init__(self, splats, optimizers: dict, cfg, scene=None):
        self.splats, self.optimizers, self.scene = splats, optimizers, scene
        self.strategy, self.opacity_reg, self.scale_reg = (
            cfg.strategy,
            cfg.opacity_reg,
            cfg.scale_reg,
        )
        self.absgrad = getattr(cfg.strategy, "absgrad", False)
        self.antialiased = getattr(cfg, "antialiased", False)
        self.camera_model = _make_lazy_cuda_obj(
            f"CameraModelType.{getattr(cfg, 'camera_model', 'pinhole').upper()}"
        )
        for name, p in splats.items():
            # the state torch.optim.Adam makes on its first step; flush() sets the steps
            on_device = optimizers[name].param_groups[0].get("fused")
            step = torch.zeros((), device=p.device) if on_device else torch.tensor(0.0)
            zeros = {k: torch.zeros_like(p) for k in ("exp_avg", "exp_avg_sq")}
            optimizers[name].state[p] = {"step": step, **zeros}
        group = optimizers["means"].param_groups[0]
        self.b1, self.b2, self.eps = *group["betas"], group["eps"]
        lr_shN = optimizers["shN"].param_groups[0]["lr"]
        self.lr_sh0_scale = optimizers["sh0"].param_groups[0]["lr"] / lr_shN
        # the SH learning rates are constant, so every step's Adam factors are known
        t = torch.arange(cfg.max_steps + 2, dtype=torch.float64)
        table = torch.stack(
            [lr_shN / (1 - self.b1**t), (1 - self.b2**t) ** -0.5], -1
        )
        table[0] = 0.0
        self.table = table.float().to(splats["means"].device)
        self.t, self.state, self.flushed, self.loss_weights, self.no_v_alpha = (
            0,
            None,
            True,
            None,
            None,
        )
        self.side_stream = torch.cuda.Stream(device=self.table.device)
        splats.fused = self  # for the strategies' operations to call restructure()

    @staticmethod
    def supports(cfg, device) -> bool:
        """Whether FusedTraining covers the configuration of a simple_trainer.py."""
        unsupported = (
            "packed",
            "sparse_grad",
            "visible_adam",
            "pose_opt",
            "pose_noise",
            "app_opt",
            "with_ut",
            "with_eval3d",
            "sh_fp16",
            "post_processing",
            "compression",
        )
        return (
            bool(torch.version.hip)
            and torch.cuda.get_device_properties(device).warp_size == 32
            and (cfg.batch_size, cfg.data_type, cfg.backend) == (1, "colmap", "cuda")
            and cfg.camera_model in ("pinhole", "ortho", "fisheye")
            and cfg.sh_degree <= 3
            and not any(getattr(cfg, k) for k in unsupported)
        )

    def cached_views(self, trainset, trainloader):
        size = trainset[0]["image"].numel() * len(trainset)
        fits = (
            size <= torch.cuda.get_device_properties(self.table.device).total_memory / 4
        )
        if fits and trainset.patch_size is None:
            return _CachedViews(trainset, self.table.device)
        return trainloader

    def _moments(self, names):
        states = [self.optimizers[k].state[self.splats[k]] for k in names]
        return [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states]

    def _sh(self, degree=0, means=None, c2w=None, radii=None):
        return _C.fused_sh(
            degree,
            means,
            c2w,
            radii,
            [self.splats["sh0"], self.splats["shN"]],
            *self._moments(("sh0", "shN")),
            self.state,
            self.pend,
            self.table,
            self.t,
            self.lr_sh0_scale,
            self.b1,
            self.b2,
            self.eps,
        )

    @torch.no_grad()
    def flush(self):
        """Brings sh0, shN, their moments and the optimizers' step counts to what the Adam
        steps give."""
        if not self.flushed:
            self._sh()
            for name, p in self.splats.items():
                self.optimizers[name].state[p]["step"].fill_(self.t)
            self.flushed = True

    def restructure(self):
        """Before the Gaussians are added, removed or moved: their delayed state is per index."""
        self.flush()
        self.state = None

    @torch.no_grad()
    def _new_gaussians(self):
        """The delayed SH state of the Gaussians after a refinement. Under MCMC, they go in
        Morton order of the central 98% of the means: those a view sees are close in memory."""
        means = self.splats["means"]
        n, device = len(means), means.device
        if isinstance(self.strategy, MCMCStrategy):
            bounds = torch.tensor([0.01, 0.99], device=device)
            lo, hi = torch.quantile(means[:: max(n >> 16, 1)], bounds, dim=0)
            q = ((means - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1).mul_(1023).int()
            order = torch.argsort(encode_morton3_vec(q[:, 0], q[:, 1], q[:, 2]))
            _update_param_with_optimizer(
                lambda name, p: torch.nn.Parameter(
                    p[order], requires_grad=p.requires_grad
                ),
                lambda key, v: v[order],
                self.splats,
                self.optimizers,
            )
            if self.scene is not None:
                self.scene.on_permute(order)
        self.state = torch.full((n,), self.t << 6, dtype=torch.int32, device=device)
        self.pend = torch.empty(n, 6, device=device)

    def rasterize(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int = 3,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        masks=None,
        backgrounds=None,
        render_mode: str = "RGB",
        **kwargs,
    ):
        """rasterization() of a training view in the render modes "RGB", "RGB+D" and "RGB+ED",
        with the outputs a trainer uses."""
        if self.state is None:
            self._new_gaussians()
        view = getattr(camtoworlds, "viewmat", None)  # an inverse is eight kernels
        view = (
            torch.linalg.inv_ex(camtoworlds)[0].contiguous() if view is None else view
        )
        colors, alphas = _Render.apply(
            self.splats["means"],
            self,
            view,
            camtoworlds.contiguous(),
            Ks.contiguous(),
            width,
            height,
            sh_degree,
            near_plane,
            far_plane,
            backgrounds,
            render_mode != "RGB",
        )
        if render_mode == "RGB+ED":
            colors = torch.cat(
                [colors[..., :3], colors[..., 3:] / alphas.clamp(min=1e-10)], -1
            )
        if masks is not None:
            colors[~masks] = 0
        return colors, alphas, self.info

    def _forward(
        self, view, c2w, Ks, width, height, degree, near, far, backgrounds, depth
    ):
        op = _make_lazy_cuda_func
        means, opacities = self.splats["means"], torch.sigmoid(self.splats["opacities"])
        radii, means2d, depths, conics, compensations = op("projection_ewa_3dgs_fused")(
            means,
            None,
            self.splats["quats"],
            torch.exp(self.splats["scales"]),
            opacities,
            view,
            Ks,
            width,
            height,
            EPS2D,
            near,
            far,
            0.0,
            self.antialiased,
            self.camera_model,
        )
        # the colors run alongside the tile intersection, on a stream of their own
        main = torch.cuda.current_stream(means.device)
        self.side_stream.wait_stream(main)
        with torch.cuda.stream(self.side_stream):
            colors = self._sh(degree, means, c2w, radii)[None]
        tile_width, tile_height = math.ceil(width / TILE_SIZE), math.ceil(
            height / TILE_SIZE
        )
        opacities = (
            opacities[None] if compensations is None else opacities * compensations
        )
        _, isect_ids, flatten_ids = op("intersect_tile")(
            means2d,
            radii,
            depths,
            conics,
            opacities,
            None,
            None,
            None,
            TILE_SIZE,
            tile_width,
            tile_height,
            True,
            False,
        )
        offsets = op("intersect_offset")(isect_ids, 1, tile_width, tile_height)
        offsets = offsets.reshape(1, tile_height, tile_width)
        main.wait_stream(self.side_stream)
        if depth:  # a fourth channel, as rasterization() adds it
            colors = torch.cat([colors, depths[..., None]], -1)
            if backgrounds is not None:
                backgrounds = torch.nn.functional.pad(backgrounds, (0, 1))
        image, alpha, _, last_ids = op("rasterize_to_pixels_3dgs")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            None,
            width,
            height,
            TILE_SIZE,
            offsets,
            flatten_ids,
            False,
            False,
        )
        self.render = (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            None,
            offsets,
            flatten_ids,
            alpha,
            last_ids,
            width,
            height,
            TILE_SIZE,
            self.absgrad,
        )
        self.projection = (view, Ks, width, height, radii, conics)
        self.compensations = compensations
        self.degree = degree
        self.info = dict(
            means2d=means2d.requires_grad_(),
            radii=radii,
            width=width,
            height=height,
            n_cameras=1,
            gaussian_ids=None,
        )
        return image, alpha

    def _backward(self, v_image, v_alpha):
        alpha = self.render[8]
        if v_image is None:
            v_image = alpha.new_zeros(*alpha.shape[:-1], self.render[2].shape[-1])
        if v_alpha is None:  # the loss did not use the alpha
            if self.no_v_alpha is None or self.no_v_alpha.shape != alpha.shape:
                self.no_v_alpha = torch.zeros_like(alpha)
            v_alpha = self.no_v_alpha
        absgrad, *self.grads, _ = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
            *self.render, v_image, v_alpha, False
        )
        self.info["means2d"].grad = self.grads[0]  # for the default strategy
        self.info["means2d"].absgrad = absgrad

    def loss(self, colors: Tensor, pixels: Tensor, ssim_lambda: float):
        """loss, l1 and 1 - ssim as l1_loss().mean() and ssim_loss() give them, without copies."""
        if self.loss_weights is None:
            self.loss_weights = pixels.new_tensor([1.0 - ssim_lambda, ssim_lambda])
        loss, sums = _L1SSIM.apply(
            colors.contiguous(), pixels, self.loss_weights, ssim_lambda
        )
        return loss, sums[0], sums[1]

    @torch.no_grad()
    def step(self, step: int, gamma: float):
        """The optimizer step; gamma is the means' learning rate schedule. It includes MCMC's
        position noise, unless the strategy refines at this step, and adds it after that."""
        s, means = self.strategy, self.splats["means"]
        mcmc, n = isinstance(s, MCMCStrategy), len(means)
        noise = None
        if mcmc and not (
            s.refine_start_iter < step < s.refine_stop_iter
            and step % s.refine_every == 0
        ):
            if s.noise_injection_stop_iter < 0 or step < s.noise_injection_stop_iter:
                noise = torch.randn_like(means)
        noise_scale = (
            self.optimizers["means"].param_groups[0]["lr"] * gamma * s.noise_lr
            if mcmc
            else 0.0
        )
        self.splats.noise_added = noise is not None
        self.t += 1
        b1_t, vscale = 1 - self.b1**self.t, (1 - self.b2**self.t) ** -0.5
        v_means2d, v_conics, v_colors, v_opacities = self.grads
        _C.fused_geometry(
            [self.splats[k] for k in GEOMETRY],
            *self._moments(GEOMETRY),
            [self.optimizers[k].param_groups[0]["lr"] / b1_t for k in GEOMETRY],
            *self.projection,
            v_means2d,
            v_conics,
            v_opacities,
            v_colors,
            self.state,
            self.pend,
            self.degree,
            self.b1,
            self.b2,
            self.eps,
            vscale,
            self.opacity_reg / n,
            self.scale_reg / (3 * n),
            noise,
            noise_scale,
            getattr(s, "noise_opacity_t", 0.0),
            getattr(s, "noise_opacity_k", 0.0),
            int(self.camera_model),
            EPS2D,
            self.compensations,
        )
        self.flushed = False

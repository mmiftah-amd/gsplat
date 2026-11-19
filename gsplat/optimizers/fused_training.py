# SPDX-License-Identifier: Apache-2.0
"""Fused single-view training of 3DGS on ROCm GPUs with 32-lane wavefronts.

DelayedSH is the part any training loop can use: it stands in for the SH coefficients in
rasterization() and applies their Adam steps when a Gaussian is next rendered.
FusedTraining adds the render, the photometric loss and the rest of the optimizer step of
a trainer that keeps the usual splats and torch.optim.Adam optimizers. The updates are
Adam's, in the optimizers' own state, so strategies and checkpoints work as usual, but for
one difference from rasterization(): positions and camera poses get no gradient through
the view direction of the SH colors. SH updates wait until a Gaussian is rendered: flush()
before reading sh0 or shN. The strategies leave PyTorch's memory cache alone meanwhile:
memory that is released and mapped again can stay slow for thousands of steps.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..cuda._backend import _C
from ..cuda._wrapper import (
    _make_lazy_cuda_func,
    _make_lazy_cuda_obj,
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)
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


class _Colors(torch.autograd.Function):
    """The colors of a view from sh0 and shN, on DelayedSH's stream. The backward keeps
    their gradient for the next delayed step, on the caller's stream as any other, and
    gives sh0 and shN none: their optimizers leave them alone."""

    @staticmethod
    def forward(ctx, sh0, shN, sh, degree, means, camera, radii):
        ctx.set_materialize_grads(False)
        sh.renders += 1
        ctx.render = sh, radii, degree, (sh.t, sh.state, sh.renders)
        sh.side_stream.wait_stream(torch.cuda.current_stream(means.device))
        with torch.cuda.stream(sh.side_stream):
            return sh._sh(degree, means, camera, radii)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, v_colors):
        sh, radii, degree, render = ctx.render
        if v_colors is not None:
            t, state, renders = render
            if t != sh.t or state is not sh.state or renders != sh.renders:
                raise RuntimeError(
                    "DelayedSH takes the gradient of its last render, once, and before "
                    "the Gaussians change."
                )
            if len(sh.awaiting) == len(sh.steppers):
                raise RuntimeError(
                    "DelayedSH takes one gradient per step of the optimizers of sh0 and "
                    "shN: its own step is taken in backward()."
                )
            sh.awaiting = {id(o) for o in sh.steppers}
            sh._step()
            _C.fused_sh_grad(v_colors.contiguous(), radii, degree, sh.state, sh.pend)
        return (None,) * 7


class _L1SSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, render, target, ssim_lambda):
        _, H, W, _ = render.shape
        ctx.scales = 1 / (3 * H * W), 1 / (3 * (H - 10) * (W - 10))
        ctx.ssim_lambda = ssim_lambda
        pixels, dm = render.new_empty(H, W, 2), render.new_empty(9, H, W)
        _C.l1_ssim(render, target, dm, None, pixels, *ctx.scales)
        ctx.save_for_backward(render, target, dm)
        sums = pixels.sum((0, 1))
        ctx.mark_non_differentiable(sums)
        return torch.lerp(sums[0], sums[1], ssim_lambda), sums

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
        return v_render, None, None


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


def cached_views(trainset, trainloader, device):
    """The training views of a simple_trainer.py on the GPU when they fit in a quarter of
    its memory, else its loader."""
    size = trainset[0]["image"].numel() * len(trainset)
    fits = size <= torch.cuda.get_device_properties(device).total_memory / 4
    if fits and trainset.patch_size is None:
        return _CachedViews(trainset, device)
    return trainloader


@torch.no_grad()
def morton_order(splats, optimizers: dict, scene=None):
    """Puts the Gaussians in Morton order of the central 98% of the means: those a view
    sees are then close in memory. For a strategy that keeps no state per Gaussian, as
    MCMCStrategy, between its refinements and outside a step; `scene` is told the order."""
    means = splats["means"]
    n, device = len(means), means.device
    bounds = torch.tensor([0.01, 0.99], device=device)
    lo, hi = torch.quantile(means[:: max(n >> 16, 1)], bounds, dim=0)
    q = ((means - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1).mul_(1023).int()
    order = torch.argsort(encode_morton3_vec(q[:, 0], q[:, 1], q[:, 2]))
    _update_param_with_optimizer(
        lambda name, p: torch.nn.Parameter(p[order], requires_grad=p.requires_grad),
        lambda key, v: v[order],
        splats,
        optimizers,
    )
    if scene is not None:
        scene.on_permute(order)


class DelayedSH:
    """sh0 and shN whose Adam steps are applied when a Gaussian is next rendered, on ROCm
    GPUs with 32-lane wavefronts.

    Pass it to rasterization() as `colors`, in place of torch.cat([sh0, shN], 1), in a
    loop that renders one view per optimizer step, with packed=False. Positions and camera
    poses then get no gradient through the view direction of the colors.

    `splats` and `optimizers` hold sh0 [N, 1, 3] and shN [N, K - 1, 3], K <= 16, by these
    names, as gsplat's strategies expect, each with a torch.optim.Adam of the same betas
    and eps. The loop steps its optimizers as before: these two get no gradient, so their
    optimizers leave them alone, and a gradient from elsewhere, as of a regularizer, is an
    error. The learning rates may change, in a constant ratio, between steps. Each
    backward() of a render takes the SH step, so a step of the optimizers of sh0 and shN
    follows it, and no other render comes between a render and its backward().

    sh0, shN and their moments are behind until flush(). Call restructure() before
    Gaussians are added, removed or moved; gsplat's strategies do. close() ends it. A
    render changes this state: one thread at a time.
    """

    names = ("sh0", "shN")

    def __init__(self, splats, optimizers: dict):
        sh0, shN = splats["sh0"], splats["shN"]
        if not (torch.version.hip and sh0.is_cuda):
            raise RuntimeError("DelayedSH needs PyTorch for ROCm and its GPU.")
        if torch.cuda.get_device_properties(sh0.device).warp_size != 32:
            raise RuntimeError("DelayedSH is for GPUs with 32-lane wavefronts.")
        if not (
            sh0.shape[1:] == (1, 3)
            and shN.shape[0::2] == (len(sh0), 3)
            and shN.shape[1] <= 15
            and sh0.dtype == shN.dtype == torch.float32
        ):
            raise ValueError(
                "DelayedSH takes float32 sh0 [N, 1, 3] and shN [N, <= 15, 3]."
            )
        if not hasattr(splats, "__dict__"):  # the strategies' operations find it there
            raise TypeError(
                "DelayedSH needs splats that take an attribute, as ParameterDict."
            )
        previous = getattr(splats, "fused", None)
        if previous is not None:
            previous.close()
        self.splats, self.optimizers = splats, optimizers
        group = self._group("shN")
        if not group["lr"] > 0:
            raise ValueError("DelayedSH needs a positive learning rate of shN.")
        self.b1, self.b2, self.eps = *group["betas"], group["eps"]
        for name in self.names:
            p, optimizer, g = splats[name], optimizers[name], self._group(name)
            if (
                type(optimizer) is not torch.optim.Adam
                or (*g["betas"], g["eps"]) != (self.b1, self.b2, self.eps)
                or g["weight_decay"]
                or g["amsgrad"]
                or g["maximize"]
            ):
                raise ValueError(
                    f"DelayedSH needs a torch.optim.Adam for {name} with the betas and "
                    "eps of shN's, and without weight decay, amsgrad or maximize."
                )
            if p not in optimizer.state:
                # the state torch.optim.Adam makes on its first step; flush() sets the steps
                on_device = g.get("fused")
                step = (
                    torch.zeros((), device=p.device) if on_device else torch.tensor(0.0)
                )
                zeros = {k: torch.zeros_like(p) for k in ("exp_avg", "exp_avg_sq")}
                optimizer.state[p] = {"step": step, **zeros}
        self.steppers = list(
            {id(optimizers[k]): optimizers[k] for k in ("sh0", "shN")}.values()
        )
        self.hooks = [o.register_step_pre_hook(self._on_step) for o in self.steppers]
        self.hooks += [
            o.register_step_post_hook(self._no_gradient) for o in self.steppers
        ]
        self.lr_sh0_scale = self._group("sh0")["lr"] / group["lr"]
        self.t = int(optimizers["shN"].state[shN]["step"])
        self.table = torch.zeros(self.t + 1, 2, device=shN.device)
        self.lr, self.state, self.pend, self.flushed = None, None, None, True
        self.renders, self.awaiting = (
            0,
            set(),
        )  # the steppers still to follow backward()
        self.side_stream = torch.cuda.Stream(device=self.table.device)
        splats.fused = self  # for the strategies' operations to call restructure()

    def __reduce__(self):
        """A copy or a pickle of the splats, which must be up to date, leaves this out."""
        if not self.flushed:
            raise RuntimeError(
                "DelayedSH: flush() before the splats are copied or pickled."
            )
        return type(None), ()

    def _group(self, name):
        p = self.splats[name]
        for group in self.optimizers[name].param_groups:
            if any(q is p for q in group["params"]):
                return group
        raise ValueError(
            f"DelayedSH: optimizers[{name!r}] does not hold splats[{name!r}]."
        )

    def _no_gradient(self, optimizer=None, args=None, kwargs=None):
        if any(self.splats[k].grad is not None for k in ("sh0", "shN")):
            raise RuntimeError(
                "sh0 or shN has a gradient that did not come through DelayedSH, which "
                "steps them: as from a regularizer, or a render it does not cover."
            )

    def _on_step(self, optimizer, args, kwargs):
        """A step of sh0's or shN's optimizer: after backward(), the one DelayedSH took."""
        self._no_gradient()
        if id(optimizer) in self.awaiting:
            self.awaiting.discard(id(optimizer))
            for name, lr in (("shN", self.lr), ("sh0", self.lr_sh0_scale * self.lr)):
                group = self._group(name)
                if self.optimizers[name] is optimizer and not math.isclose(
                    group["lr"], lr, rel_tol=1e-6
                ):
                    raise RuntimeError(
                        f"DelayedSH: the learning rate of {name} changed between "
                        "backward() and its optimizer's step."
                    )

    def close(self):
        """Ends the delayed steps: sh0, shN and their optimizers' state are up to date, for
        the optimizers to go on with them."""
        self.flush()
        for hook in self.hooks:
            hook.remove()
        if getattr(self.splats, "fused", None) is self:
            self.splats.fused = None
        self.hooks, self.state = [], None

    def _step(self):
        """One more Adam step. table[s] holds step s's bias-corrected step size of shN and
        1 / sqrt(1 - b2^s), for the steps taken and, at the current learning rate, ahead."""
        lr = self._group("shN")["lr"]
        if not math.isclose(
            self._group("sh0")["lr"], self.lr_sh0_scale * lr, rel_tol=1e-6
        ):
            raise RuntimeError(
                "DelayedSH needs the learning rates of sh0 and shN in a constant ratio."
            )
        if lr != self.lr or self.t + 2 > len(self.table):
            s = torch.arange(
                self.t + 1, max(2 * self.t + 2, 1 << 15), dtype=torch.float64
            )
            ahead = torch.stack(
                [lr / (1 - self.b1**s), (1 - self.b2**s) ** -0.5], -1
            )
            ahead = ahead.float().to(self.table.device)
            self.table, self.lr = torch.cat([self.table[: self.t + 1], ahead]), lr
        self.t += 1
        self.flushed = False

    def _moments(self, names):
        states = [self.optimizers[k].state[self.splats[k]] for k in names]
        return [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states]

    def _sh(self, degree=0, means=None, camera=None, radii=None):
        return _C.fused_sh(
            degree,
            means,
            camera,
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
            torch.cuda.current_stream(self.table.device).wait_stream(self.side_stream)
            self._sh()
            for name in self.names:
                self.optimizers[name].state[self.splats[name]]["step"].fill_(self.t)
            self.flushed = True

    def restructure(self):
        """Before the Gaussians are added, removed or moved: their delayed state is per index."""
        self.flush()
        self.state = None

    def _new_gaussians(self):
        """The delayed SH state of the Gaussians after a refinement: all are up to date."""
        n, device = len(self.splats["sh0"]), self.table.device
        state = torch.full((n,), self.t << 6, dtype=torch.int32, device=device)
        self.state, self.pend = state, torch.empty(n, 6, device=device)

    def coefficients(self):
        """All the coefficients, [N, K, 3], up to date: what rasterization() takes as SH."""
        self.flush()
        return torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

    def rasterize_view(
        self,
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        sh_degree,
        near_plane,
        far_plane,
        radius_clip,
        eps2d,
        tile_size,
        backgrounds,
        render_mode,
        absgrad,
        rasterize_mode,
        camera_model,
        plain,
    ):
        """What rasterization() returns for one view with these coefficients: the stages
        it runs, with the colors on a stream of their own alongside the tile intersection.
        None where that does not apply and nothing is trained: coefficients() then.
        `plain`: none of rasterization()'s other features is asked for."""
        sh0, shN = self.splats["sh0"], self.splats["shN"]
        if not self.hooks:
            raise RuntimeError("This DelayedSH is closed.")
        if not (
            plain
            and means.dim() == 2
            and viewmats.shape[:-2] == (1,)
            and sh_degree is not None
            and render_mode in ("RGB", "RGB+D", "RGB+ED")
            and camera_model in ("pinhole", "ortho", "fisheye")
        ):
            if torch.is_grad_enabled() and sh0.requires_grad:
                raise ValueError(
                    "DelayedSH trains with rasterization() of one view, with packed=False, "
                    "an SH degree, colors and a pinhole, ortho or fisheye camera, and with "
                    "none of its other features."
                )
            return None
        if len(means) != len(sh0) or (sh_degree + 1) ** 2 > shN.shape[1] + 1:
            raise ValueError(
                "DelayedSH: the means or the SH degree do not fit sh0 and shN."
            )
        if sh0.requires_grad != shN.requires_grad:
            raise ValueError("DelayedSH: sh0 and shN train together or not at all.")
        if self.state is None:
            DelayedSH._new_gaussians(
                self
            )  # nothing moves once the caller has read them
        elif len(self.state) != len(sh0):
            raise RuntimeError(
                "DelayedSH: restructure() comes before the Gaussians change."
            )
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations=rasterize_mode == "antialiased",
            camera_model=camera_model,
            opacities=opacities,
        )
        opacities = (
            opacities[None] if compensations is None else opacities * compensations
        )
        # what the colors read on their stream must live until this one has waited for it
        points = means.detach().contiguous()
        camera = torch.linalg.inv_ex(viewmats.detach())[0][0, :3].contiguous()
        colors = _Colors.apply(sh0, shN, self, sh_degree, points, camera, radii)[None]
        tile_width, tile_height = math.ceil(width / tile_size), math.ceil(
            height / tile_size
        )
        try:
            tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
                means2d,
                radii,
                depths,
                tile_size,
                tile_width,
                tile_height,
                conics=conics,
                opacities=opacities,
            )
            isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
            isect_offsets = isect_offsets.reshape(1, tile_height, tile_width)
        finally:  # the colors, also when this render ends here
            torch.cuda.current_stream(means.device).wait_stream(self.side_stream)
            del points, camera
        if render_mode != "RGB":  # a fourth channel, as rasterization() adds it
            colors = torch.cat([colors, depths[..., None]], -1)
            if backgrounds is not None:
                backgrounds = F.pad(backgrounds, (0, 1))
        render, alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            absgrad=absgrad,
        )
        if render_mode == "RGB+ED":
            depth = render[..., 3:] / alphas.clamp(min=1e-10)
            render = torch.cat([render[..., :3], depth], -1)
        meta = dict(
            batch_ids=None,
            camera_ids=None,
            gaussian_ids=None,
            radii=radii,
            means2d=means2d,
            depths=depths,
            conics=conics,
            opacities=opacities,
            tile_width=tile_width,
            tile_height=tile_height,
            tiles_per_gauss=tiles_per_gauss,
            isect_ids=isect_ids,
            flatten_ids=flatten_ids,
            isect_offsets=isect_offsets,
            width=width,
            height=height,
            tile_size=tile_size,
            n_batches=1,
            n_cameras=1,
        )
        return render, alphas, meta


class FusedTraining(DelayedSH):
    def __init__(self, splats, optimizers: dict, cfg, scene=None):
        self.names = tuple(splats.keys())
        super().__init__(splats, optimizers)
        self.scene = scene
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
        self.no_v_alpha = None

    @staticmethod
    def supports_delayed_sh(cfg, device) -> bool:
        """Whether a simple_trainer.py of this configuration can train with a DelayedSH."""
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

    @staticmethod
    def supports(cfg, device) -> bool:
        """Whether FusedTraining covers the configuration of a simple_trainer.py."""
        return (
            FusedTraining.supports_delayed_sh(cfg, device) and not cfg.post_processing
        )

    @torch.no_grad()
    def _new_gaussians(self):
        """Under MCMC, the Gaussians first go in Morton order."""
        if isinstance(self.strategy, MCMCStrategy):
            morton_order(self.splats, self.optimizers, self.scene)
        super()._new_gaussians()

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
        self._step()
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

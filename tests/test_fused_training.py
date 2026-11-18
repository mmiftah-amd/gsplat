# SPDX-License-Identifier: Apache-2.0
"""FusedTraining against rasterization(), torch.optim.Adam and the stock losses."""

from types import SimpleNamespace

import pytest
import torch

from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.losses import l1_loss, ssim_loss
from gsplat.optimizers.fused_training import FusedTraining
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import duplicate, inject_noise_to_position, remove, split

if not (
    torch.version.hip
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).warp_size == 32
):
    pytest.skip("needs a ROCm GPU with 32-lane wavefronts", allow_module_level=True)

DEV, W, H = "cuda", 96, 80
LRS = {
    "means": 1e-3,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh0": 2.5e-3,
    "shN": 2.5e-3 / 20,
}
K = torch.tensor([[[90.0, 0, W / 2], [0, 90.0, H / 2], [0, 0, 1.0]]], device=DEV)


def make(n=20000, sh_degree=3):
    torch.manual_seed(0)
    mid = torch.tensor([0.0, 0.0, 2.0], device=DEV)
    splats = {
        "means": torch.randn(n, 3, device=DEV) * 0.5 + mid,
        "scales": torch.log(torch.rand(n, 3, device=DEV) * 0.03 + 0.005),
        "quats": torch.randn(n, 4, device=DEV),
        "opacities": torch.randn(n, device=DEV),
        "sh0": torch.randn(n, 1, 3, device=DEV) * 0.5,
        "shN": torch.randn(n, (sh_degree + 1) ** 2 - 1, 3, device=DEV) * 0.05,
    }
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v) for k, v in splats.items()}
    )
    optimizers = {
        k: torch.optim.Adam(
            [{"params": p, "lr": LRS[k]}], eps=1e-15, betas=(0.9, 0.999)
        )
        for k, p in splats.items()
    }
    return splats, optimizers


def view(i):
    v = torch.eye(4, device=DEV)
    v[0, 3], v[1, 3] = 0.4 * (i - 1.5), 0.2 * (i % 2)
    return v[None]


def stock_render(s, viewmats, sh_degree=3):
    return rasterization(
        s["means"],
        s["quats"],
        s["scales"].exp(),
        s["opacities"].sigmoid(),
        torch.cat([s["sh0"], s["shN"]], 1),
        viewmats,
        K,
        W,
        H,
        sh_degree=sh_degree,
        packed=False,
    )


def fused_training(
    splats, optimizers, strategy=None, opacity_reg=0.0, scale_reg=0.0, **options
):
    cfg = SimpleNamespace(
        strategy=strategy or DefaultStrategy(),
        opacity_reg=opacity_reg,
        scale_reg=scale_reg,
        max_steps=100,
        **options,
    )
    return FusedTraining(splats, optimizers, cfg)


def train(a, oa, fused, target, views, degree, **options):
    """Steps of torch.optim.Adam on `a` with rasterization(), and of `fused`."""
    for i in views:
        # the one difference from stock: no position gradient through the SH direction
        coeffs = torch.cat([a["sh0"], a["shN"]], 1)
        colors = spherical_harmonics(degree, a["means"].detach(), view(i), coeffs)
        image, _, _ = rasterization(
            a["means"],
            a["quats"],
            a["scales"].exp(),
            a["opacities"].sigmoid(),
            torch.clamp_min(colors + 0.5, 0.0),
            view(i),
            K,
            W,
            H,
            sh_degree=None,
            packed=False,
            **options,
        )
        loss = (image - target).abs().mean()
        loss = loss + 0.01 * a["opacities"].sigmoid().mean()
        loss = loss + 0.01 * a["scales"].exp().mean()
        loss.backward()
        for o in oa.values():
            o.step()
            o.zero_grad(set_to_none=True)
        image, _, _ = fused.rasterize(
            torch.linalg.inv(view(i)),
            K,
            W,
            H,
            sh_degree=degree,
            render_mode=options.get("render_mode", "RGB"),
        )
        (image - target).abs().mean().backward()
        fused.step(0, 1.0)


def assert_trained_alike(a, oa, b, ob):
    for k in a:
        assert_same(b[k], a[k], k)
        state_a, state_b = oa[k].state[a[k]], ob[k].state[b[k]]
        assert float(state_b["step"]) == float(state_a["step"]), k
        for key in ("exp_avg", "exp_avg_sq"):
            assert_same(state_b[key], state_a[key], (k, key))


def assert_same(x, y, name):
    """Adam normalizes the gradient, so it turns the rounding of the few near-zero
    gradients into differences of a learning rate: compare a fraction."""
    assert x.shape == y.shape, name
    if x.numel():  # shN is empty at SH degree 0
        close = (x - y).abs() <= 1e-3 * (y.abs().max() + y.abs())
        assert close.float().mean() > 1 - 1e-3, name


@pytest.mark.parametrize("op", ["none", "remove", "duplicate", "split"])
@pytest.mark.parametrize("sh_degree", [3, 1, 0])
def test_fused_steps_are_adam_steps(op, sh_degree):
    """Also with Gaussians the views miss (their SH steps are delayed), through the
    SH degree schedule, and across the strategies' operations, which call
    restructure() themselves."""
    a, oa = make(sh_degree=sh_degree)
    b, ob = make(sh_degree=sh_degree)
    fused = fused_training(b, ob, opacity_reg=0.01, scale_reg=0.01)
    target = torch.rand(1, H, W, 3, device=DEV)
    train(a, oa, fused, target, (0, 1), 0)
    train(a, oa, fused, target, (2, 3, 0), sh_degree)
    if op != "none":
        for params, opts in ((a, oa), (b, ob)):
            mask = torch.zeros(len(params["means"]), dtype=torch.bool, device=DEV)
            mask[::7] = True
            torch.manual_seed(1)
            {"remove": remove, "duplicate": duplicate, "split": split}[op](
                params, opts, {}, mask
            )
        assert fused.flushed and fused.state is None
    train(a, oa, fused, target, (1, 2), sh_degree)
    fused.flush()
    assert_trained_alike(a, oa, b, ob)


@pytest.mark.parametrize(
    "camera_model,antialiased,render_mode",
    [
        ("pinhole", True, "RGB"),
        ("ortho", False, "RGB"),
        ("fisheye", False, "RGB"),
        ("fisheye", True, "RGB+ED"),
        ("pinhole", False, "RGB+D"),
        ("pinhole", False, "RGB+ED"),
    ],
)
def test_fused_steps_with_render_options(camera_model, antialiased, render_mode):
    """The camera models, antialiasing and the depth channel."""
    a, oa = make()
    b, ob = make()
    fused = fused_training(
        b,
        ob,
        opacity_reg=0.01,
        scale_reg=0.01,
        camera_model=camera_model,
        antialiased=antialiased,
    )
    target = torch.rand(1, H, W, 3 if render_mode == "RGB" else 4, device=DEV)
    options = dict(
        camera_model=camera_model,
        rasterize_mode="antialiased" if antialiased else "classic",
        render_mode=render_mode,
    )
    train(a, oa, fused, target, (0, 1, 2, 3), 3, **options)
    fused.flush()
    assert_trained_alike(a, oa, b, ob)


@pytest.mark.parametrize("absgrad", [False, True])
def test_render_matches_rasterization(absgrad):
    """The image, the alpha and what the default strategy reads from the render."""
    a, _ = make()
    b, ob = make()
    fused = fused_training(b, ob, DefaultStrategy(absgrad=absgrad))
    image_a, alpha_a, info_a = rasterization(
        a["means"],
        a["quats"],
        a["scales"].exp(),
        a["opacities"].sigmoid(),
        torch.cat([a["sh0"], a["shN"]], 1),
        view(1),
        K,
        W,
        H,
        sh_degree=3,
        packed=False,
        absgrad=absgrad,
    )
    image_b, alpha_b, info_b = fused.rasterize(torch.linalg.inv(view(1)), K, W, H)
    assert torch.allclose(image_a, image_b, atol=2e-5)
    assert torch.allclose(alpha_a, alpha_b, atol=2e-5)
    assert torch.equal(info_a["radii"], info_b["radii"])
    weights = torch.rand(1, H, W, 3, device=DEV)
    for image, info in ((image_a, info_a), (image_b, info_b)):
        info["means2d"].retain_grad()
        (image * weights).sum().backward()
    key = "absgrad" if absgrad else "grad"
    ref = getattr(info_a["means2d"], key)
    assert torch.allclose(
        getattr(info_b["means2d"], key), ref, rtol=2e-3, atol=1e-3 * ref.abs().max()
    )


def test_mcmc_refinement_keeps_the_gaussians():
    """A step that refines leaves the noise to the strategy, whose operations flush;
    the next render puts the Gaussians in Morton order with their moments and
    renders the same image."""
    splats, optimizers = make()
    strategy = MCMCStrategy(refine_start_iter=0, refine_every=2)
    state = strategy.initialize_state()
    fused = fused_training(splats, optimizers, strategy)
    c2w = torch.linalg.inv(view(0))
    for step in (1, 2):
        image, _, info = fused.rasterize(c2w, K, W, H)
        image.mean().backward()
        fused.step(step, 1.0)
        assert splats.noise_added == (step == 1) and not fused.flushed
        before = splats["means"].detach().clone()
        strategy.step_post_backward(splats, optimizers, state, step, info, lr=1e-3)
        assert not splats.noise_added  # the strategy's call used it up
        assert (fused.state is None) == (step == 2)
        assert torch.equal(before, splats["means"]) == (step == 1)
    before = stock_render(splats, view(0))[0]
    fused.rasterize(c2w, K, W, H)
    assert fused.state is not None
    assert torch.allclose(before, stock_render(splats, view(0))[0], atol=1e-4)
    moments = optimizers["means"].state[splats["means"]]
    assert moments["exp_avg"].shape == splats["means"].shape


@pytest.mark.parametrize("width,height", [(W, H), (101, 83), (33, 17), (16, 48)])
def test_loss_matches_stock(width, height):
    pytest.importorskip("fused_ssim")  # ssim_loss() pads without it
    splats, optimizers = make()
    fused = fused_training(splats, optimizers)
    target = torch.rand(1, height, width, 3, device=DEV)
    results = []
    for use_fused in (False, True):
        generator = torch.Generator(DEV).manual_seed(3)
        image = torch.rand(1, height, width, 3, device=DEV, generator=generator)
        image.requires_grad_()
        if use_fused:
            loss, l1, ssim = fused.loss(image, target, 0.2)
        else:
            l1 = l1_loss(image, target).mean()
            ssim = ssim_loss(image.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
            loss = torch.lerp(l1, ssim, 0.2)
        loss.backward()
        results.append((loss.detach(), l1.detach(), ssim.detach(), image.grad))
    for x, y in zip(*results):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-7)


def test_mcmc_noise_distribution():
    """The noise of the geometry step against inject_noise_to_position."""
    n, stds = 200000, []
    for fused_noise in (True, False):
        splats, optimizers = make(n)
        with torch.no_grad():
            splats["opacities"].fill_(-8.0)
            splats["scales"].fill_(-3.0)
            splats["quats"].copy_(torch.tensor([1.0, 0, 0, 0], device=DEV).expand(n, 4))
            splats["means"].fill_(100.0)  # out of view, and any order is the same
        if fused_noise:
            fused = fused_training(splats, optimizers, MCMCStrategy(noise_lr=1e6))
            optimizers["means"].param_groups[0]["lr"] = 1e-6  # the noise's scale
            image, _, _ = fused.rasterize(torch.eye(4, device=DEV)[None], K, W, H)
            image.sum().backward()
            fused.step(0, 1.0)
        else:
            with torch.no_grad():
                inject_noise_to_position(splats, optimizers, {}, 1.0)
        delta = splats["means"].detach() - 100.0
        stds.append(delta.std(0))
        assert (delta.mean(0).abs() < 0.01 * stds[-1]).all()
    assert torch.allclose(stds[0], stds[1], rtol=0.01)

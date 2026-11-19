# SPDX-License-Identifier: Apache-2.0
"""FusedTraining and DelayedSH against rasterization(), torch.optim.Adam and the stock
losses."""

import copy
from types import SimpleNamespace

import pytest
import torch

from gsplat.cuda._backend import _C
from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.losses import l1_loss, l1_ssim_loss, ssim_loss
from gsplat.optimizers.fused_training import DelayedSH, FusedTraining, morton_order
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


def adam_step(a, oa, target, i, degree, colors=None, **options):
    """A step of torch.optim.Adam on `a` with rasterization(), from view i."""
    if colors is None:
        # the one difference from stock: no position gradient through the SH direction
        coeffs = torch.cat([a["sh0"], a["shN"]], 1)
        colors = spherical_harmonics(degree, a["means"].detach(), view(i), coeffs)
        colors, degree = torch.clamp_min(colors + 0.5, 0.0), None
    image, _, _ = rasterization(
        a["means"],
        a["quats"],
        a["scales"].exp(),
        a["opacities"].sigmoid(),
        colors,
        view(i),
        K,
        W,
        H,
        sh_degree=degree,
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


def train(a, oa, fused, target, views, degree, **options):
    """Steps of torch.optim.Adam on `a` with rasterization(), and of `fused`."""
    for i in views:
        adam_step(a, oa, target, i, degree, **options)
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
        close = (x - y).abs() <= 2e-4 * (y.abs().max() + y.abs())
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


@pytest.mark.parametrize("op", ["none", "duplicate"])
@pytest.mark.parametrize("sh_degree", [3, 0])
@pytest.mark.parametrize("antialiased", [False, True])
def test_delayed_sh_in_a_plain_loop(op, sh_degree, antialiased):
    """A loop of rasterization() and torch.optim.Adam with a DelayedSH for the SH
    coefficients, against the same loop without it: with Gaussians the views miss,
    through the SH degree schedule, a change of the SH learning rates and a strategy's
    operation."""
    a, oa = make(sh_degree=sh_degree)
    b, ob = make(sh_degree=sh_degree)
    sh = DelayedSH(b, ob)
    target = torch.rand(1, H, W, 3, device=DEV)
    options = dict(rasterize_mode="antialiased" if antialiased else "classic")
    rounds = (((0, 1), 0), ((2, 3, 0), sh_degree), ((1, 2), sh_degree))
    for n, (views, degree) in enumerate(rounds):
        for i in views:
            adam_step(a, oa, target, i, degree, **options)
            adam_step(b, ob, target, i, degree, colors=sh, **options)
        for params, opts in ((a, oa), (b, ob)) if n < 2 else ():
            for name in ("sh0", "shN"):
                opts[name].param_groups[0]["lr"] *= 0.5
            if op != "none" and n == 1:
                mask = torch.zeros(len(params["means"]), dtype=torch.bool, device=DEV)
                mask[::7] = True
                duplicate(params, opts, {}, mask)
        assert (sh.state is None) == (op != "none" and n == 1)
    assert not sh.flushed
    sh.flush()
    assert_trained_alike(a, oa, b, ob)


@pytest.mark.parametrize(
    "camera_model,render_mode,absgrad,background",
    [
        ("pinhole", "RGB", True, True),
        ("ortho", "RGB+D", False, False),
        ("fisheye", "RGB+ED", False, False),
    ],
)
def test_delayed_sh_with_render_options(camera_model, render_mode, absgrad, background):
    """The camera models, the depth channel, absgrad and a background: the steps, and
    what rasterization() returns besides the image."""
    a, oa = make()
    b, ob = make()
    sh = DelayedSH(b, ob)
    target = torch.rand(1, H, W, 3 if render_mode == "RGB" else 4, device=DEV)
    options = dict(camera_model=camera_model, render_mode=render_mode, absgrad=absgrad)
    if background:
        options["backgrounds"] = torch.rand(1, 3, device=DEV)
    for i in (0, 1, 2, 3):
        adam_step(a, oa, target, i, 3, **options)
        adam_step(b, ob, target, i, 3, colors=sh, **options)
    sh.flush()
    assert_trained_alike(a, oa, b, ob)
    results = []
    for colors in (torch.cat([b["sh0"], b["shN"]], 1), sh):
        args = b["means"], b["quats"], b["scales"].exp(), b["opacities"].sigmoid()
        image, alpha, meta = rasterization(
            *args, colors, view(1), K, W, H, sh_degree=3, packed=False, **options
        )
        meta["means2d"].retain_grad()
        (image * target).sum().backward()
        results.append((image, alpha, meta))
    (image_a, alpha_a, meta_a), (image_b, alpha_b, meta_b) = results
    assert torch.allclose(image_a, image_b, atol=1e-4)
    assert torch.allclose(alpha_a, alpha_b, atol=1e-4)
    assert meta_a.keys() == meta_b.keys()
    seen = (meta_a["radii"] > 0).all(-1)  # the projection leaves the others unwritten
    for key, x in meta_a.items():
        y = meta_b[key]
        if not isinstance(x, torch.Tensor):
            assert x == y, key
        elif x.is_floating_point():
            if x.shape[:2] == seen.shape:
                x, y = x[seen], y[seen]
            assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), key
        else:
            assert torch.equal(x, y), key
    for key in ("grad", "absgrad") if absgrad else ("grad",):
        x, y = getattr(meta_a["means2d"], key), getattr(meta_b["means2d"], key)
        assert torch.allclose(x, y, rtol=2e-3, atol=1e-3 * x.abs().max()), key


def delayed_render(s, sh, i=0, sh_degree=3, **options):
    args = s["means"], s["quats"], s["scales"].exp(), s["opacities"].sigmoid()
    return rasterization(
        *args, sh, view(i), K, W, H, sh_degree=sh_degree, packed=False, **options
    )


def test_delayed_sh_with_mcmc_refinement():
    """Through a strategy's refinements and morton_order() after them, every step
    reaches the optimizers and the render is that of the coefficients."""
    splats, optimizers = make()
    strategy = MCMCStrategy(refine_start_iter=0, refine_every=2)
    state = strategy.initialize_state()
    sh = DelayedSH(splats, optimizers)
    for step in range(1, 6):
        image, _, info = delayed_render(splats, sh)
        image.mean().backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strategy.step_post_backward(splats, optimizers, state, step, info, lr=1e-3)
        assert (sh.state is None) == (step % 2 == 0)
        if sh.state is None:
            morton_order(splats, optimizers)
        with torch.no_grad():
            image = delayed_render(splats, sh)[0]
            assert torch.allclose(image, stock_render(splats, view(0))[0], atol=1e-4)
        assert float(optimizers["means"].state[splats["means"]]["step"]) == step
    sh.flush()
    assert float(optimizers["shN"].state[splats["shN"]]["step"]) == 5


@pytest.mark.parametrize("frozen", ["geometry", "sh"])
def test_delayed_sh_with_frozen_parameters(frozen):
    """What trains is what requires a gradient: the coefficients alone, or not at all."""
    a, oa = make()
    b, ob = make()
    names = ("sh0", "shN") if frozen == "sh" else ("means", "quats", "scales")
    for s in (a, b):
        for name in names + (() if frozen == "sh" else ("opacities",)):
            s[name].requires_grad_(False)
    sh = DelayedSH(b, ob)
    target = torch.rand(1, H, W, 3, device=DEV)
    before = b["sh0"].detach().clone()
    for i in (0, 1, 2):
        for s, opts, colors in (
            (a, oa, torch.cat([a["sh0"], a["shN"]], 1)),
            (b, ob, sh),
        ):
            image = delayed_render(s, colors, i)[0]
            assert image.requires_grad
            (image - target).abs().mean().backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
    sh.flush()
    assert torch.equal(before, b["sh0"]) == (frozen == "sh")
    for k in ("sh0", "shN") if frozen == "geometry" else ():
        # no view-direction term is left to differ in
        assert_same(b[k], a[k], k)
        assert_same(ob[k].state[b[k]]["exp_avg"], oa[k].state[a[k]]["exp_avg"], k)
        assert float(ob[k].state[b[k]]["step"]) == 3


def test_delayed_sh_refuses_what_it_cannot_do():
    """Uses that would give other steps than Adam's raise."""
    target = torch.rand(1, H, W, 3, device=DEV)

    def loop(change=None, loss=None, stepped=None, lr_before_step=1.0):
        splats, optimizers = make()
        sh = DelayedSH(splats, optimizers)
        for step in range(3):
            image = delayed_render(splats, sh, step)[0]
            if change is not None and step:
                change(splats, optimizers, sh, image)
            term = (
                (image - target).abs().mean() if loss is None else loss(splats, image)
            )
            term.backward()
            for name, o in optimizers.items():
                if step and name in ("sh0", "shN"):
                    o.param_groups[0]["lr"] *= lr_before_step
                if stepped is None or step == 0 or name in stepped:
                    o.step()
                o.zero_grad(set_to_none=True)
        return splats, sh

    def second_render(splats, optimizers, sh, image):
        (delayed_render(splats, sh, 2)[0].mean() + image.mean()).backward()

    def render_before_backward(splats, optimizers, sh, image):
        with torch.no_grad():
            delayed_render(splats, sh, 2)

    def no_optimizer_step(splats, optimizers, sh, image):
        delayed_render(splats, sh, 2)[0].mean().backward()
        delayed_render(splats, sh, 3)[0].mean().backward()

    def other_ratio(splats, optimizers, sh, image):
        optimizers["sh0"].param_groups[0]["lr"] *= 2

    def pruned(splats, optimizers, sh, image):
        for name in splats.keys():
            splats[name] = torch.nn.Parameter(splats[name][:-1].detach())
        delayed_render(splats, sh)

    splats, sh = loop()
    sh.close()
    with pytest.raises(RuntimeError, match="closed"):
        delayed_render(splats, sh)
    changes = second_render, render_before_backward, no_optimizer_step, other_ratio
    for change in (*changes, pruned):
        with pytest.raises(RuntimeError, match="DelayedSH"):
            loop(change)
    with pytest.raises(RuntimeError, match="DelayedSH"):  # a regularizer of shN
        loop(loss=lambda s, image: image.mean() + s["shN"].abs().mean())
    with pytest.raises(RuntimeError, match="DelayedSH"):  # only the geometry steps
        loop(stepped=("means", "quats", "scales", "opacities"))
    with pytest.raises(RuntimeError, match="DelayedSH"):  # after backward()
        loop(lr_before_step=0.5)
    splats, optimizers = make()
    groups = [{"params": [p], "lr": LRS[k]} for k, p in splats.items()]
    shared = torch.optim.Adam(groups, eps=1e-15)
    sh = DelayedSH(splats, {k: shared for k in splats.keys()})

    def closure():  # the gradient of a regularizer appears inside the step
        shared.zero_grad(set_to_none=True)
        loss = delayed_render(splats, sh)[0].mean() + splats["shN"].abs().mean()
        loss.backward()
        return loss

    with pytest.raises(RuntimeError, match="DelayedSH"):
        shared.step(closure)
    splats, optimizers = make()
    with pytest.raises(ValueError):  # more than the coefficients' SH degree
        delayed_render(splats, DelayedSH(splats, optimizers), sh_degree=4)
    wide = torch.nn.Parameter(torch.zeros(len(splats["shN"]), 24, 3, device=DEV))
    with pytest.raises(ValueError):
        DelayedSH(torch.nn.ParameterDict(dict(splats, shN=wide)), optimizers)
    optimizers["shN"] = torch.optim.Adam([splats["shN"]], weight_decay=0.1)
    with pytest.raises(ValueError):
        DelayedSH(splats, optimizers)
    splats, optimizers = make()
    with pytest.raises(TypeError):  # nowhere for the strategies to find it
        DelayedSH(dict(splats), optimizers)
    with pytest.raises(ValueError):  # the optimizers of other splats
        DelayedSH(make()[0], optimizers)


def test_delayed_sh_closes():
    """After close(), the optimizers go on with sh0 and shN as torch.optim.Adam's."""
    a, oa = make()
    b, ob = make()
    sh = DelayedSH(b, ob)
    target = torch.rand(1, H, W, 3, device=DEV)
    for i in (0, 3):
        adam_step(a, oa, target, i, 3)
        adam_step(b, ob, target, i, 3, colors=sh)
    sh.close()
    assert b.fused is None and not ob["sh0"]._optimizer_step_pre_hooks
    for i in (1, 2):
        for s, opts in ((a, oa), (b, ob)):
            coefficients = torch.cat([s["sh0"], s["shN"]], 1)
            adam_step(s, opts, target, i, 3, colors=coefficients)
    assert_trained_alike(a, oa, b, ob)


def test_kernels_check_their_inputs():
    """The entry points of the kernels refuse tensors that they would read past."""
    image = torch.rand(1, H, W, 3, device=DEV)
    dm, out = image.new_empty(9, H, W), image.new_empty(H, W, 2)
    _C.l1_ssim(image, image, dm, None, out, 1.0, 1.0)
    for bad in (
        (image, image[..., :1].contiguous(), dm, None, out),
        (image, image.cpu(), dm, None, out),
        (image, image, dm[:3].contiguous(), None, out),
    ):
        with pytest.raises(RuntimeError):
            _C.l1_ssim(*bad, 1.0, 1.0)
    n = 100
    state = torch.zeros(n, dtype=torch.int32, device=DEV)
    radii = torch.ones(1, n, 2, dtype=torch.int32, device=DEV)
    v_colors, pend = torch.zeros(n, 3, device=DEV), torch.zeros(n, 6, device=DEV)
    _C.fused_sh_grad(v_colors, radii, 3, state, pend)
    for bad in (
        (v_colors[:-1], radii, 3, state, pend),
        (v_colors, radii, 3, state, pend[::2]),
    ):
        with pytest.raises(RuntimeError):
            _C.fused_sh_grad(*bad)


def test_delayed_sh_copies_and_layouts():
    """A copy of the splats needs them up to date and is on its own, a second DelayedSH
    takes over from the first, and inputs need not be contiguous."""
    a, oa = make()
    b, ob = make()
    sh = DelayedSH(b, ob)
    target = torch.rand(1, H, W, 3, device=DEV)
    for i in (0, 3, 1):
        adam_step(a, oa, target, i, 3)
        adam_step(b, ob, target, i, 3, colors=sh)
    with pytest.raises(RuntimeError, match="flush"):
        copy.deepcopy(b)
    sh.flush()
    clone = copy.deepcopy(b)
    assert clone.fused is None and torch.equal(clone["shN"], b["shN"])
    assert_same(clone["shN"], a["shN"], "copy")
    adam_step(a, oa, target, 2, 3)
    adam_step(b, ob, target, 2, 3, colors=sh)
    sh = DelayedSH(b, ob)
    for i in (0, 2):
        adam_step(a, oa, target, i, 3)
        adam_step(b, ob, target, i, 3, colors=sh)
    sh.flush()
    assert_trained_alike(a, oa, b, ob)
    wide = torch.cat([b["means"].detach(), torch.zeros_like(b["means"])], 1)
    with torch.no_grad():
        image = rasterization(
            wide[:, :3],
            b["quats"],
            b["scales"].exp(),
            b["opacities"].sigmoid(),
            sh,
            view(1),
            K,
            W,
            H,
            sh_degree=3,
            packed=False,
        )[0]
        assert torch.allclose(image, delayed_render(b, sh, 1)[0], atol=1e-6)


def test_delayed_sh_outside_its_case():
    """Without gradients a DelayedSH renders any case, with the stock evaluation where
    its own does not apply; it does not train there."""
    a, oa = make()
    b, ob = make()
    sh = DelayedSH(b, ob)
    target = torch.rand(1, H, W, 3, device=DEV)
    for i in (0, 3):
        adam_step(a, oa, target, i, 3)
        adam_step(b, ob, target, i, 3, colors=sh)
    for views in (view(1), torch.cat([view(1), view(2)])):
        renders = []
        for s, colors in ((a, torch.cat([a["sh0"], a["shN"]], 1)), (b, sh)):
            args = s["means"], s["quats"], s["scales"].exp(), s["opacities"].sigmoid()
            Ks = K.expand(len(views), 3, 3)
            with torch.no_grad():
                renders.append(
                    rasterization(*args, colors, views, Ks, W, H, sh_degree=3)[0]
                )
        assert torch.allclose(*renders, atol=1e-4)
    with pytest.raises(ValueError):
        rasterization(*args, sh, views, Ks, W, H, sh_degree=3)


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
@pytest.mark.parametrize("slice_of_rgba", [False, True])
def test_loss_matches_stock(width, height, slice_of_rgba):
    """Also for a target that is not contiguous."""
    pytest.importorskip("fused_ssim")  # ssim_loss() pads without it
    target = torch.rand(1, height, width, 4, device=DEV)[..., :3]
    target = target if slice_of_rgba else target.contiguous()
    results = []
    for use_fused in (False, True):
        generator = torch.Generator(DEV).manual_seed(3)
        image = torch.rand(1, height, width, 3, device=DEV, generator=generator)
        image.requires_grad_()
        if use_fused:
            loss, l1, ssim = l1_ssim_loss(image, target, 0.2)
        else:
            l1 = l1_loss(image, target).mean()
            ssim = ssim_loss(image.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
            loss = torch.lerp(l1, ssim, 0.2)
        assert not (use_fused and (l1.requires_grad or ssim.requires_grad))
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

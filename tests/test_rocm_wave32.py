# SPDX-License-Identifier: Apache-2.0
"""The rasterizer kernels of 32-lane ROCm GPUs against the generic kernels, in one process.

The 32-lane kernels stop at 32 channels, so the same scene with its colors padded to 33
channels takes the generic kernels: its first channels and its geometry gradients are the
reference. A copy of flatten_ids drops what intersect_tile leaves behind it, which selects
the generic backward for the 32-lane forward."""

import math

import pytest
import torch

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)

if not (
    torch.version.hip
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).warp_size == 32
):
    pytest.skip("needs a ROCm GPU with 32-lane wavefronts", allow_module_level=True)

DEV = "cuda"


def _project(n, batch, cams, width, height, packed):
    means = torch.randn(*batch, n, 3, device=DEV) * 0.6
    means[..., 2] += 3.0
    quats = torch.randn(*batch, n, 4, device=DEV)
    scales = torch.rand(*batch, n, 3, device=DEV) * 0.08 + 0.004
    scales[..., : n // 50, :] *= 12  # some that cover many tiles
    opacities = torch.rand(*batch, n, device=DEV)
    viewmats = torch.eye(4, device=DEV).repeat(*batch, cams, 1, 1)
    viewmats[..., :, 0, 3] = torch.linspace(-0.3, 0.3, cams, device=DEV)
    Ks = torch.tensor(
        [[300.0, 0, width / 2], [0, 300.0, height / 2], [0, 0, 1]], device=DEV
    ).repeat(*batch, cams, 1, 1)
    out = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        packed=packed,
        opacities=opacities,
    )
    image_dims = (*batch, cams)
    if packed:
        batch_ids, camera_ids, gaussian_ids, _, radii, means2d, depths, conics, _ = out
        opacities = opacities.reshape(-1, n)[batch_ids, gaussian_ids]
        ids = dict(image_ids=batch_ids * cams + camera_ids, gaussian_ids=gaussian_ids)
    else:
        radii, means2d, depths, conics, _ = out
        opacities = opacities.unsqueeze(-2).expand(*image_dims, n).contiguous()
        ids = dict(image_ids=None, gaussian_ids=None)
    return image_dims, radii, means2d, depths, conics, opacities, ids


def _render(colors, proj, width, height, packed, backgrounds, masks, absgrad, keep):
    image_dims, radii, means2d, depths, conics, opacities, ids = proj
    tile_width, tile_height = math.ceil(width / 16), math.ceil(height / 16)
    n_images = math.prod(image_dims)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        16,
        tile_width,
        tile_height,
        packed=packed,
        n_images=n_images,
        conics=conics,
        opacities=opacities,
        **ids,
    )
    if not keep:
        flatten_ids = flatten_ids.clone()
    offsets = isect_offset_encode(isect_ids, n_images, tile_width, tile_height)
    offsets = offsets.reshape(*image_dims, tile_height, tile_width)
    leaves = [x.detach().requires_grad_() for x in (means2d, conics, opacities, colors)]
    image, alpha = rasterize_to_pixels(
        leaves[0],
        leaves[1],
        leaves[3],
        leaves[2],
        width,
        height,
        16,
        offsets,
        flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        packed=packed,
        absgrad=absgrad,
    )
    return image, alpha, leaves


def _assert_close(x, ref, rtol, atol, what):
    wrong = (x - ref).abs() > atol + rtol * ref.abs()
    # Float sums in another order move a sample across the alpha cutoff now and then.
    assert wrong.float().mean() < 2e-4, what


@pytest.mark.parametrize("channels", [1, 3, 4, 32])
@pytest.mark.parametrize("batch,cams", [((), 1), ((), 2), ((2,), 2)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("size", [(176, 112), (101, 83)])
def test_wave32_rasterizer_matches_generic(channels, batch, cams, packed, extras, size):
    """`extras`: backgrounds, tile masks and absgrad."""
    torch.manual_seed(0)
    width, height = size
    proj = _project(3000, batch, cams, width, height, packed)
    image_dims, shape = proj[0], proj[2].shape[:-1]
    colors = torch.rand(*shape, channels, device=DEV)
    wide = torch.cat([colors, torch.rand(*shape, 33 - channels, device=DEV)], -1)
    tile_height, tile_width = math.ceil(height / 16), math.ceil(width / 16)
    backgrounds = torch.rand(*image_dims, 33, device=DEV) if extras else None
    masks = (
        torch.rand(*image_dims, tile_height, tile_width, device=DEV) > 0.2
        if extras
        else None
    )
    w_image = torch.rand(*image_dims, height, width, channels, device=DEV)
    w_alpha = torch.rand(*image_dims, height, width, 1, device=DEV)

    results = {}
    for kind in ("wave32", "wave32 forward", "generic"):
        generic = kind == "generic"
        bg = backgrounds
        if bg is not None and not generic:
            bg = bg[..., :channels].contiguous()
        image, alpha, leaves = _render(
            wide if generic else colors,
            proj,
            width,
            height,
            packed,
            bg,
            masks,
            extras,
            keep=kind == "wave32",
        )
        image = image[..., :channels]
        ((image * w_image).sum() + (alpha * w_alpha).sum()).backward()
        grads = [x.grad[..., :channels] if x is leaves[3] else x.grad for x in leaves]
        if extras:
            grads.append(leaves[0].absgrad)
        results[kind] = (image.detach(), alpha.detach(), grads)

    ref_image, ref_alpha, ref_grads = results.pop("generic")
    for kind, (image, alpha, grads) in results.items():
        _assert_close(image, ref_image, 1e-4, 2e-5, (kind, "image"))
        _assert_close(alpha, ref_alpha, 1e-4, 2e-5, (kind, "alpha"))
        for i, (grad, ref) in enumerate(zip(grads, ref_grads)):
            _assert_close(grad, ref, 2e-3, 1e-3 * ref.abs().max() + 1e-6, (kind, i))


def test_intersect_tile_keeps_its_outputs():
    """What intersect_tile leaves behind flatten_ids does not show in its results."""
    torch.manual_seed(0)
    _, radii, means2d, depths, conics, opacities, _ = _project(
        3000, (), 1, 176, 112, False
    )
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, 16, 11, 7, conics=conics, opacities=opacities
    )
    assert flatten_ids.shape == isect_ids.shape == (int(tiles_per_gauss.sum()),)
    assert (isect_ids[1:] >= isect_ids[:-1]).all()
    assert torch.equal(
        torch.bincount(flatten_ids, minlength=tiles_per_gauss.numel()),
        tiles_per_gauss.flatten(),
    )

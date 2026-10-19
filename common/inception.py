"""Helpers for the original InceptionV1 (inception5h / Lucid) checkpoint, via the
Lucent port, and for scanning a folder of images for a layer's activations.

Layer names follow Lucid: mixed3a, mixed3b, mixed4a ... mixed4e, mixed5a, mixed5b.
Channel c of layer L is written "L:c" in the essay (e.g. 4e:55 = mixed4e channel 55).
"""
from __future__ import annotations

import pathlib
from typing import Iterable

import numpy as np
import torch
from PIL import Image

IMAGENET_SIZE = 224


def load_inceptionv1(device: str = "cuda"):
    """The original InceptionV1 weights (Lucid's model), ported to PyTorch by Lucent.
    Expects inputs in [0, 1]; the model internally rescales to the Caffe range."""
    from lucent.modelzoo import inceptionv1
    model = inceptionv1(pretrained=True)
    model.to(device).eval()
    return model


def _preprocess(path: pathlib.Path, size: int = IMAGENET_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    # Lucent's InceptionV1 takes the Caffe-era input range: 255 * x - 117.
    arr = np.asarray(img, dtype=np.float32) - 117.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.no_grad()
def scan_layer(model, layer: str, paths: Iterable[pathlib.Path], device: str = "cuda",
               batch: int = 64):
    """Run images through the model and record, for every channel of `layer`,
    the spatial max and min of its activation map on each image, plus their
    locations. Hook a *_pre_relu_conv layer to see the negative pole.
    Returns dict(max [N,C], argmax_yx [N,C,2], min [N,C], argmin_yx [N,C,2], grid, paths)."""
    paths = list(paths)
    feats = {}

    def hook(_, __, out):
        feats["x"] = out

    module = dict(model.named_modules())[layer]
    h = module.register_forward_hook(hook)
    mx, mxl, mn, mnl = [], [], [], []
    grid = None
    try:
        for i in range(0, len(paths), batch):
            xb = torch.stack([_preprocess(p) for p in paths[i:i + batch]]).to(device)
            model(xb)
            fm = feats["x"]                       # [B, C, H, W]
            B, C, H, W = fm.shape
            grid = (H, W)
            flat = fm.reshape(B, C, H * W)
            v, idx = flat.max(-1)
            mx.append(v.cpu()); mxl.append(torch.stack([idx // W, idx % W], -1).cpu())
            v, idx = flat.min(-1)
            mn.append(v.cpu()); mnl.append(torch.stack([idx // W, idx % W], -1).cpu())
    finally:
        h.remove()
    return dict(max=torch.cat(mx).numpy(), argmax_yx=torch.cat(mxl).numpy(),
                min=torch.cat(mn).numpy(), argmin_yx=torch.cat(mnl).numpy(),
                grid=grid, paths=[str(p) for p in paths])


def receptive_crop(path: pathlib.Path, yx, grid: int, size: int = IMAGENET_SIZE,
                   frac: float = 0.45) -> Image.Image:
    """Crop a window around the argmax grid location, as a rough stand-in for
    the receptive field, from the same centre-crop the model saw."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    cy, cx = (yx[0] + 0.5) * size / grid, (yx[1] + 0.5) * size / grid
    half = frac * size / 2
    box = (int(max(0, cx - half)), int(max(0, cy - half)),
           int(min(size, cx + half)), int(min(size, cy + half)))
    return img.crop(box)

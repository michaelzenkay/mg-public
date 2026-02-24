"""Checkpoint save/load utilities for model training."""

import math
import torch
import torch.nn as nn
from pathlib import Path


def save_ckpt(path: Path, model, optimizer, epoch: int, best_val: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_dp = isinstance(model, torch.nn.DataParallel)
    state = model.module.state_dict() if is_dp else model.state_dict()
    torch.save({
        "epoch": int(epoch),
        "best_val": float(best_val),
        "state_dict": state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }, path.as_posix())


def load_ckpt(path: Path, model, optimizer=None, map_location="cpu", strict=True):
    def _fix_dp_prefix(sd, want_dp):
        has_mod = next(iter(sd)).startswith("module.")
        if want_dp and not has_mod:
            return {f"module.{k}": v for k, v in sd.items()}
        if not want_dp and has_mod:
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd

    ckpt = torch.load(path.as_posix(), map_location=map_location, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    is_dp = isinstance(model, torch.nn.DataParallel)
    sd = _fix_dp_prefix(sd, want_dp=is_dp)
    model.load_state_dict(sd, strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val = float(ckpt.get("best_val", math.inf))
    return start_epoch, best_val


def load_weights_only(path, model, map_location="cpu", strict=False):
    ckpt = torch.load(path, map_location=map_location)
    sd = ckpt.get("state_dict", ckpt)
    is_dp = isinstance(model, nn.DataParallel)
    has_module = next(iter(sd)).startswith("module.")
    if is_dp and not has_module:
        sd = {f"module.{k}": v for k, v in sd.items()}
    if not is_dp and has_module:
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[init_from] loaded weights; missing={len(missing)}, unexpected={len(unexpected)}")

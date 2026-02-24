"""Mammography risk prediction model - CLI entry point.

Usage:
    python mg.py --config configs/a4.yaml [--resume path/to/checkpoint]
"""

import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace

from pipelines import run_train_staged

# Backward-compatible re-export for eval_external.py and EVAL_EXTERNAL_README
from pipelines import eval_external_mirai_dataset  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="configs/hpc.yaml", help="YAML config path")
    p.add_argument("--resume", default=None, help="dir or last.pth")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    ds_csv_path          = cfg["ds_csv_path"]
    outpath              = cfg.get("outpath", ".")
    backbone             = cfg.get("backbone", "effv2s")
    gpus                 = cfg.get("gpus", [0])
    batch_size_img       = int(cfg.get("batch_size_img", 4))
    batch_size_exam      = int(cfg.get("batch_size_exam", 2))
    num_workers          = int(cfg.get("num_workers", 8))
    epochs               = int(cfg.get("epochs", 100))
    pretrain_epochs      = int(cfg.get("pretrain_epochs", 5))
    taper_epochs         = int(cfg.get("taper_epochs", 10))
    aux_weight_final     = float(cfg.get("aux_weight_final", 0.25))
    lr_phase3_mult       = float(cfg.get("lr_phase3_mult", 0.2))
    lr_min               = float(cfg.get("lr_min", 1e-5))
    lr                   = float(cfg.get("lr", 1e-3))
    gradient_accumulation = int(cfg.get("gradient_accumulation", 2))
    results_dir          = cfg.get("results_dir", None)
    run_name             = cfg.get("run_name", None)
    require_complete     = cfg.get("require_complete", True)
    ckpt_max_yr          = int(cfg.get("ckpt_max_yr", 4))
    lr_mult_heads        = float(cfg.get("lr_mult_heads", 3.0))
    wd_backbone          = float(cfg.get("wd_backbone", 1e-4))
    wd_heads             = float(cfg.get("wd_heads", 1e-4))
    debug_frac              = float(cfg.get("debug_frac", 1.0))
    freeze_backbone_epochs  = int(cfg.get("freeze_backbone_epochs", 0))

    # Cat specs
    cat_specs = [SimpleNamespace(name=n, key=n, num_classes=k, weight=1.0)
                 for (n, k) in cfg.get("cats", [])]

    # Reg specs
    reg_specs = [SimpleNamespace(name=k.lower(), key=k, weight=1.0, metric="mse")
                 for k in cfg.get("regs", [])]

    # Risk specs
    risk_cfg = cfg.get("risk", [])
    risk_specs = [SimpleNamespace(name=r.get("name", "risk"),
                                   horizons=r.get("horizons", 5),
                                   weight=r.get("weight", 1.0),
                                   key=r.get("key", "risk"))
                  for r in risk_cfg]

    run_train_staged(
        ds_csv_path,
        epochs=epochs,
        batch_size_img=batch_size_img,
        batch_size_exam=batch_size_exam,
        lr=lr,
        pretrain_epochs=pretrain_epochs,
        taper_epochs=taper_epochs,
        aux_weight_final=aux_weight_final,
        lr_phase3_mult=lr_phase3_mult,
        lr_min=lr_min,
        gpus=gpus,
        backbone=backbone,
        num_workers=num_workers,
        outpath=outpath,
        results_dir=results_dir,
        resume_path=args.resume or cfg.get("resume", None),
        cat_specs=cat_specs,
        reg_specs=reg_specs,
        risk_specs=risk_specs,
        require_complete=require_complete,
        gradient_accumulation=gradient_accumulation,
        run_name=run_name,
        ckpt_max_yr=ckpt_max_yr,
        lr_mult_heads=lr_mult_heads,
        wd_backbone=wd_backbone,
        wd_heads=wd_heads,
        debug_frac=debug_frac,
        freeze_backbone_epochs=freeze_backbone_epochs,
    )


if __name__ == "__main__":
    main()

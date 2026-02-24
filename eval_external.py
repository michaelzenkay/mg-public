#!/usr/bin/env python3
"""
Run inference with a trained mammography breast cancer risk model.

Loads a checkpoint and runs the model on an external dataset CSV, producing
per-exam predictions for 1–5 year cancer risk plus auxiliary outputs
(KVP, mAs, target, filter, manufacturer, age, etc.) from the model's aux heads.

The dataset CSV must follow the native format used during training
(pid_acc, image_path, bc, laterality, view, etc.).

Usage
-----
    # Use default config (configs/eval.yaml):
    python eval_external.py

    # Override specific fields:
    python eval_external.py --checkpoint results/run1/ --no-cancer

    # Explicit config:
    python eval_external.py --config configs/myeval.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

from pipelines import eval_external_mirai_dataset


def _resolve_checkpoint(spec: str) -> Path:
    """Resolve a run directory or explicit .pth path to a checkpoint file.

    For a directory, auto-detects *_best.pth; falls back to last.pth.
    """
    p = Path(spec)
    if p.is_dir():
        candidates = sorted(p.glob("*_best.pth"))
        if candidates:
            ckpt = candidates[0]
            print(f"[checkpoint] auto-detected best: {ckpt.name}")
            return ckpt
        fallback = p / "last.pth"
        print(f"[checkpoint] no *_best.pth found, using last.pth")
        return fallback
    return p


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/eval.yaml",
                   help="YAML config path (default: configs/eval.yaml). "
                        "CLI args override YAML values.")
    p.add_argument("--checkpoint",
                   help="Run directory (auto-selects best checkpoint) or explicit .pth path.")
    p.add_argument("--csv",
                   help="Dataset CSV in native training format.")
    p.add_argument("--output",
                   help="Output CSV path for per-exam predictions. "
                        "Default: <checkpoint_dir>/eval_preds.csv")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--gpus", type=int, nargs="+")
    p.add_argument("--incomplete", action="store_true", default=None,
                   help="Include exams with fewer than 4 views.")
    p.add_argument("--no-cancer", action="store_true", default=None,
                   help="Exclude patients who have or develop cancer.")
    args = p.parse_args()

    # Load YAML config
    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    elif args.config != "configs/eval.yaml":
        print(f"ERROR: config not found: {cfg_path}")
        return 1

    # CLI overrides YAML
    checkpoint  = args.checkpoint  or cfg.get("checkpoint")
    csv_path    = args.csv         or cfg.get("csv")
    output      = args.output      or cfg.get("output")
    batch_size  = args.batch_size  or cfg.get("batch_size",  8)
    num_workers = args.num_workers or cfg.get("num_workers", 4)
    gpus        = args.gpus        or cfg.get("gpus",        [0])
    incomplete  = args.incomplete  if args.incomplete else cfg.get("incomplete", False)
    no_cancer   = args.no_cancer   if args.no_cancer  else cfg.get("no_cancer",  False)

    if not checkpoint:
        print("ERROR: --checkpoint required (or set 'checkpoint' in eval.yaml)")
        return 1
    if not csv_path:
        print("ERROR: --csv required (or set 'csv' in eval.yaml)")
        return 1

    ckpt = _resolve_checkpoint(checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 1
    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    output_csv = output or str(ckpt.parent / "eval_preds.csv")

    print(f"Checkpoint  : {ckpt}")
    print(f"Dataset     : {csv_path}")
    print(f"Output      : {output_csv}")
    print(f"GPUs        : {gpus}  batch={batch_size}  workers={num_workers}")
    print(f"Incomplete  : {incomplete}  no_cancer: {no_cancer}")
    print()

    eval_external_mirai_dataset(
        checkpoint_path=str(ckpt),
        csv_path=str(csv_path),
        batch_size=batch_size,
        num_workers=num_workers,
        gpus=gpus,
        require_complete=not incomplete,
        output_csv=output_csv,
        no_cancer=no_cancer,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

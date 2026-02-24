"""Lightweight smoke test for the public export.

Checks:
- example config parses and contains expected keys
- core dependencies import
- project modules import
- CLI help commands run (no data required)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_cmd(args: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    print(f"OK  {' '.join(args)}")


def main() -> int:
    cfg_path = ROOT / 'configs' / 'example_train.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    required = [
        'run_name', 'ds_csv_path', 'results_dir', 'backbone', 'epochs',
        'cats', 'regs', 'risk'
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {cfg_path}: {missing}")
    print(f"OK  parsed config: {cfg_path.name}")

    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import PIL  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import tqdm  # noqa: F401
    print('OK  third-party imports')

    import checkpoint  # noqa: F401
    import datasets  # noqa: F401
    import eval_external  # noqa: F401
    import mg  # noqa: F401
    import models  # noqa: F401
    import pipelines  # noqa: F401
    import specs  # noqa: F401
    print('OK  project module imports')

    run_cmd(['mg.py', '--help'])
    run_cmd(['eval_external.py', '--help'])

    print('SMOKE TEST PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

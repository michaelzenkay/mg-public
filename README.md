# mg - Mammography Risk Prediction

Multi-task deep learning model for breast cancer risk prediction from screening mammograms. Predicts 1-5 year cancer risk at the exam level using a staged training approach that jointly learns auxiliary imaging features and risk prediction.

## Architecture

**Backbone:** EfficientNet-V2-S (grayscale-adapted, pretrained on ImageNet)

**Multi-head design** with three task types:
- **Classification heads** - view (CC/MLO), laterality (L/R), target material, filter, manufacturer, BC status
- **Regression heads** - tube voltage (KVP), tube current (mAs), time-to-diagnosis
- **Risk head** - 5-horizon breast cancer risk (1yr, 2yr, 3yr, 4yr, 5yr)

**Exam-level inference:** For risk prediction, all 4 views of a screening exam (LCC, LMLO, RCC, RMLO) are processed independently through the shared backbone, then aggregated via masked mean pooling before the risk head.

**Staged training** (three phases):
1. **Pretrain** - image-level auxiliary tasks only (backbone warmup)
2. **Taper** - gradual blend from auxiliary to exam-level risk
3. **Risk** - exam-level risk with minimal auxiliary weight

Gradient accumulation is used for exam-level batches (4 views per exam) to manage GPU memory.

## Project Structure

```
mg/
├── mg.py              # CLI entry point
├── eval_external.py   # Standalone CLI for external dataset evaluation
├── pipelines.py       # Training orchestrators, loss/metrics, param groups, eval loops
├── models.py          # EfficientNet backbone + MultiHeadNet architecture
├── datasets.py        # Data loading, ExamDataset classes, collate functions
├── checkpoint.py      # Model checkpoint save/load utilities
├── specs.py           # Task specification dataclasses (CatSpec, RegSpec, RiskSpec)
├── configs/           # YAML training configs
├── scripts/           # Smoke test
```

## Requirements

- Python 3.10+
- PyTorch >= 2.2
- torchvision >= 0.17
- pandas, numpy, scikit-learn, Pillow, tqdm, PyYAML

## Installation

### Option A: pip

```bash
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate mg-public
```

If using GPU, install the PyTorch build matching your CUDA version (the included `environment.yml` defaults to a CPU build template).

## Training

### Config

Training is driven by a YAML config file:

```yaml
ds_csv_path: /path/to/mammo_db.csv
results_dir: /path/to/output/
backbone: effv2s
gpus: [0, 1]
batch_size_img: 10
batch_size_exam: 2
gradient_accumulation: 2
num_workers: 15
epochs: 200
lr: 0.001

cats:
  - [view, 2]
  - [laterality, 2]
  - [target, 3]
  - [filter, 5]
  - [manufacturer, 3]
  - [bc, 2]

regs:
  - KVP
  - mAs

risk:
  - name: risk
    horizons: 5
    weight: 5.0
```

Use `configs/example_train.yaml` as a starting template.

### Curriculum / Optimization

- Phase 1 (`pretrain_epochs`): aux-only
- Phase 2 (`taper_epochs`): gradually increase risk weight while tapering auxiliary weight
- Optional `freeze_backbone_epochs`: freeze backbone at the start of Phase 2 (backbone LR = 0) to warm up exam-level heads before restoring backbone updates
- Phase 3: LR drop (`lr_phase3_mult`) + cosine annealing to `lr_min`

Parameter groups in `pipelines.py` control separate backbone/head learning rates, separate weight decay, and zero weight decay for bias/norm parameters.

### Targeting / Filtering

- Auxiliary tasks do **not** require a universal minimum follow-up duration.
- Risk labels are constructed with horizon-wise masking based on available follow-up at training time.
- This lets short-follow-up images contribute to auxiliary supervision while preventing invalid risk targets.

### Run

```bash
python mg.py --config configs/your_config.yaml
```

### Resume from checkpoint

```bash
python mg.py --config configs/your_config.yaml --resume path/to/results_dir/
```

### Outputs

Each run creates a timestamped results directory containing:
- `last.pth` - most recent checkpoint
- `{heads}_{backbone}_best.pth` - best checkpoint (by validation AUC)
- `metrics.csv` - per-epoch training/validation metrics
- `val_preds_best.parquet` / `.csv` - per-exam predictions at best checkpoint
- split files (used to reproduce train/val splits)

## Dataset Format

### Training CSV

The training CSV (`ds_csv_path`) should have one row per image with columns:

| Column | Description |
|---|---|
| `pid` | Patient ID (7-digit, zero-padded) |
| `pid_acc` | `{pid}_{accession}` exam identifier |
| `image_path` | Full path to PNG mammogram |
| `laterality` | `L` or `R` |
| `view` | `CC` or `MLO` |
| `bc` | `BC` or `NOBC` |
| `months_to_dx` | Months from exam to cancer diagnosis (NaN if no cancer) |
| `followup_months` | Months of cancer-free followup |
| `mAs` | Tube current |
| `KVP` | Tube voltage |
| `Manufacturer` | `GE`, `HOLOGIC`, or `SIEMENS` |
| `target` | Anode target material |
| `filter` | Filter material |

Patient-level train/val splits (80/20) are stratified by cancer status with no patient leakage.

Unit-suffixed strings such as `age="63Y"` are parsed during dataset loading.

### External Evaluation CSV

Two formats are accepted and auto-detected at load time.

#### Native format

No DICOM metadata required — only columns needed for exam identification and outcome labeling.

```csv
pid,pid_acc,laterality,view,image_path,bc,months_to_dx,followup_months
P001,P001_E001,L,CC,/data/p001_lcc.png,NOBC,,24
P001,P001_E001,L,MLO,/data/p001_lmlo.png,NOBC,,24
P001,P001_E001,R,CC,/data/p001_rcc.png,NOBC,,24
P001,P001_E001,R,MLO,/data/p001_rmlo.png,NOBC,,24
P002,P002_E001,L,CC,/data/p002_lcc.png,BC,14,14
P002,P002_E001,L,MLO,/data/p002_lmlo.png,BC,14,14
P002,P002_E001,R,CC,/data/p002_rcc.png,BC,14,14
P002,P002_E001,R,MLO,/data/p002_rmlo.png,BC,14,14
```

| Column | Description |
|---|---|
| `pid` | Patient ID string |
| `pid_acc` | `{pid}_{accession}` — unique exam identifier |
| `laterality` | `L` or `R` |
| `view` | `CC` or `MLO` |
| `image_path` | Absolute path to 16-bit grayscale PNG |
| `bc` | `BC` (cancer) or `NOBC` |
| `months_to_dx` | Months from exam to cancer diagnosis (blank/NaN if NOBC) |
| `followup_months` | Months of known cancer-free followup (must be > 0) |

#### MIRAI format

If your CSV has `patient_id` / `file_path` columns (MIRAI convention), it is converted automatically — no preprocessing needed.

```csv
patient_id,exam_id,laterality,view,file_path,years_to_cancer,years_to_last_followup,split_group
P001,P001_E001,L,CC,/data/p001_lcc.png,100,2,test
...
```

| Column | Description |
|---|---|
| `patient_id` | Patient ID string → mapped to `pid` |
| `exam_id` | Exam identifier → mapped to `pid_acc` |
| `file_path` | Image path → mapped to `image_path` |
| `laterality` | `L` or `R` |
| `view` | `CC`, `MLO`, or prefixed forms (`LCC`, `RMLO`, etc.) — normalized automatically |
| `years_to_cancer` | Years to diagnosis; values ≥ 5 or blank → NOBC |
| `years_to_last_followup` | Years of cancer-free followup → converted to `followup_months` |

Detection is based on the presence of `patient_id` or `file_path` columns. When a MIRAI CSV is loaded, the conversion is logged: `[mirai->native] N rows | M exams | ...`

#### Notes (both formats)

- Each exam requires 4 rows: LCC, LMLO, RCC, RMLO (use `--incomplete` to relax)
- `followup_months` controls which risk horizons are valid — e.g. 24 months unlocks 1yr and 2yr only
- AUC is computed per horizon; requires both BC and NOBC exams in the dataset

## External Evaluation

For a collaborator who only needs inference, the minimal handoff is:
- this repo
- the checkpoint `.pth` file (or a `checkpoint/` directory containing it)
- a dataset CSV in native or MIRAI format

**Quickstart (3 steps):**

1. Edit `configs/example_inference.yaml` — fill in `checkpoint`, `csv`, and `output`
2. Run: `python eval_external.py`
3. Predictions are written to `output` as CSV; a same-name parquet sidecar is also attempted

The config file contains both eval parameters and the model architecture spec in one place. The `cats`/`regs`/`risk_specs` entries must match the heads in the `.pth`; the defaults in `example_inference.yaml` match the provided checkpoint. Adjust only if you trained a checkpoint with a different head configuration.

For the provided checkpoint, most users only need to change:
- `checkpoint`
- `csv`
- `output`

Change `gpus` to `[]` for CPU-only inference. On Windows, set `num_workers: 0`.

### CLI usage

```bash
# Use a config file (recommended)
python eval_external.py --config configs/example_inference.yaml

# Override specific fields from a config
python eval_external.py --config configs/example_inference.yaml --no-cancer

# Pass everything via CLI args
python eval_external.py \
    --checkpoint results/last.pth \
    --csv my_dataset.csv \
    --num-workers 8 \
    --gpus 0 1 \
    --output predictions.csv

# Include incomplete exams (fewer than 4 views)
python eval_external.py --checkpoint model.pth --csv data.csv --incomplete
```

`--no-cancer` works with either native or MIRAI-format CSVs and filters at the patient level before evaluation.

When a run directory is passed, evaluation prefers `*_best.pth` and falls back to `last.pth`.

See `configs/example_inference.yaml` for a full annotated config template.

### Prediction file

The main output CSV has one row per exam with these columns:
- `exam_id`
- `risk_h1yr_prob` ... `risk_h5yr_prob`
- `risk_h1yr_true` ... `risk_h5yr_true`
- `risk_h1yr_valid` ... `risk_h5yr_valid`

For horizons beyond the available follow-up, `*_true` is blank and `*_valid` is `False`.

### Python API

```python
import yaml
from pipelines import eval_external_mirai_dataset

run_cfg = yaml.safe_load(open("configs/example_inference.yaml"))

metrics = eval_external_mirai_dataset(
    checkpoint_path="path/to/checkpoint.pth",
    csv_path="path/to/data.csv",   # native or MIRAI format
    batch_size=8,
    gpus=[0],
    require_complete=True,
    output_csv="predictions.csv",
    run_cfg=run_cfg,               # provides backbone/cats/regs/risk_specs
)

print(f"Mean AUC: {metrics['auc_mean']:.4f}")
for h in range(1, 6):
    print(f"  {h}-year AUC: {metrics.get(f'auc_h{h}yr', float('nan')):.4f}")
```

### Example output

```
============================================================
EVALUATION RESULTS
============================================================
  1-year AUC: 0.8234
  2-year AUC: 0.8156
  3-year AUC: 0.8089
  4-year AUC: 0.7923
  5-year AUC: 0.7801
  Mean AUC:     0.8041
============================================================
```

## Troubleshooting

- **Import errors**: Run from the repo directory, or add it to `PYTHONPATH`
- **CUDA out of memory**: Reduce `--batch-size` (try 4 or 2)
- **Low exam count**: Check column names match exactly, all 4 views present per exam, `followup_months` > 0
- **MIRAI CSV not detected**: Confirm `patient_id` or `file_path` column is present; check for extra whitespace in headers
- **AUC is nan**: Dataset contains only one class (all BC or all NOBC) — AUC requires both
- **Slow loading**: Increase `--num-workers` (use 0 on Windows)

## Smoke Test

After installing dependencies, run:

```bash
python scripts/smoke_test.py
```

This checks config parsing, core imports, and CLI `--help` paths without requiring training data.

## License

See [LICENSE](LICENSE).

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
| `bc` | `bc` or `nobc` |
| `months_to_dx` | Months from exam to cancer diagnosis (NaN if no cancer) |
| `followup_months` | Months of cancer-free followup |
| `mAs` | Tube current |
| `KVP` | Tube voltage |
| `Manufacturer` | `GE`, `HOLOGIC`, or `SIEMENS` |
| `target` | Anode target material |
| `filter` | Filter material |

Patient-level train/val splits (80/20) are stratified by cancer status with no patient leakage.

Unit-suffixed strings such as `age="63Y"` are parsed during dataset loading.

### External Evaluation CSV (MIRAI format)

The external eval CSV follows the [MIRAI](https://github.com/yala/Mirai) format. No DICOM headers or auxiliary labels needed.

```csv
patient_id,exam_id,laterality,view,file_path,years_to_cancer,years_to_last_followup,split_group
P001,P001_E001,L,CC,/data/p001_lcc.png,0,1,test
P001,P001_E001,L,MLO,/data/p001_lmlo.png,0,1,test
P001,P001_E001,R,CC,/data/p001_rcc.png,0,1,test
P001,P001_E001,R,MLO,/data/p001_rmlo.png,0,1,test
P002,P002_E001,L,CC,/data/p002_lcc.png,0,1,test
P002,P002_E001,L,MLO,/data/p002_lmlo.png,0,1,test
P002,P002_E001,R,CC,/data/p002_rcc.png,0,1,test
P002,P002_E001,R,MLO,/data/p002_rmlo.png,0,1,test
```

| Column | Description |
|---|---|
| `patient_id` | Patient identifier |
| `exam_id` | `{patient_id}_{accession}` exam identifier |
| `laterality` | `L` or `R` |
| `view` | `CC` or `MLO` |
| `file_path` | Full path to PNG mammogram |
| `years_to_cancer` | Years from exam to cancer diagnosis (empty/NaN if no cancer) |
| `years_to_last_followup` | Years of cancer-free followup (must be >= 1 year) |
| `split_group` | e.g. `test` |

- Each exam needs 4 rows (LCC, LMLO, RCC, RMLO)
- `years_to_cancer`: leave empty/NaN for patients without cancer
- `years_to_last_followup`: must be >= 1 year
- By default, only complete exams (all 4 views) are evaluated

## External Evaluation

### CLI usage

```bash
# Basic
python eval_external.py --checkpoint model.pth --csv data.csv

# With options
python eval_external.py \
    --checkpoint results/last.pth \
    --csv my_dataset.csv \
    --batch_size 16 \
    --num_workers 8 \
    --gpus 0 1 \
    --output_csv predictions.csv

# Include incomplete exams
python eval_external.py --checkpoint model.pth --csv data.csv --incomplete
```

When a run directory is passed, external evaluation prefers `*_best.pth` and falls back to `last.pth` if a best checkpoint is not present.

### Python API

```python
from pipelines import eval_external_mirai_dataset

metrics = eval_external_mirai_dataset(
    checkpoint_path="path/to/checkpoint.pth",
    csv_path="path/to/data.csv",
    batch_size=8,
    gpus=[0],
    require_complete=True,
    output_csv="predictions.csv"
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
- **CUDA out of memory**: Reduce `--batch_size` (try 4 or 2)
- **Low exam count**: Check column names are exact (case-sensitive), all 4 views present, followup >= 1 year
- **Slow loading**: Increase `--num_workers`

## Smoke Test

After installing dependencies, run:

```bash
python scripts/smoke_test.py
```

This checks config parsing, core imports, and CLI `--help` paths without requiring training data.

## License

See [LICENSE](LICENSE).

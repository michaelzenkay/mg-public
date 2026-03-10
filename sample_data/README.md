# Sample Data — CBIS-DDSM (3 complete exams)

This directory contains the helper files needed to run mg-public end-to-end on
3 publicly available screening mammogram exams from the CBIS-DDSM dataset.

## What's included

```
sample_data/
├── prep_ddsm.py       # DICOM → PNG conversion script (outputs MIRAI-format CSV)
└── sample_mirai.csv   # MIRAI-format CSV for the 3 sample exams
```

---

## Quick start

### Step 1 — Download DICOMs from TCIA

Source: CBIS-DDSM, freely available from The Cancer Imaging Archive:

> https://www.cancerimagingarchive.net/collection/cbis-ddsm/

Download the full-mammogram images for these three cases from the **Calc-Test** subset:

| Patient | Folders to download |
|---|---|
| P_00038 | `Calc-Test_P_00038_LEFT_CC/MLO`, `Calc-Test_P_00038_RIGHT_CC/MLO` |
| P_00077 | `Calc-Test_P_00077_LEFT_CC/MLO`, `Calc-Test_P_00077_RIGHT_CC/MLO` |
| P_00140 | `Calc-Test_P_00140_LEFT_CC/MLO`, `Calc-Test_P_00140_RIGHT_CC/MLO` |

### Step 2 — Convert DICOMs to PNG

```bash
pip install pydicom

python sample_data/prep_ddsm.py \
    --ddsm_dir  /path/to/downloaded/dicoms/ \
    --out_dir   /path/to/pngs/ \
    --out_csv   sample_data/my_dataset.csv
```

This writes 16-bit grayscale PNGs and a MIRAI-format CSV.
`sample_mirai.csv` uses placeholder paths — update `file_path` to match where
your PNGs landed, or use the CSV produced by `prep_ddsm.py` directly.

### Step 3 — Run inference

Edit `configs/example_inference.yaml` to point at your CSV and a checkpoint, then:

```bash
python eval_external.py
```

For this sample run:
- set `checkpoint` to the shipped `.pth` or `checkpoint/` directory
- set `csv` to the CSV produced by `prep_ddsm.py` or to `sample_data/sample_mirai.csv` after fixing its placeholder paths
- set `gpus: []` if you are running on CPU
- set `num_workers: 0` on Windows

---

## Notes

- All 3 exams are NOBC with 24 months of follow-up. AUC will be `nan` (requires both
  BC and NOBC cases). Risk scores are still valid.
- `sample_mirai.csv` uses MIRAI column names and is auto-detected at load time.

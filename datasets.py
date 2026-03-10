"""Dataset classes and data loading utilities for mammography training."""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from os.path import exists
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


# Categorical maps — image-level only: target, filter, laterality, view
#                  — exam-level only:  bc
#                  — both levels:      manufacturer
_TARGET_MAP       = {'MOLYBDENUM': 1, 'RHODIUM': 2, 'TUNGSTEN': 3}
_FILTER_MAP       = {'ALUMINUM': 1, 'COPPER': 2, 'MOLYBDENUM': 3, 'RHODIUM': 4, 'SILVER': 5}
_MANUFACTURER_MAP = {'GE': 1, 'HOLOGIC': 2, 'SIEMENS': 3}
_LATERALITY_MAP   = {'L': 1, 'R': 2}
_VIEW_MAP         = {'CC': 1, 'MLO': 2}
_BC_MAP           = {'BC': 1, 'NOBC': 2}

# Regression ranges (lowercase keys) — both levels share this dict.
# pipelines.py looks up via sp.key.lower().
_REG_RANGES = {
    "kvp":             (18, 50),
    "mas":             (0, 300),
    "age":             (18, 100),
    "weight":          (0, 250),
    "months_to_dx":    (-110, 50),
}

_HORIZONS_MONTHS = [12, 24, 36, 48, 60]
_VIEW_TYPE_TO_ID = {"LCC": 0, "LMLO": 1, "RCC": 2, "RMLO": 3}

_MIRAI_VIEW_NORM = {"CC": "CC", "MLO": "MLO", "LCC": "CC", "LMLO": "MLO",
                    "RCC": "CC", "RMLO": "MLO"}


def _normalize_mirai_to_native(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect and convert MIRAI-format CSV columns to mg native format.

    MIRAI columns: patient_id, exam_id, file_path, laterality, view,
                   years_to_cancer, years_to_last_followup
    Native columns: pid, pid_acc, image_path, laterality, view,
                    bc, months_to_dx, followup_months

    Returns df unchanged if already in native format (has pid_acc or image_path).
    """
    cols = set(df.columns.str.lower())
    if 'patient_id' not in cols and 'file_path' not in cols:
        return df  # already native format

    _REQUIRED = ['patient_id', 'exam_id', 'laterality', 'view',
                 'file_path', 'years_to_last_followup']
    missing = [c for c in _REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"MIRAI CSV missing columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame(index=df.index)
    out['pid']        = df['patient_id'].astype(str).str.strip()
    out['pid_acc']    = df['exam_id'].astype(str).str.strip()
    out['image_path'] = df['file_path'].astype(str).str.strip()
    out['laterality'] = df['laterality'].astype(str).str.strip().str.upper()
    out['view']       = (df['view'].astype(str).str.strip().str.upper()
                         .map(lambda v: _MIRAI_VIEW_NORM.get(v, v)))

    ytc = pd.to_numeric(df.get('years_to_cancer',
                                pd.Series(np.nan, index=df.index)), errors='coerce')
    is_cancer = ytc.notna() & (ytc >= 0) & (ytc < 5)
    out['bc']              = np.where(is_cancer, 'BC', 'NOBC')
    out['months_to_dx']    = np.where(is_cancer, ytc * 12.0, np.nan)
    out['followup_months'] = pd.to_numeric(
        df['years_to_last_followup'], errors='coerce') * 12.0

    for col in ['KVP', 'mAs', 'age', 'weight', 'target', 'filter', 'manufacturer', 'split_group']:
        out[col] = df[col] if col in df.columns else np.nan

    log(f"[mirai->native] {len(out)} rows | {out['pid_acc'].nunique()} exams | "
        f"{out['pid'].nunique()} patients | {int(is_cancer.sum())} cancer rows")
    return out


def _load_breast_crop(fn, target_h, target_w, thresh_pct):
    """Crop to breast bounding box, then resize to (1, target_h, target_w)."""
    im_pil = Image.open(fn)
    im_np = np.array(im_pil).astype(np.float32)

    im_min, im_max = im_np.min(), im_np.max()
    im_norm = (im_np - im_min) / (im_max - im_min + 1e-5)

    mask = im_norm > thresh_pct
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        im_crop = im_norm[rmin:rmax+1, cmin:cmax+1]
    else:
        im_crop = im_norm  # fallback if threshold fails

    im_t = torch.as_tensor(im_crop).unsqueeze(0).unsqueeze(0)
    im_resized = F.interpolate(im_t, size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
    return im_resized.squeeze(0)  # (1, H, W)


def _make_risk_targets(bc, months_to_dx, followup_months):
    """
    bc: 0 or 1
    months_to_dx: float or NaN (time from exam to diagnosis if bc=1)
    followup_months: float (months of known cancer-free followup)

    Returns:
        targets: (H,) float32 — 1 if cancer by horizon, else 0
        mask: (H,) bool     — True if horizon is valid (within followup)
    """
    H = len(_HORIZONS_MONTHS)
    targets = np.zeros(H, dtype=np.float32)
    mask = np.zeros(H, dtype=bool)

    if pd.isna(followup_months) or followup_months <= 0:
        return targets, mask

    if bc == 1 and pd.isna(months_to_dx):
        return targets, mask

    if bc == 1 and not pd.isna(months_to_dx) and months_to_dx >= 0:
        effective_followup = months_to_dx
    else:
        effective_followup = followup_months

    for i, h in enumerate(_HORIZONS_MONTHS):
        if h <= effective_followup:
            mask[i] = True
            if bc == 1 and not pd.isna(months_to_dx) and 0 <= months_to_dx <= h:
                targets[i] = 1.0
        elif bc == 1 and not pd.isna(months_to_dx) and 0 <= months_to_dx <= h:
            mask[i] = True
            targets[i] = 1.0

    return targets, mask


def _write_dataset_stats(results_dir, key, data):
    """Merge stats dict into results_dir/dataset_stats.json under key."""
    path = Path(results_dir) / "dataset_stats.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    existing[key] = data
    path.write_text(json.dumps(existing, indent=2))


# Tracks files that have already been logged as unreadable; suppresses
# per-epoch repetition for persistently broken images (e.g. corrupt PNGs).
_BAD_FILES: set = set()


def _write_aux_diag(clin, img_fns, split, results_dir):
    """Compute per-split column diagnostics and write to dataset_stats.json under aux_diag[split]."""
    img_set = set(img_fns)
    df = clin[clin['image_path'].isin(img_set)].copy() if 'image_path' in clin.columns else clin.copy()

    numeric_cols = ['kvp', 'mas', 'age', 'weight', 'followup_months', 'months_to_dx']
    categorical_cols = ['manufacturer', 'bc', 'laterality', 'view', 'target', 'filter']

    numeric = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        valid = vals.dropna()
        if len(valid) == 0:
            numeric[col] = {"null_pct": 100.0}
        else:
            numeric[col] = {
                "min": round(float(valid.min()), 2),
                "max": round(float(valid.max()), 2),
                "mean": round(float(valid.mean()), 2),
                "null_pct": round(float(vals.isna().mean() * 100), 1),
            }

    categorical = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=True).head(10)
        categorical[col] = {str(k): int(v) for k, v in vc.items()}

    followup = {}
    if 'followup_months' in df.columns:
        fu = pd.to_numeric(df['followup_months'], errors='coerce')
        total = max(len(fu), 1)
        for h_yr in range(1, 6):
            n = int((fu >= h_yr * 12).sum())
            followup[f"{h_yr}yr"] = {"n": n, "pct": round(100 * n / total, 1)}

    # Merge this split into aux_diag nested key
    path = Path(results_dir) / "dataset_stats.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    aux_diag = existing.get("aux_diag", {})
    aux_diag[split] = {
        "n_images": len(img_fns),
        "n_rows_matched": len(df),
        "numeric": numeric,
        "categorical": categorical,
        "followup": followup,
    }
    existing["aux_diag"] = aux_diag
    path.write_text(json.dumps(existing, indent=2))


def log(msg):
    print(msg)


def train_test_load(incsv, suffix='.png', complete=False, results_dir=None):
    df = _normalize_mirai_to_native(pd.read_csv(incsv, low_memory=False).copy())

    # Ensure pid_acc exists
    if 'pid_acc' not in df.columns:
        if {'pid','accession'}.issubset(df.columns):
            df['pid_acc'] = df['pid'].astype(str) + "_" + df['accession'].astype(str)
        else:
            raise KeyError("Need 'pid_acc' or ('pid','accession') columns")

    if 'pid' not in df.columns:
        df['pid'] = df['pid_acc'].str.split('_').str[0]

    # Clean required fields: mAs > 0, view non-empty
    df['mAs'] = pd.to_numeric(df['mAs'], errors='coerce')
    df['view'] = df['view'].astype(str).str.strip()

    df_clean = df[df['mAs'].gt(0) & df['view'].ne('')].copy()

    # Filter complete exams if requested
    if complete and 'is_complete' in df_clean.columns:
        valid_df = df_clean[df_clean['is_complete'].fillna(False)].copy()
    else:
        valid_df = df_clean.copy()

    # Keep only rows with desired suffix
    has_suffix = valid_df['image_path'].astype(str).str.endswith(suffix, na=False)
    valid_df = valid_df[has_suffix].copy()

    # Exam list
    pid_accs = valid_df['pid_acc'].unique()
    if len(pid_accs) < 2:
        raise ValueError("Not enough exams after filtering to split.")

    # Patient-level split
    patient_bc = valid_df.groupby('pid')['bc'].apply(
        lambda x: 1 if x.str.lower().eq('bc').any() else 0
    )
    patients = patient_bc.index.values
    labels = patient_bc.values

    train_pats, val_pats = train_test_split(
        patients, test_size=0.2, random_state=42, stratify=labels
    )

    train_paths = valid_df[valid_df['pid'].isin(train_pats)]['image_path'].tolist()
    val_paths = valid_df[valid_df['pid'].isin(val_pats)]['image_path'].tolist()

    if results_dir is not None:
        bc_pats = int((labels == 1).sum())
        _write_dataset_stats(results_dir, "aux", {
            "total_images":   len(train_paths) + len(val_paths),
            "total_patients": len(patients),
            "total_exams":    int(valid_df['pid_acc'].nunique()),
            "train_images":   len(train_paths),
            "val_images":     len(val_paths),
            "train_patients": len(train_pats),
            "val_patients":   len(val_pats),
            "train_exams":    int(valid_df[valid_df['pid'].isin(train_pats)]['pid_acc'].nunique()),
            "val_exams":      int(valid_df[valid_df['pid'].isin(val_pats)]['pid_acc'].nunique()),
            "bc_patients":    bc_pats,
            "nobc_patients":  len(patients) - bc_pats,
        })

    return train_paths, val_paths


def train_test_load_exams(incsv,
                          suffix='.png',
                          results_dir: str | Path | None = None,
                          splits_csv: str | Path | None = None,
                          val_fold: int = 0,
                          n_splits: int = 5,
                          make_if_missing: bool = True,):
    """
    Split at PATIENT level to prevent leakage, return exam IDs.

    Steps:
    1. Load and filter data
    2. Get unique patients
    3. Determine cancer status per patient (any exam = cancer -> patient is cancer)
    4. Split PATIENTS into train/val (stratified by cancer status)
    5. Return exam IDs belonging to train patients vs val patients
    """
    # --- Optional: deterministic K-fold splits (exam-level CSV) ---
    if splits_csv is None and results_dir is not None:
        splits_csv = Path(results_dir) / "exam_splits_5fold.csv"

    if splits_csv is not None:
        splits_csv = Path(splits_csv)
        if (not splits_csv.exists()) and make_if_missing:
            write_exam_splits_5fold(
                incsv=incsv,
                out_csv=str(splits_csv),
                n_splits=n_splits,
                suffix=suffix,
                random_state=42,
            )
        if splits_csv.exists():
            return train_val_exams_from_splits(
                splits_csv=str(splits_csv),
                val_fold=val_fold,
                n_splits=n_splits,
                results_dir=results_dir,
            )
    df = _normalize_mirai_to_native(pd.read_csv(incsv).copy())

    # Ensure pid_acc exists
    if 'pid_acc' not in df.columns:
        if {'pid', 'accession'}.issubset(df.columns):
            df['pid_acc'] = df['pid'].astype(str) + "_" + df['accession'].astype(str)
        else:
            raise KeyError("Need 'pid_acc' or ('pid', 'accession') columns")

    # Ensure pid exists
    if 'pid' not in df.columns:
        df['pid'] = df['pid_acc'].str.split('_').str[0]

    # Filter valid images (view not empty)
    df['view'] = df['view'].astype(str).str.strip()
    df = df[df['view'].ne('')].copy()
    has_suffix = df['image_path'].astype(str).str.endswith(suffix, na=False)
    df = df[has_suffix].copy()
    log(f"After filters: {len(df)} rows, {df['pid'].nunique()} patients, {df['pid_acc'].nunique()} exams")

    # Patient-level cancer status: 1 if ANY of their exams has cancer
    patient_bc = df.groupby('pid')['bc'].apply(
        lambda x: 1 if x.str.lower().eq('bc').any() else 0
    )

    patients = patient_bc.index.values
    labels = patient_bc.values

    log(f"Patient cancer prevalence: {labels.sum()} / {len(labels)} ({100*labels.mean():.1f}%)")

    # Split PATIENTS (not exams) into train/val
    train_pats, val_pats = train_test_split(
        patients,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Get all exams for train patients
    train_exams = df[df['pid'].isin(train_pats)]['pid_acc'].unique()

    # Get all exams for val patients
    val_exams = df[df['pid'].isin(val_pats)]['pid_acc'].unique()

    # Sanity check: no patient appears in both splits
    train_pats_from_exams = set(df[df['pid_acc'].isin(train_exams)]['pid'].unique())
    val_pats_from_exams = set(df[df['pid_acc'].isin(val_exams)]['pid'].unique())
    overlap = train_pats_from_exams & val_pats_from_exams

    if len(overlap) > 0:
        raise ValueError(f"PATIENT LEAKAGE: {len(overlap)} patients in both train and val!")

    # Stats
    train_cancer = df[df['pid_acc'].isin(train_exams)].groupby('pid_acc')['bc'].apply(
        lambda x: x.str.lower().eq('bc').any()
    ).sum()
    val_cancer = df[df['pid_acc'].isin(val_exams)].groupby('pid_acc')['bc'].apply(
        lambda x: x.str.lower().eq('bc').any()
    ).sum()

    log(f"\n=== SPLIT SUMMARY ===")
    log(f"Train: {len(train_pats)} patients, {len(train_exams)} exams, {train_cancer} cancer")
    log(f"Val:   {len(val_pats)} patients, {len(val_exams)} exams, {val_cancer} cancer")
    log(f"No patient leakage: OK")
    log(f"=====================\n")

    return train_exams, val_exams


class ds2024(Dataset):
    def __init__(self, img_fns, ds, train=True,
                 target_h=1664, target_w=1024, thresh_pct=0.05,
                 split="train", results_dir=None):
        self.imgfns = img_fns
        self.clin = pd.read_csv(ds, low_memory=False)
        self.clin.columns = self.clin.columns.str.lower()
        self.train = train
        self.target_h = target_h
        self.target_w = target_w
        self.thresh_pct = thresh_pct

        if results_dir is not None:
            _write_aux_diag(self.clin, img_fns, split, results_dir)

        # Image-level categorical maps (per-view properties)
        self.target_map       = _TARGET_MAP
        self.filter_map       = _FILTER_MAP
        self.manufacturer_map = _MANUFACTURER_MAP
        self.laterality_map   = _LATERALITY_MAP
        self.view_map         = _VIEW_MAP
        # bc_map not set — bc is exam-level only

        self.reg_ranges = _REG_RANGES

        # Keys set by run_train
        self.cat_keys = []
        self.reg_keys = []

    def _parse_numeric(self, val, field_name=""):
        """Safely parse numeric value with debug."""
        if pd.isna(val): return np.nan
        try:
            # Handle string cleaning
            if isinstance(val, str):
                val = val.strip().upper()
                val = val.rstrip('Y').rstrip('KV').rstrip('MAS')  # strip units
                val = val.replace(',', '')
            return float(val)
        except (ValueError, TypeError):
            log(f"[WARN] Could not parse {field_name}='{val}' as numeric")
            return np.nan

    def getinfo(self, fn):
        # Filter info for image from clinical dataframe
        clininfo = self.clin[self.clin['image_path'] == fn]
        if len(clininfo) == 0:
            log(f"[WARN] No clinical info for: {fn}")
            cat_dict = {k: 0 for k in self.cat_keys}
            reg_dict = {k: np.nan for k in self.reg_keys}
            return cat_dict, reg_dict, pd.Series()

        row = clininfo.iloc[0]
        cat_dict, reg_dict = {}, {}
        try:
            # --- CATEGORICAL ---
            for key in self.cat_keys:
                raw_val = row.get(key.lower())
                if pd.isna(raw_val):
                    cat_dict[key] = 0
                else:
                    val = str(raw_val).upper().strip()
                    if key.lower() == 'manufacturer':
                        val = val.replace(',', '').strip()
                        if val.startswith('GE') or 'GENERAL ELECTRIC' in val: val = 'GE'
                        elif 'HOLOGIC' in val: val = 'HOLOGIC'
                        elif 'SIEMENS' in val: val = 'SIEMENS'
                    map_attr = getattr(self, f"{key.lower()}_map", {})
                    cat_dict[key] = map_attr.get(val, 0)

            # --- REGRESSION ---
            # _parse_numeric handles unit strings: age='70Y'->70, kvp='30KV'->30, etc.
            for key in self.reg_keys:
                reg_dict[key] = self._parse_numeric(row.get(key.lower()), field_name=key)

        except Exception as e:
            log(f"[ERROR] getinfo failed for {fn}: {e}")
            cat_dict = {k: 0 for k in self.cat_keys}
            reg_dict = {k: np.nan for k in self.reg_keys}

        return cat_dict, reg_dict, row

    def __len__(self):
        return len(self.imgfns)

    def __getitem__(self, idx):
        fn = self.imgfns[idx]
        if not exists(fn):
            return None

        try:
            im_patch = _load_breast_crop(fn, self.target_h, self.target_w, self.thresh_pct)
        except Exception as e:
            if fn not in _BAD_FILES:
                log(f"Skipping {fn}: {e}")
                _BAD_FILES.add(fn)
            return None

        cat_dict, reg_dict, row = self.getinfo(fn)

        try:
            followup_months = float(row.get('followup_months', np.nan))
        except (ValueError, TypeError):
            followup_months = np.nan

        cats = torch.tensor([cat_dict[k] for k in self.cat_keys], dtype=torch.long)
        reg_vals = torch.tensor([reg_dict[k] for k in self.reg_keys], dtype=torch.float32)

        # bc and months_to_dx are exam-level — read directly from row for risk targets
        bc = 1 if str(row.get('bc', '')).upper() == 'BC' else 0
        try:
            months_to_dx = float(row.get('months_to_dx', np.nan))
        except (ValueError, TypeError):
            months_to_dx = np.nan

        risk_targets, risk_mask = _make_risk_targets(bc, months_to_dx, followup_months)
        risk_targets = torch.from_numpy(risk_targets)
        risk_mask = torch.from_numpy(risk_mask)

        seg_patch = torch.zeros((1, self.target_h, self.target_w))
        return im_patch, seg_patch, cats, reg_vals, risk_targets, risk_mask


class ExamDataset(Dataset):
    """Yields all views for an exam, with exam-level labels."""

    def __init__(self, exam_ids, ds_csv, train=True, target_h=882, target_w=512,
                 thresh_pct=0.05, max_views=4, require_complete=False):
        self.exam_ids = list(exam_ids)
        self.clin = _normalize_mirai_to_native(
            pd.read_csv(ds_csv, low_memory=False))
        self.clin.columns = self.clin.columns.str.lower()
        self.clin = self.clin[self.clin['image_path'].astype(str).str.endswith('.png', na=False)].copy()
        self.train = train
        self.target_h = target_h
        self.target_w = target_w
        self.thresh_pct = thresh_pct
        self.max_views = max_views
        self.require_complete = require_complete

        # Build exam -> image paths mapping (vectorised via groupby)
        exam_set = set(self.exam_ids)
        sub = self.clin[self.clin['pid_acc'].isin(exam_set)]
        grouped_paths = sub.groupby('pid_acc')['image_path'].apply(list)
        self.exam_to_paths = {eid: grouped_paths.get(eid, []) for eid in self.exam_ids}

        # Filter to complete exams if required
        if require_complete:
            # Pre-build exam -> view_name set via groupby instead of per-exam scan
            def _get_view_names(grp):
                names = set()
                for _, row in grp.iterrows():
                    lat = str(row.get('laterality', '')).strip().upper()
                    view = str(row.get('view', '')).strip().upper()
                    if lat in ('L', 'R') and view in ('CC', 'MLO'):
                        names.add(f"{lat}{view}")
                return names
            required_views = {'LCC', 'LMLO', 'RCC', 'RMLO'}
            exam_views_map = sub.groupby('pid_acc').apply(_get_view_names)
            complete_exams = [eid for eid in self.exam_ids
                              if required_views.issubset(exam_views_map.get(eid, set()))]
            split_tag = "Train" if self.train else "Val"
            log(f"Complete exam filter [{split_tag}]: {len(complete_exams)}/{len(self.exam_ids)} exams have all 4 views")
            self.exam_ids = complete_exams

        # Exam-level categorical maps (patient/exam-level properties)
        self.bc_map           = _BC_MAP
        self.manufacturer_map = _MANUFACTURER_MAP

        self.cat_keys = []
        self.reg_keys = []
        self.reg_ranges = _REG_RANGES

    def get_exam_views(self, exam_id):
        """
        Get laterality and view for all images in an exam.

        Returns:
            dict: {view_name: image_path} where view_name is like 'LCC', 'LMLO', 'RCC', 'RMLO'
        """
        exam_rows = self.clin[self.clin['pid_acc'] == exam_id]
        views = {}

        for _, row in exam_rows.iterrows():
            lat = str(row.get('laterality', '')).strip().upper()
            view = str(row.get('view', '')).strip().upper()

            if lat in ['L', 'R'] and view in ['CC', 'MLO']:
                view_name = f"{lat}{view}"  # e.g., 'LCC', 'RMLO'
                views[view_name] = row.get('image_path', '')

        return views

    def _get_cat(self, row, key):
        raw = row.get(key.lower())
        if pd.isna(raw):
            return 0
        val = str(raw).upper().strip()
        if key.lower() == 'manufacturer':
            if val.startswith('GE') or 'GENERAL ELECTRIC' in val: val = 'GE'
            elif 'HOLOGIC' in val: val = 'HOLOGIC'
            elif 'SIEMENS' in val: val = 'SIEMENS'
        return getattr(self, f"{key.lower()}_map", {}).get(val, 0)

    def _get_reg(self, row, key):
        raw = row.get(key.lower())
        if pd.isna(raw):
            return float('nan')
        try:
            return float(raw)
        except (ValueError, TypeError):
            return float('nan')

    def __len__(self):
        return len(self.exam_ids)

    def load_complete_exam_ordered(self, exam_id):
        """
        Load views in canonical order: LCC, LMLO, RCC, RMLO.

        Returns:
            tuple: (views_list, view_names) or (None, None) if not complete
        """
        view_dict = self.get_exam_views(exam_id)
        if not {'LCC', 'LMLO', 'RCC', 'RMLO'}.issubset(view_dict.keys()):
            return None, None

        canonical_order = ['LCC', 'LMLO', 'RCC', 'RMLO']
        views = []
        view_names = []

        for view_name in canonical_order:
            fn = view_dict[view_name]
            if not exists(fn):
                return None, None
            try:
                im = _load_breast_crop(fn, self.target_h, self.target_w, self.thresh_pct)
                views.append(im)
                view_names.append(view_name)
            except Exception as e:
                if fn not in _BAD_FILES:
                    log(f"Failed to load {view_name} from {fn}: {e}")
                    _BAD_FILES.add(fn)
                return None, None

        return views, view_names

    def __getitem__(self, idx):
        eid = self.exam_ids[idx]
        view_names = []

        # If require_complete, load in canonical order
        if self.require_complete:
            ordered_views, view_names = self.load_complete_exam_ordered(eid)
            if ordered_views is None:
                return None
            views = ordered_views
            num_valid = len(views)
        else:
            # Original behavior: load all views from paths
            paths = self.exam_to_paths[eid]
            views = []
            exam_rows = self.clin[self.clin['pid_acc'] == eid]
            path_to_view_name = {}
            for _, r in exam_rows.iterrows():
                lat = str(r.get('laterality', '')).strip().upper()
                view = str(r.get('view', '')).strip().upper()
                if lat in ('L', 'R') and view in ('CC', 'MLO'):
                    path_to_view_name[str(r.get('image_path', ''))] = f"{lat}{view}"
            for fn in paths:
                if not exists(fn):
                    continue
                try:
                    im = _load_breast_crop(fn, self.target_h, self.target_w, self.thresh_pct)
                    views.append(im)
                    view_names.append(path_to_view_name.get(str(fn), "LCC"))
                except Exception:
                    continue

            if len(views) == 0:
                return None
            num_valid = len(views)

        # Pad or truncate to max_views
        while len(views) < self.max_views:
            views.append(torch.zeros(1, self.target_h, self.target_w))
        views = views[:self.max_views]

        view_mask = torch.zeros(self.max_views, dtype=torch.bool)
        view_mask[:num_valid] = True
        view_ids = torch.zeros(self.max_views, dtype=torch.long)
        for i, vn in enumerate(view_names[:self.max_views]):
            view_ids[i] = _VIEW_TYPE_TO_ID.get(vn, 0)

        views = torch.stack(views)  # (max_views, 1, H, W)

        # Exam-level labels (from first row - they're all the same for an exam)
        exam_rows = self.clin[self.clin['pid_acc'] == eid]
        row = exam_rows.iloc[0]

        # BC status
        bc_raw = str(row.get('bc', '')).upper()
        bc = 1 if bc_raw == 'BC' else 0

        months_to_dx = row.get('months_to_dx', np.nan)
        try:
            months_to_dx = float(months_to_dx)
        except (ValueError, TypeError):
            months_to_dx = np.nan

        try:
            followup_months = float(row.get('followup_months', np.nan))
        except (ValueError, TypeError):
            followup_months = np.nan

        risk_targets, risk_mask = _make_risk_targets(bc, months_to_dx, followup_months)
        risk_targets = torch.from_numpy(risk_targets)
        risk_mask = torch.from_numpy(risk_mask)

        # cat_keys/reg_keys are already filtered to exam-level properties by pipelines.py
        cats = torch.tensor([self._get_cat(row, k) for k in self.cat_keys], dtype=torch.long)
        regs = torch.tensor([self._get_reg(row, k) for k in self.reg_keys], dtype=torch.float32)

        return views, view_mask, view_ids, risk_targets, risk_mask, cats, regs, eid

def write_exam_splits_5fold(
    incsv: str,
    out_csv: str,
    n_splits: int = 5,
    suffix: str = ".png",
    random_state: int = 42,
):
    """
    Patient-level stratified K-fold; outputs EXAM-level CSV with:
      pid, pid_acc, bc, split  (split in [0..n_splits-1])
    """
    df = _normalize_mirai_to_native(pd.read_csv(incsv).copy())

    # Ensure pid_acc / pid
    if "pid_acc" not in df.columns:
        if {"pid", "accession"}.issubset(df.columns):
            df["pid_acc"] = df["pid"].astype(str) + "_" + df["accession"].astype(str)
        else:
            raise KeyError("Need 'pid_acc' or ('pid','accession') columns")
    if "pid" not in df.columns:
        df["pid"] = df["pid_acc"].astype(str).str.split("_").str[0]

    # Validity filters (match your loaders)
    df["mAs"] = pd.to_numeric(df["mAs"], errors="coerce")
    df["view"] = df["view"].astype(str).str.strip()
    df = df[df["mAs"].gt(0) & df["view"].ne("")].copy()
    df = df[df["image_path"].astype(str).str.endswith(suffix, na=False)].copy()

    # Exam-level bc (views → exam)
    exam_df = (
        df.groupby(["pid", "pid_acc"])["bc"]
          .apply(lambda x: 1 if x.astype(str).str.lower().eq("bc").any() else 0)
          .reset_index(name="bc")
    )

    # Patient-level labels
    patient_bc = exam_df.groupby("pid")["bc"].max().astype(int)
    patients = patient_bc.index.to_numpy()
    labels = patient_bc.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    pid_to_fold = {}
    for fold_id, (_, val_idx) in enumerate(skf.split(patients, labels)):
        for pid in patients[val_idx]:
            pid_to_fold[pid] = fold_id

    exam_df["split"] = exam_df["pid"].map(pid_to_fold).astype(int)

    out_csv = str(out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    exam_df[["pid", "pid_acc", "bc", "split"]].to_csv(out_csv, index=False)

    # Summary
    counts = exam_df.groupby("split")["pid_acc"].nunique()
    cancer_counts = exam_df[exam_df["bc"] == 1].groupby("split")["pid_acc"].nunique()
    log("=== 5-fold exam split summary (unique exams) ===")
    for k in range(n_splits):
        n_ex = int(counts.get(k, 0))
        n_ca = int(cancer_counts.get(k, 0))
        log(f"fold {k}: exams={n_ex}, cancer_exams={n_ca} ({(100*n_ca/max(n_ex,1)):.1f}%)")
    log(f"Wrote: {out_csv}")

    return exam_df

def train_val_exams_from_splits(
    splits_csv: str,
    val_fold: int = 0,
    n_splits: int = 5,
    results_dir=None,
):
    """Read exam_splits CSV and return (train_exams, val_exams)."""
    spl = pd.read_csv(splits_csv).copy()
    if not {"pid", "pid_acc", "bc", "split"}.issubset(spl.columns):
        raise KeyError(f"{splits_csv} must contain pid,pid_acc,bc,split")

    if val_fold < 0 or val_fold >= n_splits:
        raise ValueError(f"val_fold must be in [0, {n_splits-1}]")

    val_exams = spl[spl["split"] == val_fold]["pid_acc"].unique()
    train_exams = spl[spl["split"] != val_fold]["pid_acc"].unique()

    # Sanity: no patient overlap
    train_p = set(spl[spl["pid_acc"].isin(train_exams)]["pid"].unique())
    val_p = set(spl[spl["pid_acc"].isin(val_exams)]["pid"].unique())
    overlap = train_p & val_p
    if overlap:
        raise ValueError(f"PATIENT LEAKAGE via splits_csv: {len(overlap)} patients overlap")

    log(f"\n=== SPLIT (from CSV) ===")
    log(f"val_fold={val_fold} (split=={val_fold})")
    log(f"Train exams: {len(train_exams)}  | Val exams: {len(val_exams)}")
    log(f"Train patients: {len(train_p)}   | Val patients: {len(val_p)}")
    log(f"========================\n")

    if results_dir is not None:
        train_spl = spl[spl["split"] != val_fold]
        val_spl   = spl[spl["split"] == val_fold]
        _write_dataset_stats(results_dir, "risk", {
            "total_exams":    int(spl["pid_acc"].nunique()),
            "total_patients": int(spl["pid"].nunique()),
            "train_exams":    int(len(train_exams)),
            "val_exams":      int(len(val_exams)),
            "train_patients": int(len(train_p)),
            "val_patients":   int(len(val_p)),
            "train_bc_exams": int(train_spl[train_spl["bc"] == 1]["pid_acc"].nunique()),
            "val_bc_exams":   int(val_spl[val_spl["bc"] == 1]["pid_acc"].nunique()),
            "val_fold":       val_fold,
        })

    return train_exams, val_exams

def exam_collate(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    views = torch.stack([x[0] for x in batch])          # (B, V, 1, H, W)
    view_masks = torch.stack([x[1] for x in batch])     # (B, V)
    view_ids = torch.stack([x[2] for x in batch])       # (B, V)
    risk_targets = torch.stack([x[3] for x in batch])   # (B, H)
    risk_masks = torch.stack([x[4] for x in batch])     # (B, H)
    cats = torch.stack([x[5] for x in batch])           # (B, n_cat)
    regs = torch.stack([x[6] for x in batch])           # (B, n_reg)
    eids = [x[7] for x in batch]

    return views, view_masks, view_ids, risk_targets, risk_masks, cats, regs, eids

def safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

"""Training pipeline orchestrators and external evaluation."""

import csv
import math
import numpy as np
import pandas as pd
import yaml
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from checkpoint import save_ckpt, load_ckpt
from models import build_model
from datasets import (
    train_test_load, train_test_load_exams,
    ds2024, ExamDataset, exam_collate, safe_collate,
    _normalize_mirai_to_native,
)
def hazard_to_cumulative_risk(logits):
    """hazard_t = sigmoid(logit_t); cumulative_risk = 1 - cumprod(1 - hazard)."""
    hazards = torch.sigmoid(logits.clamp(-20, 20))
    return 1 - torch.cumprod(1 - hazards, dim=1)


def _standardize(x):
    return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)


def param_groups(model, base_lr, wd_backbone=1e-4, wd_heads=1e-4, lr_mult_heads=3.0):
    """Separate param groups for backbone vs heads with independent LR/WD.

    Group order is fixed: [decay_backbone, nodecay_backbone, decay_heads, nodecay_heads].
    _set_backbone_lr() relies on backbone always being groups 0 and 1.
    """
    decay_backbone, nodecay_backbone = [], []
    decay_heads,    nodecay_heads    = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head    = n.startswith("heads.")
        is_nodecay = (p.ndim == 1) or n.endswith(".bias")
        if is_head and is_nodecay: nodecay_heads.append(p)
        elif is_head:              decay_heads.append(p)
        elif is_nodecay:           nodecay_backbone.append(p)
        else:                      decay_backbone.append(p)
    return [
        {"params": decay_backbone,   "lr": base_lr,               "weight_decay": wd_backbone},
        {"params": nodecay_backbone, "lr": base_lr,               "weight_decay": 0.0},
        {"params": decay_heads,      "lr": base_lr*lr_mult_heads, "weight_decay": wd_heads},
        {"params": nodecay_heads,    "lr": base_lr*lr_mult_heads, "weight_decay": 0.0},
    ]


def _set_backbone_lr(optimizer, lr):
    """Set LR for backbone param groups (always groups 0 and 1, per param_groups())."""
    for g in optimizer.param_groups[:2]:
        g['lr'] = lr


def compute_aux_losses(outputs, cats, regs, *, cat_specs, reg_specs,
                       cat_idx_map, reg_idx_map, reg_ranges, device):
    total = torch.tensor(0.0, device=device)
    metrics = {}
    for sp in cat_specs:
        idx = cat_idx_map[sp.key]
        y_raw = cats[:, idx]
        valid = (y_raw > 0)
        if valid.any():
            logits = outputs[sp.name][valid]
            y = (y_raw[valid] - 1).long().to(device)
            loss = F.cross_entropy(logits, y) * sp.weight
            total = total + loss
            metrics[f"acc_{sp.name}"] = (logits.argmax(1) == y).float().mean().item()
    for sp in reg_specs:
        idx = reg_idx_map[sp.key]
        y_phys = regs[:, idx].to(device)
        valid = torch.isfinite(y_phys)
        if valid.any():
            low, high = reg_ranges[sp.key.lower()]
            rng = max(high - low, 1e-6)
            y_norm = torch.clamp((y_phys[valid] - low) / rng, 0, 1)
            pred_norm = torch.sigmoid(outputs[sp.name][valid].squeeze(1))
            total = total + F.mse_loss(pred_norm, y_norm) * sp.weight
    return total, metrics


def eval_aux(model, loader, device, cat_specs, reg_specs, cat_idx_map, reg_idx_map, reg_ranges,
             debug_frac=1.0):
    limit = max(1, int(len(loader) * debug_frac))
    model.eval()
    correct        = {sp.name: 0   for sp in cat_specs}
    total_cat      = {sp.name: 0   for sp in cat_specs}
    squared_errors = {sp.name: 0.0 for sp in reg_specs}
    reg_total      = {sp.name: 0   for sp in reg_specs}
    with torch.no_grad():
        for _i, batch in enumerate(tqdm(loader, desc="Val-Aux", total=limit)):
            if _i >= limit:
                break
            if batch is None:
                continue
            imgs, _, cats, regs, _, _ = batch
            imgs = _standardize(imgs.to(device))
            outs = model(imgs)
            for sp in cat_specs:
                y_raw = cats[:, cat_idx_map[sp.key]]
                valid = (y_raw > 0)
                if valid.any():
                    pred = outs[sp.name][valid].argmax(1).cpu()
                    correct[sp.name]   += (pred == y_raw[valid] - 1).sum().item()
                    total_cat[sp.name] += valid.sum().item()
            for sp in reg_specs:
                y_raw = regs[:, reg_idx_map[sp.key]]
                valid = ~torch.isnan(y_raw)
                if valid.any():
                    pred = outs[sp.name][valid].squeeze().cpu()
                    squared_errors[sp.name] += ((pred - y_raw[valid].cpu()) ** 2).sum().item()
                    reg_total[sp.name]      += valid.sum().item()
    metrics = {}
    for sp in cat_specs:
        if total_cat[sp.name] > 0:
            metrics[f"acc_{sp.name}"] = correct[sp.name] / total_cat[sp.name]
    for sp in reg_specs:
        if reg_total[sp.name] > 0:
            metrics[f"mse_{sp.name}"] = squared_errors[sp.name] / reg_total[sp.name]
    return metrics


@torch.no_grad()
def eval_one_epoch_exam(model, loader, device, log_path=None, debug_frac=1.0):
    limit = max(1, int(len(loader) * debug_frac))
    model.eval()

    all_probs, all_targets, all_masks, all_eids = [], [], [], []

    for _i, batch in enumerate(tqdm(loader, desc="Val", total=limit)):
        if _i >= limit:
            break
        if batch is None:
            continue

        views, view_masks, view_ids, risk_targets, risk_masks, _, _, eids = batch
        views = views.to(device)
        view_masks = view_masks.to(device)
        view_ids = view_ids.to(device)

        outs = model(views, view_masks, view_ids=view_ids)
        logits = list(outs.values())[0] if len(outs) == 1 else outs['risk']
        probs = hazard_to_cumulative_risk(logits).cpu()

        all_probs.append(probs)
        all_targets.append(risk_targets)
        all_masks.append(risk_masks)
        all_eids.extend(eids)

    all_probs   = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    all_masks   = torch.cat(all_masks)

    metrics = {}
    aucs = []
    for h in range(all_probs.shape[1]):
        mask_h = all_masks[:, h].numpy()
        if mask_h.sum() < 10:
            aucs.append(float('nan'))
            continue
        y_true = all_targets[mask_h, h].numpy()
        y_prob = all_probs[mask_h, h].numpy()
        if len(np.unique(y_true)) < 2:
            aucs.append(float('nan'))
            continue
        auc = roc_auc_score(y_true, y_prob)
        aucs.append(auc)
        metrics[f'auc_h{h+1}yr'] = auc

    metrics['auc_mean'] = float(np.nanmean(aucs))
    metrics['loss'] = F.binary_cross_entropy(
        all_probs[all_masks], all_targets[all_masks]
    ).item()

    if log_path is not None:
        probs_np   = all_probs.numpy()
        targets_np = all_targets.numpy()
        masks_np   = all_masks.numpy()
        rows = []
        for i, eid in enumerate(all_eids):
            row = {'exam_id': eid}
            for h in range(probs_np.shape[1]):
                yr = h + 1
                row[f'risk_h{yr}yr_prob']  = float(probs_np[i, h])
                row[f'risk_h{yr}yr_true']  = float(targets_np[i, h]) if masks_np[i, h] else np.nan
                row[f'risk_h{yr}yr_valid'] = bool(masks_np[i, h])
            rows.append(row)
        df = pd.DataFrame(rows)
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        out_csv = lp.with_suffix(".csv")
        out_parquet = lp.with_suffix(".parquet")
        # CSV is the required artifact used by downstream reporting; parquet is optional.
        df.to_csv(out_csv, index=False)
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"[val-log] wrote {len(df)} exam predictions to {out_csv} and {out_parquet}")
        except Exception as exc:
            print(f"[val-log] wrote {len(df)} exam predictions to {out_csv} (parquet skipped: {exc})")

    return metrics


def _curriculum_weights(ep, pretrain_epochs, taper_epochs, aux_weight_final):
    """Return (aux_weight, risk_weight) for the current epoch."""
    if ep < pretrain_epochs:
        return 1.0, 0.0
    elif ep < pretrain_epochs + taper_epochs:
        progress = (ep - pretrain_epochs) / taper_epochs
        return 1.0 - (1.0 - aux_weight_final) * progress, progress
    else:
        return aux_weight_final, 1.0


def train_one_epoch_aux(model, loader, optimizer, device, *,
                        cat_specs, reg_specs, cat_idx_map, reg_idx_map,
                        reg_ranges, aux_weight, debug_frac=1.0):
    """Image-level auxiliary training (per-view cats + regs).

    Returns:
        avg weighted aux loss for the epoch.
    """
    limit = max(1, int(len(loader) * debug_frac))
    model.train()
    loss_total, n = 0.0, 0
    for _i, batch in enumerate(tqdm(loader, desc="Aux", total=limit)):
        if _i >= limit:
            break
        if batch is None:
            continue
        imgs, _, cats, regs, _, _ = batch
        imgs = _standardize(imgs.to(device))
        outs = model(imgs)
        loss, _ = compute_aux_losses(
            outs, cats, regs,
            cat_specs=cat_specs, reg_specs=reg_specs,
            cat_idx_map=cat_idx_map, reg_idx_map=reg_idx_map,
            reg_ranges=reg_ranges, device=device,
        )
        (loss * aux_weight).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item() * aux_weight * imgs.size(0)
        n          += imgs.size(0)
    return loss_total / max(n, 1)


def train_one_epoch_risk(model, loader, optimizer, device, *,
                         risk_specs, exam_cat_specs, exam_reg_specs,
                         exam_cat_idx_map, exam_reg_idx_map,
                         reg_ranges, aux_weight, gradient_accumulation, debug_frac=1.0):
    """Exam-level risk training with gradient accumulation and exam-level aux.

    Returns:
        avg risk loss for the epoch.
    """
    limit = max(1, int(len(loader) * debug_frac))
    model.train()
    loss_total, n = 0.0, 0
    step_i = -1
    optimizer.zero_grad()
    for step_i, batch in enumerate(tqdm(loader, desc="Risk", total=limit)):
        if step_i >= limit:
            break
        if batch is None:
            continue
        views, view_masks, view_ids, risk_targets, risk_masks, cats, regs, _ = batch
        if not risk_masks.any():
            continue

        views        = _standardize(views.to(device))
        view_masks   = view_masks.to(device)
        view_ids     = view_ids.to(device)
        risk_targets = risk_targets.to(device)
        risk_masks   = risk_masks.to(device)

        outs      = model(views, view_masks, view_ids=view_ids)
        logits    = outs[risk_specs[0].name]
        cum_risk  = hazard_to_cumulative_risk(logits)
        risk_loss = F.binary_cross_entropy(cum_risk[risk_masks], risk_targets[risk_masks])

        aux_loss_exam, _ = compute_aux_losses(
            outs, cats, regs,
            cat_specs=exam_cat_specs, reg_specs=exam_reg_specs,
            cat_idx_map=exam_cat_idx_map, reg_idx_map=exam_reg_idx_map,
            reg_ranges=reg_ranges, device=device,
        )
        loss = (risk_loss + aux_loss_exam * aux_weight) / gradient_accumulation
        loss.backward()

        if (step_i + 1) % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_total += loss.item() * gradient_accumulation * views.size(0)
        n          += views.size(0)

    # flush any partial accumulation at end of loop
    if step_i >= 0 and (step_i + 1) % gradient_accumulation != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer.zero_grad()

    return loss_total / max(n, 1)


def run_train_staged(ds_csv_path, epochs=100, batch_size_img=4, batch_size_exam=2,
                     lr=3e-4, pretrain_epochs=5, taper_epochs=10,
                     aux_weight_final=0.25, lr_phase3_mult=0.2, lr_min=1e-5,
                     freeze_backbone_epochs=0,
                     gpus=[0], backbone="effv2s", num_workers=4,
                     target_h=882, target_w=512,
                     outpath=".", results_dir=None, resume_path=None,
                     cat_specs=None, reg_specs=None, risk_specs=None, require_complete=True,
                     gradient_accumulation=2, run_name=None, ckpt_max_yr=4,
                     lr_mult_heads=3.0, wd_backbone=1e-4, wd_heads=1e-4,
                     debug_frac=1.0):
    """Three-phase staged training:
    Phase 1 (0..pretrain_epochs):             aux tasks only — backbone pretraining.
    Phase 2 (pretrain..pretrain+taper):       linear taper from aux-only to risk-only.
    Phase 3 (pretrain+taper..epochs):         risk dominates; LR stepped down then cosine decayed.

    Gradient accumulation: for exam batches, accumulates gradients over N micro-steps
    so effective batch = batch_size_exam * gradient_accumulation without the memory cost.
    """
    cat_specs  = cat_specs  or []
    reg_specs  = reg_specs  or []
    risk_specs = risk_specs or []

    # Results dir — resolve before anything else so splits land in the right folder
    if resume_path:
        rp = Path(resume_path)
        results_dir = rp.parent if rp.is_file() else rp
    else:
        run_tag     = datetime.now().strftime("%m%d%H%M")
        folder_name = f"{run_tag}_{run_name}" if run_name else run_tag
        results_dir = (Path(results_dir) / folder_name if results_dir
                       else Path(outpath) / "results_staged" / folder_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save run config for reproducibility / eval re-use
    run_cfg = dict(
        run_name=run_name, ds_csv_path=ds_csv_path, backbone=backbone,
        epochs=epochs, lr=lr, pretrain_epochs=pretrain_epochs, taper_epochs=taper_epochs,
        aux_weight_final=aux_weight_final, lr_phase3_mult=lr_phase3_mult, lr_min=lr_min,
        freeze_backbone_epochs=freeze_backbone_epochs,
        lr_mult_heads=lr_mult_heads, wd_backbone=wd_backbone, wd_heads=wd_heads,
        ckpt_max_yr=ckpt_max_yr,
        batch_size_img=batch_size_img, batch_size_exam=batch_size_exam,
        gradient_accumulation=gradient_accumulation, gpus=gpus,
        require_complete=require_complete,
        cat_specs=[s.name for s in cat_specs],
        reg_specs=[s.name for s in reg_specs],
        risk_specs=[dict(name=s.name, horizons=s.horizons, weight=s.weight) for s in risk_specs],
    )
    (results_dir / "run_config.yaml").write_text(yaml.dump(run_cfg, default_flow_style=False))

    # Paths
    head_tag     = "-".join([s.name for s in cat_specs + reg_specs + risk_specs]) or "model"
    ckpt_last    = results_dir / "last.pth"
    ckpt_best    = results_dir / f"{head_tag}_{backbone}_best.pth"
    csv_path     = results_dir / "metrics.csv"
    best_preds_log = results_dir / "val_preds_best"

    # Data splits
    img_train, img_val = train_test_load(ds_csv_path, results_dir=str(results_dir))
    exam_train, exam_val = train_test_load_exams(ds_csv_path, results_dir=str(results_dir))

    # Image-level datasets (aux tasks)
    train_ds_img = ds2024(img_train, ds_csv_path, train=True,
                          target_h=target_h, target_w=target_w,
                          split="train", results_dir=str(results_dir))
    val_ds_img   = ds2024(img_val,   ds_csv_path, train=False,
                          target_h=target_h, target_w=target_w,
                          split="val", results_dir=str(results_dir))
    # Image-level filtered specs — exclude exam/outcome labels (bc, months_to_dx)
    # so the cats/regs tensor dimensions match these specs exactly.
    img_cat_specs   = [s for s in cat_specs if s.key.lower() != 'bc']
    img_reg_specs   = [s for s in reg_specs  if s.key.lower() != 'months_to_dx']
    img_cat_idx_map = {s.key: i for i, s in enumerate(img_cat_specs)}
    img_reg_idx_map = {s.key: i for i, s in enumerate(img_reg_specs)}
    for ds in (train_ds_img, val_ds_img):
        ds.cat_keys = [s.key for s in img_cat_specs]
        ds.reg_keys = [s.key for s in img_reg_specs]

    # Exam-level datasets (risk task)
    train_ds_exam = ExamDataset(exam_train, ds_csv_path, train=True,
                                target_h=target_h, target_w=target_w,
                                require_complete=require_complete)
    val_ds_exam   = ExamDataset(exam_val,   ds_csv_path, train=False,
                                target_h=target_h, target_w=target_w,
                                require_complete=require_complete)
    # Exam-level filtered specs — only outcome/shared labels (bc, manufacturer, age, months_to_dx)
    exam_cat_specs   = [s for s in cat_specs if s.key.lower() in {'bc', 'manufacturer'}]
    exam_reg_specs   = [s for s in reg_specs  if s.key.lower() in {'age', 'months_to_dx'}]
    exam_cat_idx_map = {s.key: i for i, s in enumerate(exam_cat_specs)}
    exam_reg_idx_map = {s.key: i for i, s in enumerate(exam_reg_specs)}
    for ds in (train_ds_exam, val_ds_exam):
        ds.cat_keys = [s.key for s in exam_cat_specs]
        ds.reg_keys = [s.key for s in exam_reg_specs]

    # Loaders
    pw = num_workers > 0  # persistent_workers requires num_workers > 0
    train_dl_img  = DataLoader(train_ds_img,  batch_size=batch_size_img,  shuffle=True,
                               num_workers=num_workers, collate_fn=safe_collate,
                               persistent_workers=pw)
    val_dl_img    = DataLoader(val_ds_img,    batch_size=batch_size_img,  shuffle=False,
                               num_workers=num_workers, collate_fn=safe_collate,
                               persistent_workers=pw)
    train_dl_exam = DataLoader(train_ds_exam, batch_size=batch_size_exam, shuffle=True,
                               num_workers=num_workers, collate_fn=exam_collate,
                               persistent_workers=pw)
    val_dl_exam   = DataLoader(val_ds_exam,   batch_size=batch_size_exam, shuffle=False,
                               num_workers=num_workers, collate_fn=exam_collate,
                               persistent_workers=pw)

    # Model + optimizer
    model, device = build_model(
        backbone=backbone, in_chans=1, pretrained=True, gpus=gpus,
        cat_specs=cat_specs, reg_specs=reg_specs, risk_specs=risk_specs
    )
    base_lr   = 3e-4 if "v2s" in backbone else lr
    optimizer = torch.optim.AdamW(
        param_groups(model, base_lr, wd_backbone=wd_backbone,
                     wd_heads=wd_heads, lr_mult_heads=lr_mult_heads),
        lr=base_lr)

    # Resume
    start_epoch, best_auc = 0, -1.0
    if resume_path:
        rp = Path(resume_path)
        if rp.is_dir():
            rp = rp / "last.pth"
        print(f"[resume] {rp}")
        start_epoch, best_val = load_ckpt(rp, model, optimizer, map_location=device, strict=True)
        best_auc = -best_val if best_val != math.inf else -1.0

    reg_ranges = train_ds_img.reg_ranges

    # Backbone freeze: initialize state based on start_epoch (handles resume correctly)
    _freeze_end = pretrain_epochs + freeze_backbone_epochs
    backbone_frozen = freeze_backbone_epochs > 0 and pretrain_epochs <= start_epoch < _freeze_end
    if backbone_frozen:
        _set_backbone_lr(optimizer, 0.0)
        print(f"  [Backbone frozen] resuming in freeze window (epochs {pretrain_epochs}..{_freeze_end - 1})")

    # CSV fields
    fields = (["epoch", "aux_weight", "risk_weight", "train_aux_loss", "train_risk_loss",
               "val_auc_mean", "val_auc_ckpt"]
              + [f"val_auc_h{h+1}yr" for h in range(5)]
              + [f"val_acc_{s.name}" for s in cat_specs])
    write_header = not csv_path.exists()

    phase3_lr_set = False
    scheduler     = None

    with open(csv_path, "a", newline="", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if write_header:
            w.writeheader()

        for ep in range(start_epoch, epochs):
            aux_weight, risk_weight = _curriculum_weights(
                ep, pretrain_epochs, taper_epochs, aux_weight_final)

            # Backbone freeze window: first freeze_backbone_epochs of Phase 2
            now_frozen = freeze_backbone_epochs > 0 and pretrain_epochs <= ep < _freeze_end
            if now_frozen and not backbone_frozen:
                _set_backbone_lr(optimizer, 0.0)
                backbone_frozen = True
                print(f"  [Backbone frozen] epochs {pretrain_epochs}..{_freeze_end - 1} "
                      f"(heads-only training)")
            elif not now_frozen and backbone_frozen:
                _set_backbone_lr(optimizer, base_lr)
                backbone_frozen = False
                print(f"  [Backbone unfrozen] at epoch {ep}, backbone LR → {base_lr:.2e}")

            # Phase 3: one-time LR step-down + cosine annealing
            if risk_weight == 1.0 and not phase3_lr_set:
                for g in optimizer.param_groups:
                    g['lr'] *= lr_phase3_mult
                phase3_epochs = max(epochs - ep, 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=phase3_epochs, eta_min=lr_min)
                phase3_lr_set = True
                print(f"  [Phase 3] LR → {optimizer.param_groups[0]['lr']:.2e}, "
                      f"cosine over {phase3_epochs} epochs to {lr_min:.1e}")

            print(f"\nEpoch {ep}: aux={aux_weight:.2f}  risk={risk_weight:.2f}")

            # --- Aux training (image-level) ---
            avg_aux_loss = 0.0
            if aux_weight > 0 and (cat_specs or reg_specs):
                avg_aux_loss = train_one_epoch_aux(
                    model, train_dl_img, optimizer, device,
                    cat_specs=img_cat_specs, reg_specs=img_reg_specs,
                    cat_idx_map=img_cat_idx_map, reg_idx_map=img_reg_idx_map,
                    reg_ranges=reg_ranges, aux_weight=aux_weight,
                    debug_frac=debug_frac,
                )

            # --- Risk training (exam-level) ---
            torch.cuda.empty_cache()
            avg_risk_loss = 0.0
            if risk_weight > 0 and risk_specs:
                avg_risk_loss = train_one_epoch_risk(
                    model, train_dl_exam, optimizer, device,
                    risk_specs=risk_specs,
                    exam_cat_specs=exam_cat_specs, exam_reg_specs=exam_reg_specs,
                    exam_cat_idx_map=exam_cat_idx_map, exam_reg_idx_map=exam_reg_idx_map,
                    reg_ranges=reg_ranges, aux_weight=aux_weight,
                    gradient_accumulation=gradient_accumulation,
                    debug_frac=debug_frac,
                )

            # --- Validation ---
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            aux_metrics  = (eval_aux(model, val_dl_img, device,
                                     img_cat_specs, img_reg_specs,
                                     img_cat_idx_map, img_reg_idx_map, reg_ranges,
                                     debug_frac=debug_frac)
                            if aux_weight > 0 and img_cat_specs else {})
            risk_metrics = (eval_one_epoch_exam(model, val_dl_exam, device,
                                                debug_frac=debug_frac)
                            if risk_specs else {'auc_mean': float('nan')})

            # Checkpoint AUC: mean over horizons 1..ckpt_max_yr (excludes noisy 5yr at small N)
            ckpt_auc = float(np.nanmean([risk_metrics.get(f'auc_h{h+1}yr', float('nan'))
                                         for h in range(ckpt_max_yr)]))

            row = {
                "epoch":           ep,
                "aux_weight":      f"{aux_weight:.2f}",
                "risk_weight":     f"{risk_weight:.2f}",
                "train_aux_loss":  f"{avg_aux_loss:.4f}",
                "train_risk_loss": f"{avg_risk_loss:.4f}",
                "val_auc_mean":    f"{risk_metrics['auc_mean']:.4f}",
                "val_auc_ckpt":    f"{ckpt_auc:.4f}",
            }
            for h in range(5):
                row[f"val_auc_h{h+1}yr"] = f"{risk_metrics.get(f'auc_h{h+1}yr', float('nan')):.4f}"
            for sp in cat_specs:
                row[f"val_acc_{sp.name}"] = f"{aux_metrics.get(f'acc_{sp.name}', float('nan')):.4f}"

            print(row)
            w.writerow(row)

            if scheduler is not None:
                scheduler.step()

            save_ckpt(ckpt_last, model, optimizer, epoch=ep, best_val=best_auc)

            if ep >= pretrain_epochs and ckpt_auc > best_auc and not math.isnan(ckpt_auc):
                best_auc = ckpt_auc
                save_ckpt(ckpt_best, model, optimizer, epoch=ep, best_val=best_auc)
                print(f"  [saved best] auc_ckpt({ckpt_max_yr}yr)={best_auc:.4f} -> {ckpt_best}")
                if risk_specs:
                    _ = eval_one_epoch_exam(model, val_dl_exam, device, log_path=best_preds_log)

    # Fallback for resumed runs / interrupted best-save logging: regenerate if best checkpoint exists but val_preds_best.csv is missing.
    best_preds_csv = best_preds_log.with_suffix(".csv")
    if risk_specs and ckpt_best.exists() and not best_preds_csv.exists():
        print(f"[val-log] missing {best_preds_csv.name}; regenerating from best checkpoint {ckpt_best.name}")
        try:
            _ = load_ckpt(ckpt_best, model, optimizer, map_location=device, strict=True)
            _ = eval_one_epoch_exam(model, val_dl_exam, device, log_path=best_preds_log)
        except Exception as exc:
            print(f"[val-log] WARNING: could not regenerate {best_preds_csv.name}: {exc}")


def eval_external_mirai_dataset(checkpoint_path, csv_path, batch_size=8, num_workers=4,
                                 gpus=[0], require_complete=True, output_csv=None,
                                 no_cancer=False, run_cfg=None):
    """Evaluate a trained model on a dataset using the same loading as training.

    Args:
        checkpoint_path: Path to .pth file or directory containing *_best.pth (preferred)
                        or last.pth (fallback)
        csv_path: Path to native CSV (with pid_acc, bc, image_path, etc.)
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        gpus: List of GPU IDs
        require_complete: Only evaluate complete exams (all 4 views)
        output_csv: Optional path to save per-exam predictions
        no_cancer: If True, exclude all patients who have/develop cancer
        run_cfg: Dict with model spec (backbone, cats, regs, risk_specs, target_h, target_w).
                 If not provided, loaded from run_config.yaml alongside the checkpoint.

    Returns:
        dict: Metrics including auc_h1yr..auc_h5yr, auc_mean
    """
    # Resolve checkpoint
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        best_cands = sorted(ckpt_path.glob("*_best.pth"))
        if best_cands:
            ckpt_path = best_cands[0]
            print(f"[eval_external] using best checkpoint: {ckpt_path.name}")
        else:
            ckpt_path = ckpt_path / "last.pth"
            print(f"[eval_external] best checkpoint not found, falling back to: {ckpt_path.name}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load model spec — from caller-supplied dict or run_config.yaml alongside checkpoint
    if run_cfg is None or 'backbone' not in run_cfg:
        run_config_path = ckpt_path.parent / "run_config.yaml"
        if not run_config_path.exists():
            raise FileNotFoundError(
                f"Model spec not found: pass backbone/cats/regs/risk_specs in your eval config, "
                f"or place run_config.yaml in {ckpt_path.parent}"
            )
        run_cfg = yaml.safe_load(run_config_path.read_text())
    backbone = run_cfg.get('backbone', 'effv2s')

    # Reconstruct specs from saved config
    cat_specs  = [SimpleNamespace(name=n, key=n, num_classes=k, weight=1.0)
                  for (n, k) in run_cfg.get('cats', [])]
    reg_specs  = [SimpleNamespace(name=k.lower(), key=k, weight=1.0, metric="mse")
                  for k in run_cfg.get('regs', [])]
    risk_specs = [SimpleNamespace(name=r.get('name', 'risk'),
                                  horizons=r.get('horizons', 5),
                                  weight=r.get('weight', 1.0),
                                  key=r.get('key', 'risk'))
                  for r in run_cfg.get('risk_specs', [])]

    # Fallback: reconstruct cat/reg specs from state_dict shapes if missing in run_config
    if not cat_specs and 'cat_specs' in run_cfg:
        print("Reconstructing cat specs from state_dict shapes...")
        ckpt_tmp = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt_tmp.get('state_dict', ckpt_tmp.get('model_state_dict', ckpt_tmp))
        for name in run_cfg['cat_specs']:
            w_key = f"heads.{name}.weight"
            if w_key in sd:
                cat_specs.append(SimpleNamespace(name=name, key=name,
                                                 num_classes=sd[w_key].shape[0], weight=1.0))
        del ckpt_tmp
    if not reg_specs and 'reg_specs' in run_cfg:
        reg_specs = [SimpleNamespace(name=n, key=n, weight=1.0, metric="mse")
                     for n in run_cfg['reg_specs']]
    cat_specs, reg_specs, risk_specs = _filter_specs_to_checkpoint_heads(
        ckpt_path, cat_specs, reg_specs, risk_specs
    )

    print(f"Config: backbone={backbone}, "
          f"cats={[s.name for s in cat_specs]}, "
          f"regs={[s.name for s in reg_specs]}, "
          f"risk={[s.name for s in risk_specs]}")

    # Filter to no-cancer patients if requested
    if no_cancer:
        df = _normalize_mirai_to_native(pd.read_csv(csv_path, low_memory=False).copy())
        if 'pid' not in df.columns:
            if 'pid_acc' in df.columns:
                df['pid'] = df['pid_acc'].astype(str).str.split('_').str[0]
            else:
                raise KeyError("Need 'pid' or 'pid_acc' for patient-level filtering")
        if 'bc' not in df.columns:
            raise KeyError("Need 'bc' after CSV normalization for patient-level filtering")
        cancer_pids = df[df['bc'].astype(str).str.upper() == 'BC']['pid'].unique()
        before      = df['pid'].nunique()
        df          = df[~df['pid'].isin(cancer_pids)].copy()
        print(f"[no_cancer] {df['pid'].nunique()}/{before} patients remain")
        temp_csv      = Path(csv_path).parent / f"_temp_eval_{Path(csv_path).name}"
        df.to_csv(temp_csv, index=False)
        effective_csv = str(temp_csv)
    else:
        effective_csv = csv_path
        temp_csv      = None

    try:
        train_exams, val_exams = train_test_load_exams(
            effective_csv, make_if_missing=False,
        )
        all_exams = np.concatenate([train_exams, val_exams])
        print(f"Total exams: {len(all_exams)}")

        target_h = run_cfg.get('target_h', 882)
        target_w = run_cfg.get('target_w', 512)
        eval_ds  = ExamDataset(all_exams, effective_csv, train=False,
                               target_h=target_h, target_w=target_w,
                               require_complete=require_complete)
        eval_ds.cat_keys = [s.key for s in cat_specs]
        eval_ds.reg_keys = [s.key for s in reg_specs]
        print(f"Dataset: {len(eval_ds)} exams")

        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=exam_collate)

        model, device = build_model(
            backbone=backbone, in_chans=1, pretrained=False, gpus=gpus,
            cat_specs=cat_specs, reg_specs=reg_specs, risk_specs=risk_specs,
        )
        load_ckpt(ckpt_path, model, optimizer=None, map_location=device, strict=True)
        print("Model loaded")

        log_path = Path(output_csv).with_suffix("") if output_csv else None
        metrics  = eval_one_epoch_exam(model, eval_dl, device, log_path=log_path)

        print("\n" + "="*50)
        for h in range(5):
            key = f'auc_h{h+1}yr'
            if key in metrics:
                print(f"  {h+1}-year AUC: {metrics[key]:.4f}")
        print(f"  Mean AUC:     {metrics.get('auc_mean', float('nan')):.4f}")
        print("="*50)
        return metrics

    finally:
        if temp_csv is not None and temp_csv.exists():
            temp_csv.unlink()


def _checkpoint_state_dict(ckpt_path: Path):
    ckpt_tmp = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    try:
        return ckpt_tmp.get('state_dict', ckpt_tmp.get('model_state_dict', ckpt_tmp))
    finally:
        del ckpt_tmp


def _filter_specs_to_checkpoint_heads(ckpt_path: Path, cat_specs, reg_specs, risk_specs):
    """Drop specs whose heads are not present in checkpoint state_dict (handles stale run_config specs)."""
    sd = _checkpoint_state_dict(ckpt_path)

    def _has_head(name: str) -> bool:
        return f"heads.{name}.weight" in sd or f"module.heads.{name}.weight" in sd

    cat_specs_f = [s for s in cat_specs if _has_head(s.name)]
    reg_specs_f = [s for s in reg_specs if _has_head(s.name)]
    risk_specs_f = [s for s in risk_specs if _has_head(s.name)]

    dropped_c = [s.name for s in cat_specs if not _has_head(s.name)]
    dropped_r = [s.name for s in reg_specs if not _has_head(s.name)]
    dropped_k = [s.name for s in risk_specs if not _has_head(s.name)]
    if dropped_c or dropped_r or dropped_k:
        print("[spec-filter] dropped missing checkpoint heads:",
              f"cats={dropped_c or []} regs={dropped_r or []} risk={dropped_k or []}")

    return cat_specs_f, reg_specs_f, risk_specs_f


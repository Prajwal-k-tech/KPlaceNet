"""L5 Kerala pinpoint cells — fine-tune the L2 winner on the OSV-5M Kerala slice.

Phases:
    prep   deterministic train/val/test CSVs from extracted ids
    cells  300 k-means cells over Kerala train coords (pinpoint grid)
    train  python -m src.train with --init-backbone-from (L2 winner), layer4 unfreeze
    eval   BEFORE (global L2) vs AFTER (L5 Kerala) on val + test, per-town breakdown
    all    prep -> cells -> train -> eval

Windows-safe: pathlib + sys.executable subprocess (no shell strings).

Usage:
    python scripts/run_l5_kerala.py --phase prep
    python scripts/run_l5_kerala.py --phase cells
    python scripts/run_l5_kerala.py --phase train --epochs 10
    python scripts/run_l5_kerala.py --phase eval
    python scripts/run_l5_kerala.py --phase all
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "kerala_osv5m"
IDS_TRAIN = DATA / "ids" / "ids_train.csv"
IDS_TEST = DATA / "ids" / "ids_test.csv"
TRAIN_CSV = DATA / "kerala_train.csv"
VAL_CSV = DATA / "kerala_val.csv"
TEST_CSV = DATA / "kerala_test.csv"

CKPT_DIR = ROOT / "checkpoints" / "l5_kerala_kmeans300"
CELLS_JSON = CKPT_DIR / "cells.json"
L2_CKPT = ROOT / "checkpoints" / "l2_imagenet_1.0_layer4" / "last.pt"
RESULTS_JSON = CKPT_DIR / "results.json"

NUM_CELLS = 300
SEED = 42
VAL_SIZE = 1000
CSV_FIELDS = ["image_path", "lat", "lon", "town", "osv_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L5 Kerala pinpoint cells pipeline")
    p.add_argument("--phase", choices=["prep", "cells", "train", "eval", "all"], default="all")
    p.add_argument("--epochs", type=int, default=10, help="Train epochs (default 10)")
    p.add_argument("--batch-size", type=int, default=16, help="Train batch (layer4 regime default 16)")
    p.add_argument("--lr", type=float, default=3e-4, help="LR (warm start from L2 winner)")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers for training")
    p.add_argument("--max-samples", type=int, default=None, help="Smoke: cap train samples")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help=f"Override ckpt dir (default {CKPT_DIR})")
    p.add_argument("--l5-ckpt", type=str, default=None,
                   help="Eval: L5 checkpoint (default <ckpt-dir>/last.pt)")
    p.add_argument("--limit", type=int, default=None, help="Eval smoke: cap samples per split")
    p.add_argument("--dump-samples", action="store_true",
                   help="Eval: write per-sample prediction CSVs (samples_<split>_<label>.csv)")
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    p.add_argument("--force", action="store_true", help="Re-run train even if metrics JSON exists")
    return p.parse_args()


def ckpt_dir_for(args: argparse.Namespace) -> Path:
    return Path(args.checkpoint_dir) if args.checkpoint_dir else CKPT_DIR


# ---------------------------------------------------------------------------
# Phase: prep
# ---------------------------------------------------------------------------


def read_ids(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_kerala_csv(path: Path, rows: list[dict], split_dir: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "image_path": f"{split_dir}/images/{r['osv_id']}.jpg",
                "lat": r["lat"],
                "lon": r["lon"],
                "town": r.get("town", ""),
                "osv_id": r["osv_id"],
            })


def phase_prep() -> None:
    import numpy as np

    train_rows = read_ids(IDS_TRAIN)
    test_rows = read_ids(IDS_TEST)
    n = len(train_rows)
    if n <= VAL_SIZE:
        raise SystemExit(f"[prep] not enough train rows: {n}")

    rng = np.random.RandomState(SEED)
    val_idx = set(rng.choice(n, size=VAL_SIZE, replace=False).tolist())
    fit_rows = [r for i, r in enumerate(train_rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in val_idx]

    write_kerala_csv(TRAIN_CSV, fit_rows, "train")
    write_kerala_csv(VAL_CSV, val_rows, "train")
    write_kerala_csv(TEST_CSV, test_rows, "test")

    print(f"[prep] train rows: {len(fit_rows)} -> {TRAIN_CSV}")
    print(f"[prep] val rows:   {len(val_rows)} -> {VAL_CSV} (seeded holdout, seed={SEED})")
    print(f"[prep] test rows:  {len(test_rows)} -> {TEST_CSV}")
    for name, rows in [("train", fit_rows), ("val", val_rows), ("test", test_rows)]:
        towns: dict = {}
        for r in rows:
            t = r.get("town", "?")
            towns[t] = towns.get(t, 0) + 1
        print(f"[prep] {name} towns: {dict(sorted(towns.items(), key=lambda kv: -kv[1]))}")


# ---------------------------------------------------------------------------
# Phase: cells
# ---------------------------------------------------------------------------


def phase_cells() -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from src.cells import build_cells, cell_balance_stats, save_cells_json
    from src.dataset import GeoDataset
    from src.eval import haversine_km_vec

    if not TRAIN_CSV.exists():
        raise SystemExit("[cells] run --phase prep first")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ds = GeoDataset(str(TRAIN_CSV), image_size=224, train=True)
    coords = ds.get_coords()
    coords_np = coords.cpu().numpy() if hasattr(coords, "cpu") else np.asarray(coords)
    print(f"[cells] building {NUM_CELLS} k-means cells from {len(coords_np)} train coords ...")
    cells = build_cells(coords_np, K=NUM_CELLS, method="kmeans", seed=SEED)
    save_cells_json(cells, CELLS_JSON)
    print(f"[cells] saved {len(cells)} cells -> {CELLS_JSON}")

    cent = np.array([[c.centroid_lat, c.centroid_lon] for c in cells])
    d = haversine_km_vec(
        cent[:, 0][:, None], cent[:, 1][:, None], cent[None, :, 0], cent[None, :, 1]
    )
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    print(f"[cells] nearest-neighbour centroid distance km: p10={np.percentile(nn, 10):.2f} "
          f"median={np.median(nn):.2f} p90={np.percentile(nn, 90):.2f}")
    try:
        print(f"[cells] balance: {cell_balance_stats(cells)}")
    except Exception as e:  # pragma: no cover - reporting only
        print(f"[cells] balance stats unavailable: {e}")


# ---------------------------------------------------------------------------
# Phase: train
# ---------------------------------------------------------------------------


def phase_train(args: argparse.Namespace) -> None:
    ckpt_dir = ckpt_dir_for(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics_l5_kerala.json"
    if metrics_path.exists() and not args.force:
        print(f"[train] metrics already exist: {metrics_path} (use --force to re-run) — skipping")
        return
    if not CELLS_JSON.exists():
        raise SystemExit("[train] run --phase cells first")
    if not L2_CKPT.exists():
        raise SystemExit(f"[train] L2 winner checkpoint missing: {L2_CKPT}")

    cmd = [
        sys.executable, "-m", "src.train",
        "--csv", str(TRAIN_CSV),
        "--num-cells", str(NUM_CELLS),
        "--cells-json", str(CELLS_JSON),
        "--init-backbone-from", str(L2_CKPT),
        "--unfreeze-last-block",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num-workers", str(args.workers),
        "--seed", str(SEED),
        "--device", args.device,
        "--checkpoint-dir", str(ckpt_dir),
        "--run-tag", "l5_kerala",
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]

    print(f"[train] {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    print(f"[train] returncode={rc} in {time.time() - t0:.0f}s")
    if rc != 0:
        raise SystemExit(f"[train] FAILED (rc={rc})")


# ---------------------------------------------------------------------------
# Phase: eval (before vs after)
# ---------------------------------------------------------------------------


def _resolve_device(pref: str):
    import torch

    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _eval_model(ckpt_path: Path, csv_path: Path, limit, device, cells_path=None,
                batch_size: int = 64):
    """Predict lat/lon via argmax cell centroid; returns (metrics, per-sample dists).

    cells_path: full cells JSON (with bboxes) used for the top-1 "true cell"
    assignment. Checkpoint cells are trimmed dicts, so assign_cells (which
    needs Cell objects) uses this file instead.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(ROOT))
    from src.cells import assign_cells, load_cells_json
    from src.dataset import GeoDataset
    from src.eval import haversine_km_vec, within_km_accuracy
    from src.model import build_model

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cells = sorted(ckpt["cells"], key=lambda c: c["cell_id"])
    num_cells = len(cells)
    model = build_model(num_cells=num_cells, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    centroids = np.array([[c["centroid_lat"], c["centroid_lon"]] for c in cells], dtype=np.float64)

    cells_full = None
    if cells_path is not None and Path(cells_path).exists():
        try:
            loaded = load_cells_json(cells_path)
            if len(loaded) == num_cells:
                cells_full = loaded
        except Exception:
            cells_full = None

    ds = GeoDataset(str(csv_path), image_size=224, train=False)
    n = min(limit, len(ds)) if limit else len(ds)
    loader = DataLoader(Subset(ds, list(range(n))), batch_size=batch_size,
                        shuffle=False, num_workers=0)

    pred_ids_all, true_all = [], []
    with torch.no_grad():
        for images, true_coords in loader:
            logits = model(images.to(device))
            pred_ids_all.append(logits.argmax(dim=1).cpu().numpy())
            if isinstance(true_coords, torch.Tensor):
                true_all.append(true_coords.cpu().numpy())
            else:
                true_all.append(np.asarray(true_coords))

    pred_ids = np.clip(np.concatenate(pred_ids_all), 0, num_cells - 1)
    true = np.concatenate(true_all)
    if true.ndim == 1:
        true = true[None, :]

    pred = centroids[pred_ids]
    dists = haversine_km_vec(pred[:, 0], pred[:, 1], true[:, 0], true[:, 1])
    metrics = within_km_accuracy(pred[:, 0], pred[:, 1], true[:, 0], true[:, 1],
                                 thresholds_km=(1, 25, 200))
    metrics["p90_km"] = float(np.percentile(dists, 90))
    metrics["n"] = int(len(dists))
    metrics["cell_top1_pct"] = float("nan")
    if cells_full is not None:
        try:
            assigned = np.asarray(assign_cells(true, cells_full)).astype(int)
            metrics["cell_top1_pct"] = float((assigned == pred_ids).mean() * 100.0)
        except Exception:
            pass
    return metrics, dists, true, pred


def _per_town(df, dists) -> dict:
    import numpy as np

    out: dict = {}
    for town, idx in df.groupby("town").groups.items():
        dd = dists[list(idx)]
        out[str(town)] = {
            "n": int(len(dd)),
            "mean_km": float(dd.mean()),
            "median_km": float(np.median(dd)),
            "within_25km": float((dd <= 25).mean() * 100.0),
            "within_200km": float((dd <= 200).mean() * 100.0),
        }
    return out


def phase_eval(args: argparse.Namespace) -> None:
    import pandas as pd

    l5_ckpt = Path(args.l5_ckpt) if args.l5_ckpt else ckpt_dir_for(args) / "last.pt"
    if not l5_ckpt.exists():
        raise SystemExit(f"[eval] L5 checkpoint missing: {l5_ckpt} (run --phase train first)")
    if not L2_CKPT.exists():
        raise SystemExit(f"[eval] L2 checkpoint missing: {L2_CKPT}")

    device = _resolve_device(args.device)
    print(f"[eval] device={device} | before={L2_CKPT} | after={l5_ckpt} | limit={args.limit}")

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "l2_ckpt": str(L2_CKPT),
            "l5_ckpt": str(l5_ckpt),
            "cells_json": str(CELLS_JSON),
            "num_cells": NUM_CELLS,
            "val_size": VAL_SIZE,
            "seed": SEED,
            "limit": args.limit,
        },
        "splits": {},
    }

    models = [
        ("before", "L2 global (quad-tree cells)", L2_CKPT,
         ROOT / "checkpoints" / "l2_cells_full.json"),
        ("after", "L5 Kerala (k-means cells)", l5_ckpt, CELLS_JSON),
    ]
    for split_name, csv_path in [("val", VAL_CSV), ("test", TEST_CSV)]:
        df = pd.read_csv(csv_path)
        if args.limit:
            df = df.iloc[: args.limit]
        split_out: dict = {"n": len(df)}
        for label, desc, ckpt, cells_path in models:
            metrics, dists, true, pred = _eval_model(ckpt, csv_path, args.limit, device, cells_path)
            split_out[label] = {
                "overall": metrics,
                "per_town": _per_town(df, dists),
                "checkpoint": str(ckpt),
                "desc": desc,
            }
            if args.dump_samples:
                out = CKPT_DIR / f"samples_{split_name}_{label}.csv"
                with open(out, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(
                        fh,
                        fieldnames=["image_path", "lat", "lon", "town",
                                    "pred_lat", "pred_lon", "km"],
                    )
                    w.writeheader()
                    for i, row in enumerate(df.reset_index(drop=True).to_dict("records")):
                        w.writerow({
                            "image_path": row["image_path"],
                            "lat": row["lat"], "lon": row["lon"], "town": row["town"],
                            "pred_lat": float(pred[i, 0]), "pred_lon": float(pred[i, 1]),
                            "km": float(dists[i]),
                        })
                print(f"[eval] samples -> {out}")
            print(f"[eval] {split_name:4s} {label:6s} mean={metrics['mean_km']:8.1f} km  "
                  f"median={metrics['median_km']:7.1f}  @25km={metrics['within_25km']:5.1f}%  "
                  f"@200km={metrics['within_200km']:5.1f}%  top1={metrics['cell_top1_pct']}")
        results["splits"][split_name] = split_out

    if args.limit is None:
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] results -> {RESULTS_JSON}")
    else:
        print("[eval] --limit set (smoke); results NOT written")


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    phases = ["prep", "cells", "train", "eval"] if args.phase == "all" else [args.phase]
    for ph in phases:
        print(f"\n{'=' * 60}\n[phase] {ph}\n{'=' * 60}")
        if ph == "prep":
            phase_prep()
        elif ph == "cells":
            phase_cells()
        elif ph == "train":
            phase_train(args)
        elif ph == "eval":
            phase_eval(args)


if __name__ == "__main__":
    main()

"""Eval — haversine within 1/25/200km (Section 5 primary metric).

No new metrics without re-plan. L3 adds ECE/coverage, L4 adds cell balance.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.dataset import GeoDataset
    from src.eval_contract import validate_eval_checkpoint
    from src.model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import GeoDataset
    from src.eval_contract import validate_eval_checkpoint
    from src.model import build_model


# ---------------------------------------------------------------------------
# Haversine — great-circle distance in km
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two (lat,lon) points in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def haversine_km_vec(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine for N pairs."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def within_km_accuracy(
    pred_lats: np.ndarray,
    pred_lons: np.ndarray,
    true_lats: np.ndarray,
    true_lons: np.ndarray,
    thresholds_km: Tuple[int, ...] = (1, 25, 200),
) -> Dict[str, float]:
    """Compute within-X-km accuracy.

    Returns:
        dict with keys like "within_1km", "within_25km", "within_200km" (0-100 %)
        plus "mean_km" and "median_km".
    """
    dists = haversine_km_vec(pred_lats, pred_lons, true_lats, true_lons)
    out: Dict[str, float] = {}
    for thr in thresholds_km:
        out[f"within_{thr}km"] = float((dists <= thr).mean() * 100)
    out["mean_km"] = float(dists.mean()) if len(dists) else float("nan")
    out["median_km"] = float(np.median(dists)) if len(dists) else float("nan")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KPlaceNet eval — within 1/25/200km via haversine")
    p.add_argument("--csv", type=str, required=True, help="Eval CSV (image_path,lat,lon) — IM2GPS3k or OSV-5M test split")
    p.add_argument("--checkpoint", type=str, default="checkpoints/last.pt", help="Path to checkpoint .pt")
    p.add_argument("--num-cells", type=int, default=300, help="Must match training num_cells")
    p.add_argument("--batch-size", type=int, default=32, help="Eval batch size")
    p.add_argument("--image-size", type=int, default=224, help="Must match training image_size")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows)")
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    return p.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[eval] device={device} | csv={args.csv} | ckpt={args.checkpoint}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Evaluation checkpoint not found: {ckpt_path}. "
            "Train a model first; evaluation will not use random weights."
        )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    num_cells, cells = validate_eval_checkpoint(ckpt)
    print(f"[eval] loaded checkpoint epoch={ckpt.get('epoch', '?')} loss={ckpt.get('loss', '?')}")
    if num_cells != args.num_cells:
        print(f"[eval] using checkpoint num_cells={num_cells} (CLI was {args.num_cells})")

    # Load eval dataset (no cell_ids — we predict them)
    ds = GeoDataset(args.csv, image_size=args.image_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_cells=num_cells, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    model.to(device)
    model.eval()

    # Class ids and geographic centroids must come from the same checkpoint.
    centroids = np.array(
        [[cell["centroid_lat"], cell["centroid_lon"]] for cell in cells],
        dtype=np.float64,
    )

    # Inference
    all_pred_lats: List[float] = []
    all_pred_lons: List[float] = []
    all_true_lats: List[float] = []
    all_true_lons: List[float] = []

    for images, true_coords in tqdm(loader, desc="Eval", unit="batch"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred_ids = logits.argmax(dim=1).cpu().numpy()

        # Map cell id → centroid lat/lon
        pred_lats = centroids[pred_ids, 0]
        pred_lons = centroids[pred_ids, 1]

        # true_coords is (B,2) tensor of (lat,lon) from dataset
        if isinstance(true_coords, torch.Tensor):
            true_np = true_coords.cpu().numpy()
        else:
            true_np = np.asarray(true_coords)

        if true_np.ndim == 1:
            true_np = true_np[None, :]

        all_pred_lats.extend(pred_lats.tolist())
        all_pred_lons.extend(pred_lons.tolist())
        all_true_lats.extend(true_np[:, 0].tolist())
        all_true_lons.extend(true_np[:, 1].tolist())

    # Metrics
    metrics = within_km_accuracy(
        np.array(all_pred_lats),
        np.array(all_pred_lons),
        np.array(all_true_lats),
        np.array(all_true_lons),
        thresholds_km=(1, 25, 200),
    )

    # Print table — Section 5 locked metrics
    print("\n" + "=" * 52)
    print(" KPlaceNet Eval — within X km (haversine)")
    print("=" * 52)
    print(f" Samples: {len(all_pred_lats)}")
    print(f" within   1km : {metrics['within_1km']:6.2f}%")
    print(f" within  25km : {metrics['within_25km']:6.2f}%")
    print(f" within 200km : {metrics['within_200km']:6.2f}%")
    print(f" mean distance: {metrics['mean_km']:.1f} km")
    print(f" median distance: {metrics['median_km']:.1f} km")
    print("=" * 52 + "\n")

    # Also emit JSON for notebook parsing
    import json

    print("[json]" + json.dumps({**metrics, "n": len(all_pred_lats)}))


if __name__ == "__main__":
    main()

"""KPlaceNet interactive demo (single file).

Set KPLACENET_CHECKPOINT and optionally KPLACENET_CALIBRATION to select
artifacts. Model loading is lazy; inference returns a coarse cell estimate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT = Path(os.environ.get(
    "KPLACENET_CHECKPOINT",
    str(ROOT / "checkpoints" / "l2_imagenet_1.0_layer4" / "last.pt"),
)).expanduser()
CALIB_JSON = Path(os.environ.get(
    "KPLACENET_CALIBRATION",
    str(ROOT / "checkpoints" / "l3_uncertainty_results.json"),
)).expanduser()
SIDECAR_CELLS = ROOT / "checkpoints" / "l2_cells_full.json"  # full boxes (same centroids)
ABSTAIN_THRESHOLD = 0.5
IMAGE_SIZE = 224

_CACHE: dict = {}


def _load_calibration():
    """Load calibration explicitly; a missing file means uncalibrated mode."""
    from src.demo_config import load_calibration

    return load_calibration(CALIB_JSON)


def _load_resources():
    """Lazy, cached model + cells + calibration. Returns dict."""
    if _CACHE:
        return _CACHE
    import torch
    from src.cells import Cell
    from src.eval_contract import validate_eval_checkpoint
    from src.model import build_model

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Demo checkpoint not found: {CHECKPOINT}. Train a model first; "
            "the demo will not run with random weights."
        )
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu")
    num_cells, cells_raw = validate_eval_checkpoint(ckpt)
    # Same pattern as scripts/run_l3_uncertainty.py (ckpt cells carry centroids only).
    cells = [
        Cell(
            cell_id=int(c["cell_id"]),
            lat_min=0, lat_max=0, lon_min=0, lon_max=0,
            centroid_lat=float(c["centroid_lat"]),
            centroid_lon=float(c["centroid_lon"]),
        )
        for c in sorted(cells_raw, key=lambda x: x["cell_id"])
    ]
    # Enrich degenerate (zeroed) boxes from the sidecar full-cell JSON so the
    # map rectangle is honest. Same centroids, adds real bounding boxes.
    try:
        if SIDECAR_CELLS.exists() and all(
            c.lat_min == 0 and c.lat_max == 0 for c in cells
        ):
            side = {int(c["cell_id"]): c for c in json.loads(
                SIDECAR_CELLS.read_text(encoding="utf-8"))["cells"]}
            for c in cells:
                s = side.get(c.cell_id)
                if s:
                    c.lat_min, c.lat_max = float(s["lat_min"]), float(s["lat_max"])
                    c.lon_min, c.lon_max = float(s["lon_min"]), float(s["lon_max"])
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_cells=num_cells, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    from src.demo_config import sha256_file, verify_calibration_checkpoint

    calibration = _load_calibration()
    checkpoint_matches = verify_calibration_checkpoint(
        calibration, sha256_file(CHECKPOINT)
    )
    _CACHE.update(model=model, cells=cells,
                  T=calibration.temperature,
                  temperature_fitted=calibration.temperature_fitted,
                  quantile=calibration.conformal_quantile,
                  calibration_matches_checkpoint=checkpoint_matches,
                  device=device, K=num_cells)
    return _CACHE


def render_map(lat: float, lon: float, cell) -> "plt.Figure":
    """Panel A: global equirectangular + Panel B: +/-5 deg zoom."""
    plt.close("all")  # avoid figure accumulation across repeated calls
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, title, x0, x1, y0, y1 in [
        (axes[0], "Global prediction", -180, 180, -90, 90),
        (axes[1], "Zoom (+/-5 deg)", max(-180, lon - 5), min(180, lon + 5),
         max(-90, lat - 5), min(90, lat + 5)),
    ]:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(title)
        ax.set_xticks(range(-180, 181, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.grid(True, linewidth=0.4, alpha=0.7)
        # Cell bounding box (skip if degenerate zero box).
        if not (cell.lat_min == 0 and cell.lat_max == 0
                and cell.lon_min == 0 and cell.lon_max == 0):
            w = cell.lon_max - cell.lon_min
            h = cell.lat_max - cell.lat_min
            if w > 0 and h > 0:
                ax.add_patch(Rectangle((cell.lon_min, cell.lat_min), w, h,
                                       fill=False, linewidth=1.5))
        ax.plot(lon, lat, marker="*", markersize=12)
        ax.text(lon, lat, f"  {lat:.2f}, {lon:.2f}", fontsize=8,
                va="bottom", ha="left")
    fig.tight_layout()
    return fig  # caller (gradio) owns lifecycle; plt.close("all") on next call


def predict_one(image):
    """Run inference on one PIL image. Returns dict with scalars + figure.

    Keys: lat, lon, cell_id, cal_conf, raw_conf, T, set_size, K,
          verdict ("ACCEPT"/"ABSTAIN"), box, quantile, figure.
    """
    import math
    import torch
    import torch.nn.functional as F
    from src.dataset import get_transforms
    from src.uncertainty import abstention_mask, conformal_prediction_set

    if image is None:
        raise ValueError("No image provided.")
    res = _load_resources()
    model, cells, T, q = res["model"], res["cells"], res["T"], res["quantile"]
    device, K = res["device"], res["K"]

    tfm = get_transforms(IMAGE_SIZE, train=False)
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        raw_probs = F.softmax(logits, dim=1)
        cal_probs = F.softmax(logits / max(T, 1e-3), dim=1)
    raw_conf = float(raw_probs.max().item())
    cal_conf = float(cal_probs.max().item())
    pred_id = int(cal_probs.argmax(dim=1).item())
    cell = cells[pred_id]
    lat, lon = float(cell.centroid_lat), float(cell.centroid_lon)

    cset = conformal_prediction_set(cal_probs.cpu(), quantile=q)
    set_size = int(cset.sum().item())
    accept = bool(abstention_mask(cal_probs.cpu(), threshold=ABSTAIN_THRESHOLD)[0].item())

    fig = render_map(lat, lon, cell)
    return {
        "lat": lat, "lon": lon, "cell_id": pred_id,
        "cal_conf": cal_conf, "raw_conf": raw_conf, "T": float(T),
        "set_size": set_size, "K": int(K),
        "verdict": "ACCEPT" if accept else "ABSTAIN",
        "temperature_fitted": res["temperature_fitted"],
        "conformal_fitted": q is not None,
        "calibration_matches_checkpoint": res["calibration_matches_checkpoint"],
        "box": (cell.lat_min, cell.lat_max, cell.lon_min, cell.lon_max),
        "quantile": q, "figure": fig,
        "finite": math.isfinite(lat) and math.isfinite(lon),
    }


def _format_markdown(d: dict) -> str:
    verdict = d["verdict"]
    badge = "**ACCEPT**" if verdict == "ACCEPT" else "**ABSTAIN**"
    q_txt = f"{d['quantile']:.4f}" if d["quantile"] is not None else "not fitted"
    lat0, lat1, lon0, lon1 = d["box"]
    confidence_label = "Uncalibrated softmax confidence"
    if d["temperature_fitted"]:
        confidence_label = (
            "Temperature-scaled confidence (checkpoint verified)"
            if d["calibration_matches_checkpoint"]
            else "Temperature-scaled confidence (artifact match unverified)"
        )
    set_label = (
        f"Conformal set size: **{d['set_size']} of {d['K']}**"
        if d["conformal_fitted"]
        else "Prediction set: top-1 only (no conformal guarantee)"
    )
    return (
        f"### Prediction: {badge}\n\n"
        f"- Predicted lat/lon: **{d['lat']:.4f}, {d['lon']:.4f}**\n"
        f"- Cell id: **{d['cell_id']}** (box lat [{lat0:.2f}, {lat1:.2f}], "
        f"lon [{lon0:.2f}, {lon1:.2f}])\n"
        f"- {confidence_label}: **{d['cal_conf'] * 100:.2f}%**\n"
        f"- Raw (unscaled) confidence: {d['raw_conf'] * 100:.2f}%\n"
        f"- Temperature T: {d['T']:.4f} | conformal quantile: {q_txt}\n"
        f"- {set_label}\n"
        f"- Threshold decision at max probability 0.5: **{verdict}**\n\n"
        f"_Note: haversine distance cannot be known for an uploaded photo "
        f"(no ground truth) — distance is only measurable on the labeled test set._"
    )


def build_predict_fn():
    """Importable entry point (no server launch). Returns predict_one."""
    return predict_one


def build_ui():
    import gradio as gr

    fn = build_predict_fn()

    def _gradio_fn(img):
        d = fn(img)
        return d["figure"], _format_markdown(d)

    with gr.Blocks(title="KPlaceNet demo") as demo:
        gr.Markdown("# KPlaceNet demo — coarse photo geolocation")
        gr.Markdown(
            "This classifier maps an image to one of the checkpoint's saved "
            "geographic cells and reports the winning cell centroid. Treat it as "
            "a coarse estimate, not an exact GPS location. Confidence and the "
            "0.5 threshold are only as meaningful as the supplied calibration "
            "artifact and evaluation protocol."
        )
        calibration = _load_calibration()
        calibration_state = (
            "temperature scaling and conformal threshold loaded; "
            "checkpoint hash checked on first inference"
            if calibration.temperature_fitted and calibration.conformal_fitted
            else "temperature scaling loaded; no conformal threshold"
            if calibration.temperature_fitted
            else "conformal threshold loaded without temperature scaling"
            if calibration.conformal_fitted
            else "no calibration artifact; raw softmax and top-1 only"
        )
        gr.Markdown(
            f"Checkpoint: `{CHECKPOINT}`  \n"
            f"Calibration: `{CALIB_JSON}` ({calibration_state})"
        )
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload photo")
            map_out = gr.Image(label="Predicted location map")
        info_out = gr.Markdown(label="Prediction details")
        img_in.change(fn=_gradio_fn, inputs=img_in, outputs=[map_out, info_out])
    return demo


if __name__ == "__main__":
    build_ui().launch()

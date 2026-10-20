"""KPlaceNet interactive demo (single file).

Global mode: L2 winner (l2_imagenet_1.0_layer4) + L3 calibration
(temperature + conformal), plotly world map (real countries).
Kerala mode: published L6 geographic smoothing when installed, L5 fallback; raw softmax only
(no calibration exists for this checkpoint), zoomed Kerala map.
Lazy, separately cached loads; one-image inference per mode.
Set KPLACENET_CHECKPOINT and optionally KPLACENET_CALIBRATION to select
artifacts. Model loading is lazy; inference returns a coarse cell estimate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import plotly.graph_objects as go

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
SIDECAR_CELLS = Path(os.environ.get(
    "KPLACENET_CELLS", str(ROOT / "checkpoints" / "l2_cells_full.json"),
)).expanduser()
from src.demo_config import kerala_artifact_paths
KERALA_CHECKPOINT, KERALA_CELLS = kerala_artifact_paths(ROOT)
ABSTAIN_THRESHOLD = 0.5
IMAGE_SIZE = 224

MODE_GLOBAL = "global"
MODE_KERALA = "kerala"
MODE_LABELS = {
    MODE_GLOBAL: "World · global model",
    MODE_KERALA: "Kerala area · regional model",
}
MODE_KEYS = {label: key for key, label in MODE_LABELS.items()}
MODE_KEYS.update({key: key for key in MODE_LABELS})

_CACHE: dict = {}
_KERALA_CACHE: dict = {}


def _mode_key(mode) -> str:
    """Map a radio label (or internal key) to an internal mode key."""
    if mode is None:
        return MODE_GLOBAL
    return MODE_KEYS.get(str(mode), MODE_GLOBAL)


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
                if s and (
                    abs(float(s["centroid_lat"]) - c.centroid_lat) < 1e-6
                    and abs(float(s["centroid_lon"]) - c.centroid_lon) < 1e-6
                ):
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


def _load_kerala_resources():
    """Lazy, cached Kerala model + full-box cells. Returns dict.

    The L5 checkpoint has no calibration, so no temperature or quantile is
    loaded here. Boxes come from cells.json (checkpoint cells carry centroids
    only), matching how the global sidecar enriches its degenerate boxes.
    """
    if _KERALA_CACHE:
        return _KERALA_CACHE
    import torch
    from src.cells import load_cells_json
    from src.eval_contract import validate_eval_checkpoint
    from src.model import build_model

    ckpt = torch.load(str(KERALA_CHECKPOINT), map_location="cpu")
    num_cells, cells_raw = validate_eval_checkpoint(ckpt)
    cells = sorted(load_cells_json(KERALA_CELLS), key=lambda c: c.cell_id)
    if len(cells) != num_cells or any(
        c.cell_id != saved["cell_id"]
        or abs(c.centroid_lat - float(saved["centroid_lat"])) > 1e-6
        or abs(c.centroid_lon - float(saved["centroid_lon"])) > 1e-6
        for c, saved in zip(cells, cells_raw)
    ):
        raise ValueError("Kerala sidecar cells do not match the checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_cells=num_cells, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    bank_path = ROOT / "checkpoints/l7_kerala_hybrid/train_bank.pt"
    bank = None
    if bank_path.is_file():
        from src.demo_config import sha256_file
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        if bank["checkpoint_sha256"] != sha256_file(KERALA_CHECKPOINT):
            raise ValueError("KNN bank belongs to a different checkpoint; remove it or install its matching L6 pack")
        bank = {"x": bank["x"].to(device), "labels": bank["labels"].to(device)}
    _KERALA_CACHE.update(model=model, cells=cells, device=device, bank=bank,
                         K=num_cells, T=None, quantile=None)
    return _KERALA_CACHE


def render_map(lat: float, lon: float, cell):
    """World map with real countries: 300 region dots + cell box + prediction star."""
    res = _load_resources()
    traces = [go.Scattergeo(
        lat=[float(c.centroid_lat) for c in res["cells"]],
        lon=[float(c.centroid_lon) for c in res["cells"]],
        mode="markers",
        marker={"size": 3, "color": "rgba(30,144,255,0.45)"},
        name="300 model regions",
        hovertemplate="cell %{customdata}<extra></extra>",
        customdata=[int(c.cell_id) for c in res["cells"]],
    )]
    # Predicted cell bounding box as a red outline (skip if degenerate).
    if (cell.lon_max - cell.lon_min) > 0 and (cell.lat_max - cell.lat_min) > 0:
        box_lon = [cell.lon_min, cell.lon_max, cell.lon_max, cell.lon_min, cell.lon_min]
        box_lat = [cell.lat_min, cell.lat_min, cell.lat_max, cell.lat_max, cell.lat_min]
        traces.append(go.Scattergeo(
            lat=[float(v) for v in box_lat], lon=[float(v) for v in box_lon],
            mode="lines", line={"width": 2, "color": "red"},
            name=f"cell {cell.cell_id} box",
        ))
    traces.append(go.Scattergeo(
        lat=[float(lat)], lon=[float(lon)],
        mode="markers", marker={"size": 14, "color": "red", "symbol": "star"},
        name=f"predicted {float(lat):.2f},{float(lon):.2f}",
    ))
    fig = go.Figure(
        data=traces,
        layout={
            "geo": {
                "scope": "world",
                "projection": {"type": "equirectangular"},
                "showland": True, "landcolor": "rgb(235,235,220)",
                "showocean": True, "oceancolor": "rgb(200,225,245)",
                "showcountries": True, "countrycolor": "rgb(150,150,150)",
                "coastlinecolor": "rgb(120,120,120)", "showframe": True,
            },
            "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
            "title": f"Model prediction — cell {cell.cell_id} "
                      f"({float(lat):.2f}, {float(lon):.2f})",
            "showlegend": True,
        },
    )
    return _style_map(fig)


def render_kerala_map(lat: float, lon: float, cell, fine=False):
    """Kerala map: same style as the world map, zoomed to the state.

    300 cell centroids + predicted cell box + prediction star. The geo axis
    ranges crop the equirectangular projection to roughly 8.0-12.8N /
    74.5-77.5E, so the same land/ocean/country colors read at state scale.
    """
    res = _load_kerala_resources()
    traces = [go.Scattergeo(
        lat=[float(c.centroid_lat) for c in res["cells"]],
        lon=[float(c.centroid_lon) for c in res["cells"]],
        mode="markers",
        marker={"size": 3, "color": "rgba(30,144,255,0.45)"},
        name="300 Kerala cells",
        hovertemplate="cell %{customdata}<extra></extra>",
        customdata=[int(c.cell_id) for c in res["cells"]],
    )]
    # Predicted cell bounding box as a red outline (skip if degenerate).
    if not fine and (cell.lon_max - cell.lon_min) > 0 and (cell.lat_max - cell.lat_min) > 0:
        box_lon = [cell.lon_min, cell.lon_max, cell.lon_max, cell.lon_min, cell.lon_min]
        box_lat = [cell.lat_min, cell.lat_min, cell.lat_max, cell.lat_max, cell.lat_min]
        traces.append(go.Scattergeo(
            lat=[float(v) for v in box_lat], lon=[float(v) for v in box_lon],
            mode="lines", line={"width": 2, "color": "red"},
            name=f"cell {cell.cell_id} box",
        ))
    traces.append(go.Scattergeo(
        lat=[float(lat)], lon=[float(lon)],
        mode="markers", marker={"size": 14, "color": "red", "symbol": "star"},
        name=f"predicted {float(lat):.2f},{float(lon):.2f}",
    ))
    traces.append(go.Scattergeo(
        lat=[12.50, 11.87, 11.26, 10.53, 9.93, 9.59, 8.89, 8.52],
        lon=[74.99, 75.37, 75.78, 76.21, 76.27, 76.52, 76.61, 76.94],
        text=["Kasaragod", "Kannur", "Kozhikode", "Thrissur", "Kochi", "Kottayam", "Kollam", "Thiruvananthapuram"],
        mode="markers+text", textposition="middle right",
        textfont={"size": 13, "color": "#334b60"},
        marker={"size": 4, "color": "#657b8e"},
        showlegend=False, hoverinfo="skip",
    ))
    fig = go.Figure(
        data=traces,
        layout={
            "geo": {
                "scope": "world",
                "projection": {"type": "equirectangular"},
                "lataxis": {"range": [7.7, 13.1]},
                "lonaxis": {"range": [72.6, 80.2]},
                "showland": True, "landcolor": "rgb(235,235,220)",
                "showocean": True, "oceancolor": "rgb(200,225,245)",
                "showcountries": True, "countrycolor": "rgb(150,150,150)",
                "coastlinecolor": "rgb(120,120,120)", "showframe": True,
            },
            "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
            "title": f"Kerala mode prediction — cell {cell.cell_id} "
                      f"({float(lat):.2f}, {float(lon):.2f})",
            "showlegend": True,
        },
    )
    return _style_map(fig, kerala=True)


def _style_map(fig, kerala=False):
    """Consistent, spacious map styling for projection on a classroom screen."""
    fig.update_geos(
        resolution=50, landcolor="#E9E1D1", oceancolor="#E7EAE2",
        lakecolor="#E7EAE2", showlakes=True, bgcolor="#E7EAE2",
        coastlinecolor="#A4A18E", countrycolor="#BCB6A6", showframe=False,
    )
    if kerala:
        fig.update_geos(
            lataxis={"range": [7.2, 13.8], "showgrid": True, "dtick": 2, "gridcolor": "#D3D8CF", "gridwidth": 0.5},
            lonaxis={"range": [69.5, 84.5], "showgrid": True, "dtick": 2, "gridcolor": "#D3D8CF", "gridwidth": 0.5},
        )
    for trace in fig.data:
        if trace.name and ("cells" in trace.name or "regions" in trace.name):
            trace.marker.update(size=4, color="rgba(23,44,66,0.38)")
            trace.name = "Model regions"
        elif trace.name and "box" in trace.name:
            trace.line.update(color="#D5663F", width=2)
            trace.update(fill="toself", fillcolor="rgba(213,102,63,0.10)")
            trace.name = "Selected cell bounds"
        elif trace.name and "predicted" in trace.name:
            trace.marker.update(size=17, color="#D5663F", symbol="circle",
                                line={"color": "white", "width": 3})
            trace.name = "Predicted location"
            trace.hovertemplate = "%{lat:.4f}° N, %{lon:.4f}° E<extra>Prediction</extra>"
    fig.update_layout(
        height=410, autosize=True, title=None,
        paper_bgcolor="#E7EAE2", plot_bgcolor="#E7EAE2",
        font={"family": "Arial, sans-serif", "color": "#172C42", "size": 12},
        margin={"l": 0, "r": 0, "t": 18, "b": 44},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center",
                "y": -0.04, "yanchor": "top", "font": {"size": 13}},
    )
    return fig


def _predict_kerala(image, use_knn=True, use_fine=True) -> dict:
    """Raw softmax (no calibration) -> top cell centroid + Kerala map."""
    import math
    import torch
    import torch.nn.functional as F
    from src.dataset import get_transforms

    if image is None:
        raise ValueError("No image provided.")
    res = _load_kerala_resources()
    model, cells = res["model"], res["cells"]
    device, K = res["device"], res["K"]

    tfm = get_transforms(IMAGE_SIZE, train=False)
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.backbone(x).flatten(1)
        raw_probs = F.softmax(model.head(features), dim=1)
        combined = raw_probs
        hybrid = bool(use_knn and res.get("bank") is not None)
        if hybrid:
            from src.retrieval import knn_prob
            votes = knn_prob(features, res["bank"]["x"], res["bank"]["labels"], num_cells=K)
            combined = .8 * raw_probs + .2 * votes
    raw_conf = float(raw_probs.max().item())
    pred_id = int(combined.argmax(dim=1).item())
    cell = cells[pred_id]
    lat, lon = float(cell.centroid_lat), float(cell.centroid_lon)

    fine = None
    if use_fine and (ROOT / "checkpoints/fine_retrieval/bank.pt").is_file():
        from src.fine_inference import load_fine_resources, predict_fine
        if "fine_resources" not in res:
            from src.demo_config import sha256_file
            res["fine_resources"] = load_fine_resources(ROOT, device, sha256_file(KERALA_CHECKPOINT))
        fine = predict_fine(x, features, res["fine_resources"])
        lat, lon = fine["lat"], fine["lon"]
    fig = render_kerala_map(lat, lon, cell, fine=bool(fine))
    return {
        "fine": bool(fine), "similarity": fine["similarity"] if fine else None,
        "reference_id": fine["reference_id"] if fine else None,
        "reference_path": str(ROOT / "data/kerala_osv5m/train/images" / (fine["reference_id"] + ".jpg")) if fine else None,
        "mode": MODE_KERALA,
        "hybrid": hybrid, "hybrid_conf": float(combined.max().item()),
        "pipeline": "Fine retrieval: ResNet + DINOv2" if fine else "ResNet + KNN (80/20)" if hybrid else "ResNet classifier",
        "lat": lat, "lon": lon, "cell_id": pred_id,
        "cal_conf": None, "raw_conf": raw_conf, "T": None,
        "set_size": None, "K": int(K),
        "verdict": "RAW",
        "box": (cell.lat_min, cell.lat_max, cell.lon_min, cell.lon_max),
        "quantile": None, "figure": fig,
        "finite": math.isfinite(lat) and math.isfinite(lon),
    }


def predict_one(image, mode=MODE_GLOBAL, use_knn=True, use_fine=True):
    """Run inference on one PIL image. Returns dict with scalars + figure.

    mode: "global" (calibrated L2/L3) or "kerala" (raw L5, uncalibrated).
    Global keys: lat, lon, cell_id, cal_conf, raw_conf, T, set_size, K,
          verdict ("ACCEPT"/"ABSTAIN"), box, quantile, figure.
    Kerala keys: same shape, with cal_conf/T/quantile/set_size = None
          and verdict = "RAW" (no threshold is applied).
    """
    if _mode_key(mode) == MODE_KERALA:
        return _predict_kerala(image, use_knn=use_knn, use_fine=use_fine)

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
        "mode": MODE_GLOBAL,
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
    if d.get("fine"):
        return (f"### Fine-coordinate retrieval\n\n"
                f"- Position: **{d['lat']:.6f}, {d['lon']:.6f}**\n"
                f"- Pipeline: {d['pipeline']}\n"
                f"- Retrieved training image ID: {d['reference_id']}\n"
                f"- Cosine similarity: {d['similarity']:.4f} (not calibrated confidence)\n\n"
                "The output is the matched training photograph's coordinate, not a cell center. "
                "Descriptors combine normalized ResNet and DINOv2 features equally. "
                "The bank excludes validation/test images. Fine does not mean verified accuracy; "
                "outside-region images can still retrieve a wrong match.")
    lat0, lat1, lon0, lon1 = d["box"]
    if d.get("mode") == MODE_KERALA:
        return (
            f"### Prediction (Kerala-area mode)\n\n"
            f"- Pipeline: {d.get('pipeline', 'ResNet classifier')}\n"
            f"- Selected-cell score: {d.get('hybrid_conf', d['raw_conf']):.2%} (uncalibrated)\n"
            f"- Predicted lat/lon: **{d['lat']:.4f}, {d['lon']:.4f}**\n"
            f"- Cell id: **{d['cell_id']}** (box lat [{lat0:.2f}, {lat1:.2f}], "
            f"lon [{lon0:.2f}, {lon1:.2f}])\n"
            f"- Classifier maximum score: **{d['raw_conf'] * 100:.2f}%** (uncalibrated)\n"
            f"- Conformal set: n/a — Kerala mode is uncalibrated\n"
            f"- Calibration: none; optional KNN blends 80% classifier with 20% training-only neighbor votes.\n"
            f"- Decision: raw confidence only (no ACCEPT/ABSTAIN threshold)\n\n"
            f"_Note: haversine distance cannot be known for an uploaded photo "
            f"(no ground truth) — distance is only measurable on the labeled test set._"
        )

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
    """Importable entry point (no server launch). Returns predict_one.

    The returned callable takes (image, mode="global"); mode may be an
    internal key ("global"/"kerala") or the matching UI radio label.
    """
    return predict_one


DEMO_CSS = """
html, body {background: #F5F1E8 !important; color-scheme: light;}
html body .gradio-container {width: calc(100% - 96px) !important; max-width: none !important; margin: 0 auto !important; padding: 24px 0 !important; font-size: 16px; background: #F5F1E8; color: #172C42;}
.gradio-container .main {padding: 0 !important;}
[data-testid="block-info"] {color: #172C42 !important;}
#hero {display: flex; align-items: baseline; gap: 28px; background: transparent; padding: 0 0 16px; border-radius: 0; border-bottom: 1px solid #D8D0C4; margin-bottom: 8px;}
#hero h1 {white-space: nowrap; font-family: Arial, Helvetica, sans-serif; color: #172C42; font-size: 34px; font-weight: 750; letter-spacing: -1px; margin: 0 0 5px;}
#hero p {color: #626A70; font-size: 17px; margin: 0; max-width: 1000px;}
.eyebrow {color: #B84F2D; letter-spacing: 1.5px; font-size: 11px; font-weight: 700;}
#result {background: #FAF7F0; border: 1px solid #D8D0C4; border-radius: 10px; padding: 18px 22px; min-height: 140px;}
#result h2 {font-family: Arial, Helvetica, sans-serif; font-size: 25px; font-weight: 700; color: #172C42; margin: 6px 0 12px;}
.result-grid {display: grid; grid-template-columns: 1fr; gap: 16px; font-variant-numeric: tabular-nums;}
.result-grid span {display: block; color: #626A70; font-size: 12px; margin-bottom: 5px;}
.result-grid strong {font-size: 22px; color: #172C42;}
.maps-link {display: inline-block; margin-top: 16px; padding: 11px 15px; background: #172C42; color: #F5F1E8 !important; border-radius: 5px; font-size: 15px; font-weight: 600; text-decoration: none !important;}
.maps-link:hover {background: #243F59;}
.maps-link:focus-visible {outline: 3px solid #D5663F; outline-offset: 3px;}
.result-note {color: #626A70; font-size: 14px; margin: 12px 0 0;}
#run-button {background: #172C42; color: #F5F1E8; border: 0; border-bottom: 3px solid #D5663F; min-height: 52px; font-size: 18px; font-weight: 600;}
#run-button:hover {background: #243F59;}
#reference-gallery .grid-container {grid-template-columns: minmax(0, 440px) !important; grid-auto-rows: 190px !important;}
#reference-gallery .gallery-item, #reference-gallery .thumbnail-item {height: 190px !important; min-height: 0 !important;}
#reference-gallery img {height: 100% !important; width: 100% !important; object-fit: contain !important;}
.demo-gallery {background: transparent !important; height: auto !important;}
.demo-gallery .grid-wrap {height: auto !important;}
.demo-gallery .grid-container {grid-template-columns: repeat(5, minmax(0, 1fr)) !important; grid-auto-rows: 160px !important;}
.demo-gallery .gallery-item, .demo-gallery .thumbnail-item {min-width: 0 !important; height: 160px !important;}
.demo-gallery .caption-label {display: block !important; opacity: 1 !important; position: absolute !important; bottom: 0 !important; left: 0 !important; right: 0 !important; color: #172C42 !important; background: rgba(250,247,240,.97) !important; padding: 7px 8px !important; font-size: 14px !important; line-height: 1.3 !important; white-space: normal !important;}
#drop-status {min-height: 0 !important; padding: 0 !important;}
#drop-status p:empty {display: none;}
#drop-status p {font-size: 14px; color: #B84F2D;}
#workspace {gap: 24px;}
#workspace h3, #examples-heading h3 {font-size: 21px;}
#location-map {background: #E7EAE2; border: 1px solid #D8D0C4; border-radius: 8px; overflow: hidden;}
#example-tabs button {font-size: 16px;}
footer {display: none !important;}
button:focus-visible {outline: 3px solid #D5663F !important; outline-offset: 3px;}
@media(min-width: 1100px) {#workspace {flex-wrap: nowrap !important;} #workspace > div {min-width: 0 !important;}}
@media(max-width: 1050px) {.demo-gallery .grid-container {grid-template-columns: repeat(3, minmax(0, 1fr)) !important;}}
@media(max-width: 760px) {html body .gradio-container.gradio-container {width: calc(100% - 28px) !important; padding: 14px 0 !important;} #hero {display: block; padding: 18px;} #hero h1 {font-size: 26px;} .result-grid {grid-template-columns: 1fr 1fr; gap: 14px;} .result-grid strong {font-size: 19px;} .demo-gallery .grid-container {grid-template-columns: repeat(2, minmax(0, 1fr)) !important;} }
"""


def _result_html(d=None):
    if d is None:
        return ('<div class="eyebrow" style="color:#B84F2D">PREDICTION SUMMARY</div>'
                '<h2>Where was this photo taken?</h2>'
                '<p class="result-note">Upload a street scene or choose an example, then select Locate photo.</p>')
    kerala = d.get("mode") == MODE_KERALA
    confidence = d.get("hybrid_conf", d["raw_conf"]) if kerala else d["cal_conf"]
    status = "Regional assumption · not verified" if kerala else ("Accepted estimate" if d["verdict"] == "ACCEPT" else "Uncertain estimate · abstained")
    calibrated = not kerala and d.get("temperature_fitted", False)
    verified = calibrated and d.get("calibration_matches_checkpoint", False)
    score_label = ("Calibrated confidence" if verified else "Temperature-scaled score" if calibrated else "Raw model score")
    if d.get("hybrid"):
        score_label = "Combined retrieval score"
    note = ("Raw model score · uncalibrated; this is not a measured probability of correctness."
            if not calibrated else "Calibrated model score · the model abstains below 50% confidence."
            if verified else "Historical temperature scaling · calibration artifact identity unverified. The model abstains below a 50% score.")
    if d.get("hybrid"):
        note = "ResNet + training-only KNN, weighted 80/20. Combined score is uncalibrated, not probability of correctness."
    if kerala:
        note += " Assumes a scene from the Kerala-area dataset; outside-region images also receive a prediction. The training rectangle includes neighboring areas."
    fine = bool(d.get("fine"))
    if fine:
        score_label = "Appearance similarity (cosine)"
        note = "ResNet + DINOv2 retrieves a training photograph's coordinate. Similarity is not probability of correctness. Regional input is assumed; wrong distant matches remain possible."
    score_text = f"{d['similarity']:.3f}" if fine else f"{confidence:.1%}"
    title = "Retrieved location" if fine else f"{'Kerala area' if kerala else 'Global'} · predicted cell {d['cell_id']}"
    coordinate_note = "Marker: retrieved training-photo coordinate." if fine else "The marker represents a cell centroid."
    return (
        f'<div class="eyebrow" style="color:#B84F2D">{status.upper()}</div>' 
        f'<h2>{title}</h2>'
        '<div class="result-grid">'
        f'<div><span>Latitude</span><strong>{d["lat"]:.4f}°</strong></div>'
        f'<div><span>Longitude</span><strong>{d["lon"]:.4f}°</strong></div>'
        f'<div><span>{score_label}</span><strong>{score_text}</strong></div>'
        f'</div><p class="result-note">{note} {coordinate_note}</p>'
        f'<a class="maps-link" href="https://www.google.com/maps/search/?api=1&amp;query={d["lat"]:.8f}%2C{d["lon"]:.8f}" '
        'target="_blank" rel="noopener noreferrer">Open in Google Maps ↗</a>'
    )


def _empty_map(mode):
    kerala = _mode_key(mode) == MODE_KERALA
    path = KERALA_CELLS if kerala else SIDECAR_CELLS
    fig = go.Figure()
    if path.is_file():
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        fig.add_trace(go.Scattergeo(
            lat=[c["centroid_lat"] for c in cells],
            lon=[c["centroid_lon"] for c in cells],
            mode="markers", name="Model regions",
            marker={"size": 4, "color": "#172C42"},
            hovertemplate="Training-derived cell center<extra></extra>",
        ))
    fig.update_layout(geo={"projection": {"type": "equirectangular"},
                           "showland": True, "showocean": True, "showcountries": True})
    if kerala:
        fig.update_geos(lataxis_range=[7.7, 13.1], lonaxis_range=[73.5, 79.0])
        fig.add_trace(go.Scattergeo(
            lat=[11.26, 9.93, 9.59, 8.52], lon=[75.78, 76.27, 76.52, 76.94],
            text=["Kozhikode", "Kochi", "Kottayam", "Thiruvananthapuram"],
            mode="text", textposition="middle right", textfont={"size": 13},
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_annotation(text="Model regions · select a photo to predict", x=0.5, y=1,
                       xref="paper", yref="paper", showarrow=False,
                       font={"size": 12, "color": "#515D66"})
    return _style_map(fig, kerala=kerala)


def _demo_theme(gr):
    # The interface is intentionally light even with a dark OS preference.
    # Mirror every theme token instead of mixing dark components with light CSS.
    import inspect
    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="stone", font=["Arial", "Helvetica", "sans-serif"]).set(
        body_background_fill="#F5F1E8", body_text_color="#172C42",
        body_text_color_subdued="#626A70", block_background_fill="#FAF7F0",
        block_label_background_fill="#EEE7DA", block_label_text_color="#172C42",
        background_fill_primary="#FAF7F0", background_fill_secondary="#EEE7DA",
        input_background_fill="#FFFCF7", border_color_primary="#D8D0C4",
        checkbox_label_background_fill="#EEE7DA",
        checkbox_label_background_fill_selected="#E4D7C5",
        checkbox_label_text_color="#172C42", checkbox_label_text_color_selected="#172C42",
    )
    values = theme.to_dict()["theme"]
    allowed = inspect.signature(theme.set).parameters
    return theme.set(**{key: values[key[:-5]] for key in values
                        if key.endswith("_dark") and key in allowed and key[:-5] in values})


def build_ui():
    import gradio as gr

    def run_prediction(img, mode, method):
        if img is None:
            gr.Warning("Choose a photo before running the model.")
            return _empty_map(mode), _result_html(), "Upload a photo to see model details.", gr.update(value=[], visible=False)
        try:
            d = predict_one(img, mode, use_knn=method != "classifier", use_fine=method == "fine")
            if method == "fine" and d.get("mode") == MODE_KERALA and not d.get("fine"):
                gr.Warning("Fine retrieval assets are missing; showing the available regional model.")
            if method == "hybrid" and d.get("mode") == MODE_KERALA and not d.get("hybrid"):
                gr.Warning("The matching KNN bank is not installed; this result uses ResNet alone.")
        except Exception as exc:
            raise gr.Error("Prediction could not complete. Check the model checkpoint and try again.") from exc
        path = d.get("reference_path")
        references = [(path, "Training reference " + d["reference_id"])] if path and Path(path).is_file() else []
        return d["figure"], _result_html(d), _format_markdown(d), gr.update(value=references, visible=bool(references))

    def reset_result(mode):
        return _empty_map(mode), _result_html(), "Run a prediction to inspect calibration and cell bounds.", gr.update(value=[], visible=False)

    # Gradio 6 moved theme/css from Blocks to launch; retain older releases.
    style = {} if int(gr.__version__.split(".")[0]) >= 6 else {
        "css": DEMO_CSS, "theme": _demo_theme(gr),
    }
    with gr.Blocks(title="KPlaceNet | Visual geolocation", fill_width=True, **style) as demo:
        gr.HTML('<h1>KPlaceNet</h1>'
                '<p>Photo geolocation · Choose a street scene, retrieve a location, and inspect the visual match.</p>', elem_id="hero")
        with gr.Row(equal_height=False, elem_id="workspace"):
            with gr.Column(scale=4, min_width=300, elem_id="photo-panel"):
                gr.Markdown("### Photograph and model")
                mode_in = gr.Radio(choices=[(MODE_LABELS[MODE_KERALA], MODE_KERALA),
                                            (MODE_LABELS[MODE_GLOBAL], MODE_GLOBAL)],
                                   value=MODE_KERALA, label="Geographic scope")
                img_in = gr.Image(type="pil", format="png", label="Street scene", height=250,
                                  sources=["upload", "clipboard"], interactive=True, elem_id="street-image")
                gr.HTML('<p role="status" aria-live="polite"></p>', elem_id="drop-status")
                method_in = gr.Radio(choices=[("Fine coordinates · ResNet + DINOv2", "fine"),
                                              ("Regional cells · ResNet + KNN", "hybrid"),
                                              ("Regional cells · classifier only", "classifier")],
                                         value="fine" if (ROOT / "checkpoints/fine_retrieval/bank.pt").is_file() else "hybrid",
                                         label="Regional prediction method")
                run = gr.Button("Locate photo →", variant="primary", elem_id="run-button")
                gr.Markdown("Fine retrieval returns a matched training-photo coordinate. Cell methods return one of 300 centers. Regional mode assumes Kerala-area input and cannot reject outside-region photos.")
            with gr.Column(scale=6, min_width=300, elem_id="map-panel"):
                gr.Markdown("### Predicted region")
                map_out = gr.Plot(value=_empty_map(MODE_KERALA), label="Location map", show_label=False, elem_id="location-map")
            with gr.Column(scale=3, min_width=260, elem_id="summary-panel"):
                gr.Markdown("### Prediction")
                result_out = gr.HTML(_result_html(), elem_id="result")
        reference_out = gr.Gallery(label="Retrieved training reference (different from the uploaded test photo)",
                                   columns=1, height=230, object_fit="contain", visible=False,
                                   allow_preview=True, interactive=False, elem_id="reference-gallery")
        gr.Markdown("### Demo photographs", elem_id="examples-heading")
        gr.Markdown("Choose a photograph, then select **Locate photo**. "
                    "Selected test-set photos; these examples do not represent average performance.")
        groups = [
            ("Kerala examples", MODE_KERALA, ROOT / "data/kerala_osv5m/test/images", [
                ("319636679783973.jpg", "Idukki"),
                ("1083931132408866.jpg", "Kottayam"),
                ("313393843511505.jpg", "Kochi"),
                ("288990592896618.jpg", "Thiruvananthapuram"),
                ("905069160348732.jpg", "Southern Kerala")]),
            ("Global examples", MODE_GLOBAL, ROOT / "data/osv5m_test/images", [
                ("3002826110043032.jpg", "Rajasthan"),
                ("271239751462351.jpg", "Rajasthan"),
                ("905946183585986.jpg", "United Kingdom"),
                ("1472138156457085.jpg", "Ecuador"),
                ("758042071550066.jpg", "Kerala")]),
        ]
        example_label = gr.Textbox(label="Selected demonstration", interactive=False, visible=False)
        with gr.Tabs(elem_id="example-tabs"):
            for label, mode, folder, photos in groups:
                with gr.Tab(label):
                    examples = [[str(folder / filename), mode, caption]
                                for filename, caption in photos if (folder / filename).exists()]
                    if examples:
                        gallery = gr.Gallery(
                            value=[(row[0], row[2]) for row in examples], label=label,
                            columns=5, rows=1, height=175, object_fit="cover", buttons=[],
                            allow_preview=False, interactive=False, elem_classes=["demo-gallery"],
                        )

                        def select_example(evt: gr.SelectData, items=examples):
                            path, selected_mode, caption = items[evt.index]
                            from PIL import Image
                            return Image.open(path).convert("RGB"), selected_mode, caption

                        select_example.__annotations__["evt"] = gr.SelectData
                        gallery.select(select_example, outputs=[img_in, mode_in, example_label],
                                       show_progress="hidden")
        with gr.Accordion("Prediction details", open=False):
            info_out = gr.Markdown("Run a prediction to inspect calibration and cell bounds.")
            gr.Markdown("Fine retrieval combines ResNet and DINOv2 descriptors and returns the matched training photo coordinate. Cosine similarity is not a probability. Kerala-area scores are uncalibrated. Optional KNN uses training photographs only and blends votes 80/20 with the classifier; none of these regional methods detects outside-region input. "
                        "Global mode uses temperature scaling and conformal prediction. "
                        "Cell bounding boxes are geographic extents, not confidence intervals. "
                        "Distance error requires a known reference location.")
        gr.Markdown("**KPlaceNet** · ResNet-50 + DINOv2 visual geolocation · Street imagery: OSV-5M (CC-BY-SA-4.0)")
        outputs = [map_out, result_out, info_out, reference_out]
        run.click(run_prediction, inputs=[img_in, mode_in, method_in], outputs=outputs)
        img_in.change(reset_result, inputs=[mode_in], outputs=outputs)
        mode_in.change(reset_result, inputs=[mode_in], outputs=outputs)
        method_in.change(reset_result, inputs=[mode_in], outputs=outputs)
        mode_in.change(lambda mode: gr.update(visible=_mode_key(mode) == MODE_KERALA), inputs=[mode_in], outputs=[method_in])
    return demo


def launch_demo(**kwargs):
    """Launch with the classroom theme on both Gradio 5 and 6."""
    import gradio as gr

    if int(gr.__version__.split(".")[0]) >= 6:
        kwargs.setdefault("css", DEMO_CSS)
        kwargs.setdefault("theme", _demo_theme(gr))
        # Gradio scopes selectors inside media rules; root layout must be global.
        kwargs.setdefault("head", "<style>" + DEMO_CSS + "</style>")
        kwargs.setdefault("js", r"""() => {
            // Native Gradio handles File drops; browser images arrive as URLs/HTML.
            const imageZone = event => event.target instanceof Element && event.target.closest('#street-image');
            const status = message => {
                const node = document.querySelector('#drop-status p');
                if (node) node.textContent = message;
            };
            document.addEventListener('dragover', event => {
                if (imageZone(event)) event.preventDefault();
            }, true);
            document.addEventListener('drop', async event => {
                if (!imageZone(event) || !event.dataTransfer) return;
                const transfer = event.dataTransfer;
                status('');
                if (transfer.files.length) return;
                event.preventDefault();
                event.stopImmediatePropagation();
                const html = new DOMParser().parseFromString(transfer.getData('text/html'), 'text/html');
                const source = html.querySelector('img')?.getAttribute('src') ||
                    transfer.getData('text/uri-list').split('\n').find(line => line && !line.startsWith('#')) ||
                    transfer.getData('text/plain').trim();
                try {
                    const url = new URL(source);
                    if (!['https:', 'http:', 'data:'].includes(url.protocol)) throw new Error('Unsupported image link');
                    status('Loading dropped image…');
                    const response = await fetch(url.href, {credentials: 'omit', signal: AbortSignal.timeout(15000)});
                    if (!response.ok) throw new Error('Image request failed');
                    const blob = await response.blob();
                    if (!blob.type.startsWith('image/')) throw new Error('The link is not an image');
                    const file = new File([blob], 'dropped-image.' + (blob.type.split('/')[1] || 'jpg'), {type: blob.type});
                    const input = document.querySelector('#street-image input[type=file]');
                    if (!input) throw new Error('Upload input unavailable');
                    const files = new DataTransfer();
                    files.items.add(file);
                    input.files = files.files;
                    input.dispatchEvent(new Event('change', {bubbles: true}));
                    status('');
                } catch (error) {
                    status('This website blocked the image or supplied a page link. Save the image to your device, then drop the saved file here or click Upload.');
                }
            }, true);
            let timer;
            window.addEventListener('resize', () => {
                clearTimeout(timer);
                timer = setTimeout(() => document.querySelectorAll('.js-plotly-plot').forEach(
                    plot => window.Plotly && window.Plotly.Plots.resize(plot)), 150);
            });
        }""")
    return build_ui().launch(**kwargs)


if __name__ == "__main__":
    launch_demo()

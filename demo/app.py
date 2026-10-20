"""KPlaceNet interactive demo (single file).

Global mode: L2 winner (l2_imagenet_1.0_layer4) + L3 calibration
(temperature + conformal), plotly world map (real countries).
Kerala mode: L5 Kerala pinpoint (l5_kerala_kmeans300), raw softmax only
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
KERALA_CHECKPOINT = ROOT / "checkpoints" / "l5_kerala_kmeans300" / "last.pt"
KERALA_CELLS = ROOT / "checkpoints" / "l5_kerala_kmeans300" / "cells.json"
ABSTAIN_THRESHOLD = 0.5
IMAGE_SIZE = 224

MODE_GLOBAL = "global"
MODE_KERALA = "kerala"
MODE_LABELS = {
    MODE_GLOBAL: "World · global model",
    MODE_KERALA: "Kerala · regional model",
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

    _KERALA_CACHE.update(model=model, cells=cells, device=device,
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


def render_kerala_map(lat: float, lon: float, cell):
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
    traces.append(go.Scattergeo(
        lat=[12.50, 11.87, 11.26, 10.53, 9.93, 9.59, 8.89, 8.52],
        lon=[74.99, 75.37, 75.78, 76.21, 76.27, 76.52, 76.61, 76.94],
        text=["Kasaragod", "Kannur", "Kozhikode", "Thrissur", "Kochi", "Kottayam", "Kollam", "Thiruvananthapuram"],
        mode="markers+text", textposition="middle right",
        textfont={"size": 11, "color": "#334b60"},
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
        resolution=50, landcolor="#e7eee8", oceancolor="#dcebf2",
        lakecolor="#dcebf2", showlakes=True, bgcolor="#f3f7f9",
        coastlinecolor="#91aaa9", countrycolor="#b6c7c6", showframe=False,
    )
    for trace in fig.data:
        if trace.name and ("cells" in trace.name or "regions" in trace.name):
            trace.marker.update(size=4, color="rgba(28,125,139,0.35)")
            trace.name = "Model regions"
        elif trace.name and "box" in trace.name:
            trace.line.update(color="#d08b32", width=2)
            trace.update(fill="toself", fillcolor="rgba(208,139,50,0.12)")
            trace.name = "Selected cell bounds"
        elif trace.name and "predicted" in trace.name:
            trace.marker.update(size=17, color="#d36540", symbol="circle",
                                line={"color": "white", "width": 3})
            trace.name = "Predicted location"
            trace.hovertemplate = "%{lat:.4f}° N, %{lon:.4f}° E<extra>Prediction</extra>"
    fig.update_layout(
        height=490, autosize=True, title=None,
        paper_bgcolor="#f3f7f9", plot_bgcolor="#f3f7f9",
        font={"family": "Georgia, serif", "color": "#243e53", "size": 12},
        margin={"l": 8, "r": 8, "t": 12, "b": 58},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center",
                "y": -0.04, "yanchor": "top", "font": {"size": 11}},
    )
    return fig


def _predict_kerala(image) -> dict:
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
        logits = model(x)
        raw_probs = F.softmax(logits, dim=1)
    raw_conf = float(raw_probs.max().item())
    pred_id = int(raw_probs.argmax(dim=1).item())
    cell = cells[pred_id]
    lat, lon = float(cell.centroid_lat), float(cell.centroid_lon)

    fig = render_kerala_map(lat, lon, cell)
    return {
        "mode": MODE_KERALA,
        "lat": lat, "lon": lon, "cell_id": pred_id,
        "cal_conf": None, "raw_conf": raw_conf, "T": None,
        "set_size": None, "K": int(K),
        "verdict": "RAW",
        "box": (cell.lat_min, cell.lat_max, cell.lon_min, cell.lon_max),
        "quantile": None, "figure": fig,
        "finite": math.isfinite(lat) and math.isfinite(lon),
    }


def predict_one(image, mode=MODE_GLOBAL):
    """Run inference on one PIL image. Returns dict with scalars + figure.

    mode: "global" (calibrated L2/L3) or "kerala" (raw L5, uncalibrated).
    Global keys: lat, lon, cell_id, cal_conf, raw_conf, T, set_size, K,
          verdict ("ACCEPT"/"ABSTAIN"), box, quantile, figure.
    Kerala keys: same shape, with cal_conf/T/quantile/set_size = None
          and verdict = "RAW" (no threshold is applied).
    """
    if _mode_key(mode) == MODE_KERALA:
        return _predict_kerala(image)

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
    lat0, lat1, lon0, lon1 = d["box"]
    if d.get("mode") == MODE_KERALA:
        return (
            f"### Prediction (Kerala mode)\n\n"
            f"- Predicted lat/lon: **{d['lat']:.4f}, {d['lon']:.4f}**\n"
            f"- Cell id: **{d['cell_id']}** (box lat [{lat0:.2f}, {lat1:.2f}], "
            f"lon [{lon0:.2f}, {lon1:.2f}])\n"
            f"- Raw confidence: **{d['raw_conf'] * 100:.2f}%** (uncalibrated)\n"
            f"- Conformal set: n/a — Kerala mode is uncalibrated\n"
            f"- Calibration: none — raw softmax argmax, no temperature applied\n"
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
.gradio-container {max-width: 1440px !important; background: #f5f7f8;}
.gradio-container label, .gradio-container label span {color: #243e53 !important;}
.gradio-container .block-title, .gradio-container legend {color: #243e53 !important;}
#hero {background: #132e42; padding: 30px 36px; border-radius: 14px; margin-bottom: 18px;}
#hero h1 {font-family: Georgia, serif; color: #fff; font-size: 38px; font-weight: 400; margin: 6px 0 10px;}
#hero p {color: #c4d8e2; font-size: 16px; max-width: 780px;}
.eyebrow {color: #7fd4cb; letter-spacing: 2px; font-size: 11px; font-weight: 700;}
#result {background: #fff; border: 1px solid #dbe4e9; border-radius: 12px; padding: 22px; min-height: 145px;}
#result h2 {font-family: Georgia, serif; font-size: 27px; color: #183b50; margin: 6px 0 12px;}
.result-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;}
.result-grid span {display: block; color: #617789; font-size: 12px; margin-bottom: 5px;}
.result-grid strong {font-size: 20px; color: #143b50;}
.result-note {color: #617789; font-size: 12px; margin: 14px 0 0;}
#run-button {background: #146f76; color: white; border: 0; min-height: 48px;}
#metrics {padding: 14px 20px; border-left: 3px solid #1c8589; background: #eaf2f3; margin: 14px 0;}
#metrics p {color: #345566; font-size: 14px;}
@media(max-width: 700px) {#hero {padding: 22px;} #hero h1 {font-size: 30px;} .result-grid {grid-template-columns: 1fr;} }
"""


def _result_html(d=None):
    if d is None:
        return ('<div class="eyebrow" style="color:#146f76">PREDICTION SUMMARY</div>'
                '<h2>Where was this photo taken?</h2>'
                '<p class="result-note">Upload a street scene or choose an example, then select Locate photo.</p>')
    kerala = d.get("mode") == MODE_KERALA
    confidence = d["raw_conf"] if kerala else d["cal_conf"]
    status = "Regional estimate" if kerala else ("Accepted estimate" if d["verdict"] == "ACCEPT" else "Uncertain estimate · abstained")
    calibrated = not kerala and d.get("temperature_fitted", False)
    note = ("Raw model score · uncalibrated; this is not a measured probability of correctness."
            if not calibrated else "Calibrated model score · the model abstains below 50% confidence.")
    return (
        f'<div class="eyebrow" style="color:#146f76">{status.upper()}</div>'
        f'<h2>{"Kerala" if kerala else "Global"} · predicted cell {d["cell_id"]}</h2>'
        '<div class="result-grid">'
        f'<div><span>Latitude</span><strong>{d["lat"]:.4f}°</strong></div>'
        f'<div><span>Longitude</span><strong>{d["lon"]:.4f}°</strong></div>'
        f'<div><span>{"Calibrated confidence" if calibrated else "Raw model score"}</span><strong>{confidence:.1%}</strong></div>'
        f'</div><p class="result-note">{note} The marker represents a cell centroid.</p>'
    )


def _empty_map(mode):
    fig = go.Figure(go.Scattergeo(lat=[], lon=[], showlegend=False))
    fig.update_layout(geo={"projection": {"type": "equirectangular"},
                           "showland": True, "showocean": True, "showcountries": True})
    if _mode_key(mode) == MODE_KERALA:
        fig.update_geos(lataxis_range=[7.7, 13.1], lonaxis_range=[72.6, 80.2])
    return _style_map(fig)


def _demo_theme(gr):
    return gr.themes.Soft(primary_hue="teal", neutral_hue="slate").set(
        body_background_fill_dark="#f5f7f8", body_text_color_dark="#243e53",
        body_text_color_subdued_dark="#617789", block_background_fill_dark="#ffffff",
        block_label_background_fill_dark="#eaf2f3", block_label_text_color_dark="#243e53",
        background_fill_primary_dark="#ffffff", background_fill_secondary_dark="#eef3f5",
        input_background_fill_dark="#ffffff", border_color_primary_dark="#dbe4e9",
        checkbox_label_background_fill_dark="#eaf2f3",
        checkbox_label_background_fill_selected_dark="#cfe9e7",
        checkbox_label_text_color_dark="#243e53",
        checkbox_label_text_color_selected_dark="#243e53",
    )


def build_ui():
    import gradio as gr

    def run_prediction(img, mode):
        if img is None:
            gr.Warning("Choose a photo before running the model.")
            return _empty_map(mode), _result_html(), "Upload a photo to see model details."
        try:
            d = predict_one(img, mode)
        except Exception as exc:
            raise gr.Error("Prediction could not complete. Check the model checkpoint and try again.") from exc
        return d["figure"], _result_html(d), _format_markdown(d)

    def reset_result(mode):
        return _empty_map(mode), _result_html(), "Run a prediction to inspect calibration and cell bounds."

    # Gradio 6 moved theme/css from Blocks to launch; retain older releases.
    style = {} if int(gr.__version__.split(".")[0]) >= 6 else {
        "css": DEMO_CSS, "theme": _demo_theme(gr),
    }
    with gr.Blocks(title="KPlaceNet | Visual geolocation", **style) as demo:
        gr.HTML('<div class="eyebrow">COMPUTER VISION / VISUAL GEOLOCATION</div>'
                '<h1>A place, from a photograph.</h1>'
                '<p>KPlaceNet learns geographic cues from street scenes. Explore a global model '
                'and a model specialized for Kerala, one photograph at a time.</p>', elem_id="hero")
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300):
                gr.Markdown("### 01 / Choose your view")
                mode_in = gr.Radio(choices=[(MODE_LABELS[MODE_KERALA], MODE_KERALA),
                                            (MODE_LABELS[MODE_GLOBAL], MODE_GLOBAL)],
                                   value=MODE_KERALA, label="Geographic scope")
                img_in = gr.Image(type="pil", label="Street scene", height=340,
                                  sources=["upload", "clipboard"])
                run = gr.Button("Locate photo →", variant="primary", elem_id="run-button")
                gr.Markdown("The model predicts one of 300 geographic cells. Its marker shows the cell's center, so estimates are regional.")
            with gr.Column(scale=7, min_width=380):
                gr.Markdown("### 02 / Explore the prediction")
                map_out = gr.Plot(value=_empty_map(MODE_KERALA), label="Location map", show_label=False)
                result_out = gr.HTML(_result_html(), elem_id="result")
        gr.Markdown("**Kerala benchmark** · 42 km median error · 41% within 25 km · 357 held-out test photos. "
                    "These are dataset results, not the accuracy of the uploaded photo.", elem_id="metrics")
        gr.Markdown("### 03 / Try a demonstration photo")
        gr.Markdown("Examples select their matching model. Select **Locate photo** to reveal the prediction. "
                    "These are curated examples, including a failure case; they do not represent average performance.")
        groups = [
            ("Kerala examples", MODE_KERALA, ROOT / "data/kerala_osv5m/test/images", [
                ("319636679783973.jpg", "Idukki · curated hit"),
                ("1083931132408866.jpg", "Kottayam · curated hit"),
                ("313393843511505.jpg", "Kochi · curated hit"),
                ("288990592896618.jpg", "Thiruvananthapuram · curated hit"),
                ("905069160348732.jpg", "Far south · failure case")]),
            ("Global examples", MODE_GLOBAL, ROOT / "data/osv5m_test/images", [
                ("3002826110043032.jpg", "Rajasthan · curated hit"),
                ("271239751462351.jpg", "Rajasthan · curated hit"),
                ("905946183585986.jpg", "United Kingdom · curated hit"),
                ("1472138156457085.jpg", "Ecuador · curated hit"),
                ("758042071550066.jpg", "Kerala · global failure case")]),
        ]
        example_label = gr.Textbox(label="Selected demonstration", interactive=False)
        for label, mode, folder, photos in groups:
            examples = [[str(folder / filename), mode, caption]
                        for filename, caption in photos if (folder / filename).exists()]
            if examples:
                gr.Examples(examples=examples, inputs=[img_in, mode_in, example_label],
                            label=label, examples_per_page=5)
        with gr.Accordion("Research details · confidence, calibration & cell bounds", open=False):
            info_out = gr.Markdown("Run a prediction to inspect calibration and cell bounds.")
            gr.Markdown("Kerala uses raw softmax scores without calibration or an abstention threshold. "
                        "Global mode uses temperature scaling and conformal prediction. "
                        "Cell bounding boxes are geographic extents, not confidence intervals. "
                        "Distance error requires a known reference location.")
        gr.Markdown("**KPlaceNet** · ResNet-50 visual geolocation · Street imagery: OSV-5M (CC-BY-SA-4.0)")
        outputs = [map_out, result_out, info_out]
        run.click(run_prediction, inputs=[img_in, mode_in], outputs=outputs)
        img_in.change(reset_result, inputs=[mode_in], outputs=outputs)
        mode_in.change(reset_result, inputs=[mode_in], outputs=outputs)
    return demo


def launch_demo(**kwargs):
    """Launch with the classroom theme on both Gradio 5 and 6."""
    import gradio as gr

    if int(gr.__version__.split(".")[0]) >= 6:
        kwargs.setdefault("css", DEMO_CSS)
        kwargs.setdefault("theme", _demo_theme(gr))
    return build_ui().launch(**kwargs)


if __name__ == "__main__":
    launch_demo()

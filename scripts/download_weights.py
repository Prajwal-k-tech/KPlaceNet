"""Download the pretrained KPlaceNet weights from the GitHub release.

Weights are not stored in git (GitHub rejects files over 100 MB). This script
pulls them from the ``weights-v1`` release and writes each file to the path the
code expects under ``checkpoints/``. Existing non-empty files are skipped, so
it is safe to re-run.

Usage:
    python scripts/download_weights.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = "weights-v1"
BASE_URL = (
    "https://github.com/Asterisk-Hunter/KPlaceNet/releases/download/"
    f"{RELEASE_TAG}"
)

# Release asset name -> destination relative to the repo root.
ASSETS: dict[str, str] = {
    "l2_imagenet_1.0_layer4_last.pt": "checkpoints/l2_imagenet_1.0_layer4/last.pt",
    "l2_cells_full.json": "checkpoints/l2_cells_full.json",
    "l3_uncertainty_results.json": "checkpoints/l3_uncertainty_results.json",
    "l5_kerala_kmeans300_last.pt": "checkpoints/l5_kerala_kmeans300/last.pt",
    "l5_kerala_kmeans300_best.pt": "checkpoints/l5_kerala_kmeans300/best.pt",
    "l5_kerala_kmeans300_cells.json": "checkpoints/l5_kerala_kmeans300/cells.json",
    "l5_kerala_kmeans300_metrics.json": "checkpoints/l5_kerala_kmeans300/metrics_l5_kerala.json",
    "l5_kerala_kmeans300_results.json": "checkpoints/l5_kerala_kmeans300/results.json",
}

CHUNK_BYTES = 1 << 20  # 1 MiB


def _mb(num_bytes: float) -> str:
    return f"{num_bytes / 1e6:.1f} MB"


def download(asset: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.relative_to(ROOT)}")
        return

    url = f"{BASE_URL}/{asset}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"  {asset} -> {dest.relative_to(ROOT)}")

    request = urllib.request.Request(url, headers={"User-Agent": "KPlaceNet-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r    {_mb(done)} / {_mb(total)} ({100.0 * done / total:5.1f}%)",
                    end="",
                    flush=True,
                )

    size = partial.stat().st_size
    print()
    if total and size != total:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download ({size} of {total} bytes)")
    partial.replace(dest)
    print(f"    done ({_mb(size)})")


def main() -> int:
    print(f"Downloading weights from the '{RELEASE_TAG}' release into {ROOT / 'checkpoints'}")
    failures: list[tuple[str, str]] = []
    for asset, relative_dest in ASSETS.items():
        try:
            download(asset, ROOT / relative_dest)
        except (urllib.error.URLError, RuntimeError, OSError) as exc:
            failures.append((asset, str(exc)))
            print(f"    FAILED: {exc}")

    if failures:
        print("\nSome downloads failed:")
        for asset, err in failures:
            print(f"  {asset}: {err}")
        print("\nRetry with: python scripts/download_weights.py")
        return 1

    print("\nAll weights in place. Launch the demo with: python demo/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

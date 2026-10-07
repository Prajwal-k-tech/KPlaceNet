"""Dependency-free parsing of calibration artifacts used by the demo."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalibrationConfig:
    temperature: float
    conformal_quantile: float | None
    temperature_fitted: bool
    checkpoint_sha256: str | None = None

    @property
    def conformal_fitted(self) -> bool:
        return self.conformal_quantile is not None


def load_calibration(path: Path) -> CalibrationConfig:
    """Load a validated calibration file; missing files mean raw top-1 mode."""
    if not path.is_file():
        return CalibrationConfig(temperature=1.0, conformal_quantile=None,
                                 temperature_fitted=False)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read calibration artifact {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("calibration artifact must contain a JSON object")

    temperature_fitted = "temperature" in data
    try:
        temperature = float(data.get("temperature", 1.0))
    except (TypeError, ValueError) as error:
        raise ValueError("calibration temperature must be a finite positive number") from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration temperature must be a finite positive number")

    raw_quantile = data.get("conformal_quantile")
    if raw_quantile is None:
        quantile = None
    elif isinstance(raw_quantile, str) and raw_quantile.lower() in {"infinity", "+infinity", "inf", "+inf"}:
        quantile = math.inf
    else:
        try:
            quantile = float(raw_quantile)
        except (TypeError, ValueError) as error:
            raise ValueError("conformal_quantile must be in [0, 1], positive infinity, or null") from error
        if math.isnan(quantile) or quantile < 0 or (math.isfinite(quantile) and quantile > 1):
            raise ValueError("conformal_quantile must be in [0, 1], positive infinity, or null")
        if math.isinf(quantile) and quantile < 0:
            raise ValueError("conformal_quantile must be positive infinity when infinite")

    provenance = data.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("calibration provenance must be a JSON object")
    checkpoint_hash = provenance.get("checkpoint_sha256")
    if checkpoint_hash is not None:
        if (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in checkpoint_hash)):
            raise ValueError("provenance.checkpoint_sha256 must be a 64-character SHA-256 hex digest")

    return CalibrationConfig(
        temperature=temperature,
        conformal_quantile=quantile,
        temperature_fitted=temperature_fitted,
        checkpoint_sha256=checkpoint_hash.lower() if checkpoint_hash is not None else None,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_calibration_checkpoint(config: CalibrationConfig, checkpoint_hash: str) -> bool:
    """Reject a known mismatch; return false when the artifact has no hash."""
    if config.checkpoint_sha256 is None:
        return False
    if config.checkpoint_sha256 != checkpoint_hash.lower():
        raise ValueError("calibration artifact was fitted for a different checkpoint")
    return True

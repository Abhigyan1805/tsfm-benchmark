"""Zero-shot time-series foundation model wrappers with a strict license gate.

Two families are approved by policy:

* **TimesFM 2.5 only** -- ``google/timesfm-2.5-200m-pytorch`` (Apache-2.0).
  TimesFM 3.0 weights ship under the ``timesfm-non-commercial-license-v1.0``
  and are refused outright, including by accident of a custom ``model_id``.
* **Chronos-Bolt only** -- the four ``amazon/chronos-bolt-*`` checkpoints whose
  Apache-2.0 license has been verified per model card. Any other Chronos id is
  refused until its license is verified and added to the table below.

Weights are never downloaded unless ``TSBENCH_ALLOW_MODEL_DOWNLOAD=1``; both
the license check and the env gate run before any import of a heavyweight
backend. The pinned revisions and license ids are recorded in :meth:`info`.

Heavy dependencies (``torch``, ``timesfm``, ``chronos``) are imported lazily
inside the wrappers so this package stays importable on a stdlib-only host.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from tsbench.base import Forecaster as Forecaster
from tsbench.base import ModelInfo as ModelInfo

DOWNLOAD_ENV = "TSBENCH_ALLOW_MODEL_DOWNLOAD"
PERMISSIVE_LICENSES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"})


def as_float_array(values: list[float]) -> Any:
    """Return the contract's 1-D numpy array, or a list without numpy."""
    try:
        import numpy as np
    except ImportError:
        return list(values)
    return np.asarray(values, dtype=float)


class WeightPolicyError(PermissionError):
    """Raised when weights are blocked by the license/download gate."""


class DownloadNotAllowed(WeightPolicyError):
    """Weights need the ``TSBENCH_ALLOW_MODEL_DOWNLOAD=1`` opt-in."""


class UnverifiedWeights(WeightPolicyError):
    """Weight license is not verified permissive (or the id is off-policy)."""


@dataclass(frozen=True)
class VerifiedWeights:
    """A checkpoint whose permissive license was verified against its card."""

    model_id: str
    license: str
    revision: str
    source: str
    verified_on: str
    params: int | None = None


# Per-model-card verification (2026-09-19): license apache-2.0, ungated.
TIMESFM_APPROVED: dict[str, VerifiedWeights] = {
    "google/timesfm-2.5-200m-pytorch": VerifiedWeights(
        model_id="google/timesfm-2.5-200m-pytorch",
        license="apache-2.0",
        revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        source="https://huggingface.co/google/timesfm-2.5-200m-pytorch",
        verified_on="2026-09-19",
        params=231_289_280,
    ),
}

CHRONOS_VERIFIED: dict[str, VerifiedWeights] = {
    "amazon/chronos-bolt-tiny": VerifiedWeights(
        model_id="amazon/chronos-bolt-tiny",
        license="apache-2.0",
        revision="a0e552de83495b5c28c14c71c374f3e33280b340",
        source="https://huggingface.co/amazon/chronos-bolt-tiny",
        verified_on="2026-09-19",
        params=8_652_672,
    ),
    "amazon/chronos-bolt-mini": VerifiedWeights(
        model_id="amazon/chronos-bolt-mini",
        license="apache-2.0",
        revision="251268337516a88e253628c43e1d26ec577b376b",
        source="https://huggingface.co/amazon/chronos-bolt-mini",
        verified_on="2026-09-19",
        params=21_236_096,
    ),
    "amazon/chronos-bolt-small": VerifiedWeights(
        model_id="amazon/chronos-bolt-small",
        license="apache-2.0",
        revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",
        source="https://huggingface.co/amazon/chronos-bolt-small",
        verified_on="2026-09-19",
        params=47_718_016,
    ),
    "amazon/chronos-bolt-base": VerifiedWeights(
        model_id="amazon/chronos-bolt-base",
        license="apache-2.0",
        revision="5d9f166d69f47aef3401367a7b842e78fe97b121",
        source="https://huggingface.co/amazon/chronos-bolt-base",
        verified_on="2026-09-19",
        params=205_292_928,
    ),
}


def download_allowed() -> bool:
    """True when the operator opted into downloading pretrained weights."""
    return os.environ.get(DOWNLOAD_ENV) == "1"


def clear_backend_caches() -> None:
    """Drop every process-level TSFM backend/compile cache."""
    from tsbench.models.tsfm.chronos import clear_backend_cache as _chronos
    from tsbench.models.tsfm.timesfm import clear_backend_cache as _timesfm

    _chronos()
    _timesfm()


def require_weights_allowed(model_id: str, weights: VerifiedWeights) -> None:
    """Enforce the license gate, then the download opt-in, before loading."""
    if weights.license not in PERMISSIVE_LICENSES:
        raise UnverifiedWeights(
            f"refusing {model_id!r}: license {weights.license!r} is not in the "
            f"verified permissive set {sorted(PERMISSIVE_LICENSES)}"
        )
    if not download_allowed():
        raise DownloadNotAllowed(
            f"refusing to load weights for {model_id!r}: set "
            f"{DOWNLOAD_ENV}=1 to allow pretrained checkpoint download/load "
            "(see docs/colab-handoff.md)"
        )


def check_timesfm_model_id(model_id: str) -> VerifiedWeights:
    """Return the approved TimesFM 2.5 entry or refuse the checkpoint."""
    if model_id in TIMESFM_APPROVED:
        return TIMESFM_APPROVED[model_id]
    lowered = model_id.lower()
    if "timesfm-3" in lowered or "timesfm_3" in lowered or "timesfm3" in lowered:
        raise UnverifiedWeights(
            f"refusing {model_id!r}: TimesFM 3.0 pretrained weights are "
            "distributed under timesfm-non-commercial-license-v1.0 and are "
            "restricted to non-commercial, non-production use; only the "
            "TimesFM 2.5 line is approved (google/timesfm-2.5-200m-pytorch)"
        )
    raise UnverifiedWeights(
        f"refusing {model_id!r}: not an approved TimesFM checkpoint; approved: "
        f"{sorted(TIMESFM_APPROVED)} (TimesFM 2.5 line only)"
    )


def check_chronos_model_id(model_id: str) -> VerifiedWeights:
    """Return the verified Chronos-Bolt entry or refuse the checkpoint."""
    if model_id in CHRONOS_VERIFIED:
        return CHRONOS_VERIFIED[model_id]
    raise UnverifiedWeights(
        f"refusing {model_id!r}: license not verified permissive; verified "
        f"Chronos-Bolt checkpoints: {sorted(CHRONOS_VERIFIED)}"
    )


def flatten_floats(values: Any) -> list[float]:
    """Flatten array/tensor-like output to a finite ``list[float]``."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    flat: list[float] = []

    def _walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for sub in item:
                _walk(sub)
            return
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"non-finite value in model output: {item!r}")
        flat.append(number)

    _walk(values)
    return flat


def median_point(forecast: Any) -> list[float]:
    """Reduce a Chronos-style quantile forecast to a point forecast.

    Accepts a torch tensor or nested lists shaped ``(batch, quantiles, h)``,
    ``(batch, h)`` or ``(h,)``. For quantile forecasts the median quantile of
    the first batch item is used.
    """
    values = forecast.tolist() if hasattr(forecast, "tolist") else forecast
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"cannot read a forecast from {values!r}")
    first = values[0]
    if isinstance(first, (list, tuple)):
        if first and isinstance(first[0], (list, tuple)):
            quantiles = list(first)
            return flatten_floats(quantiles[len(quantiles) // 2])
        return flatten_floats(first)
    return flatten_floats(values)


__all__ = [
    "CHRONOS_VERIFIED",
    "DOWNLOAD_ENV",
    "PERMISSIVE_LICENSES",
    "TIMESFM_APPROVED",
    "ChronosBolt",
    "DownloadNotAllowed",
    "Forecaster",
    "ModelInfo",
    "TimesFM25",
    "UnverifiedWeights",
    "VerifiedWeights",
    "WeightPolicyError",
    "as_float_array",
    "check_chronos_model_id",
    "check_timesfm_model_id",
    "clear_backend_caches",
    "download_allowed",
    "flatten_floats",
    "median_point",
    "require_weights_allowed",
]


def __getattr__(name: str) -> Any:
    if name == "TimesFM25":
        from tsbench.models.tsfm.timesfm import TimesFM25

        return TimesFM25
    if name == "ChronosBolt":
        from tsbench.models.tsfm.chronos import ChronosBolt

        return ChronosBolt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

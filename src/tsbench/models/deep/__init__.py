"""Seeded deep forecasters (LSTM, small Transformer) for the GPU tier.

Heavy dependencies (``torch``) are imported lazily inside
:meth:`LSTMForecaster.fit` / :meth:`TransformerForecaster.fit`, so this package
stays importable in a stdlib-only worker environment. Real training runs on a
GPU in Colab; see ``docs/colab-handoff.md``.

The shared helpers below are pure Python on purpose: they carry the
windowing/seeding/shape contract that the CPU-only tests check without torch.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from tsbench.base import Forecaster as Forecaster
from tsbench.base import ModelInfo as ModelInfo


def as_float_array(values: list[float]) -> Any:
    """Return the contract's 1-D numpy array, or a list without numpy."""
    try:
        import numpy as np
    except ImportError:
        return list(values)
    return np.asarray(values, dtype=float)


def require_torch() -> Any:
    """Import torch lazily, with an actionable error when it is missing."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "deep forecasters need torch; install the GPU extra "
            "(pip install torch) or run this tier in Colab "
            "(docs/colab-handoff.md)"
        ) from exc
    return torch


def resolve_device(torch_module: Any, device: str | None = None) -> str:
    """Resolve the training/inference device (CUDA when available)."""
    if device:
        return str(device)
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def set_seed(seed: int) -> int:
    """Seed every RNG that is importable: stdlib, numpy and torch."""
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed


def to_float_list(values: Any) -> list[float]:
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
            raise ValueError(f"non-finite value in forecast output: {item!r}")
        flat.append(number)

    _walk(values)
    return flat


def normalize(values: Sequence[float]) -> tuple[list[float], float, float]:
    """Z-normalize a series, returning ``(normalized, mean, std)``."""
    series = [float(value) for value in values]
    if not series:
        raise ValueError("cannot normalize an empty series")
    mean = sum(series) / len(series)
    variance = sum((value - mean) ** 2 for value in series) / len(series)
    std = math.sqrt(variance)
    if std < 1e-12:
        std = 1.0
    return [(value - mean) / std for value in series], mean, std


def make_windows(
    values: Sequence[float], context: int, horizon: int
) -> tuple[list[list[float]], list[list[float]]]:
    """Split a series into supervised ``(context, horizon)`` windows."""
    context = int(context)
    horizon = int(horizon)
    if context < 1 or horizon < 1:
        raise ValueError("context and horizon must be at least 1")
    series = [float(value) for value in values]
    if len(series) < context + horizon:
        raise ValueError(
            f"need at least context + horizon = {context + horizon} points, "
            f"got {len(series)}"
        )
    inputs: list[list[float]] = []
    targets: list[list[float]] = []
    for start in range(len(series) - context - horizon + 1):
        inputs.append(series[start : start + context])
        targets.append(series[start + context : start + context + horizon])
    return inputs, targets


__all__ = [
    "Forecaster",
    "LSTMForecaster",
    "ModelInfo",
    "TransformerForecaster",
    "as_float_array",
    "make_windows",
    "normalize",
    "require_torch",
    "resolve_device",
    "set_seed",
    "to_float_list",
]


def __getattr__(name: str) -> Any:
    if name == "LSTMForecaster":
        from tsbench.models.deep.lstm import LSTMForecaster

        return LSTMForecaster
    if name == "TransformerForecaster":
        from tsbench.models.deep.transformer import TransformerForecaster

        return TransformerForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

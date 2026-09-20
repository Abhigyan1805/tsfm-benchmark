"""Zero-shot Chronos-Bolt wrapper (verified-permissive license gate)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tsbench.models.tsfm import (
    CHRONOS_VERIFIED,
    DOWNLOAD_ENV,
    Forecaster,
    ModelInfo,
    as_float_array,
    check_chronos_model_id,
    download_allowed,
    flatten_floats,
    median_point,
    require_weights_allowed,
)

__all__ = ["ChronosBolt", "clear_backend_cache"]

DEFAULT_MODEL_ID = "amazon/chronos-bolt-base"

# Process-level cache keyed by pinned checkpoint: the rolling-origin backtest
# builds a fresh model per window, so the loaded pipeline must be shared or the
# checkpoint would be reloaded for every one of hundreds of windows.
_BACKEND_CACHE: dict[tuple[str, str], Any] = {}


def clear_backend_cache() -> None:
    """Drop the process-level Chronos-Bolt pipeline cache."""
    _BACKEND_CACHE.clear()


def _load_pipeline_class() -> Any:
    try:
        from chronos import ChronosBoltPipeline
    except ImportError:
        try:
            from chronos import BaseChronosPipeline as ChronosBoltPipeline
        except ImportError as exc:
            raise ImportError(
                "ChronosBolt needs the chronos-forecasting package; install it "
                "with pip install chronos-forecasting in the GPU session "
                "(docs/colab-handoff.md)"
            ) from exc
    return ChronosBoltPipeline


def _context_tensor(values: list[float]) -> Any:
    """Real pipelines get a torch tensor; injected fakes work without torch."""
    try:
        import torch
    except ImportError:
        return list(values)
    return torch.tensor(values, dtype=torch.float32)


class ChronosBolt(Forecaster):
    """Zero-shot Chronos-Bolt forecaster.

    ``fit`` only stores the context; ``predict`` loads a checkpoint from the
    verified Apache-2.0 table behind the license/download gate and returns the
    median quantile as the point forecast. Passing ``backend`` injects an
    already-loaded pipeline object (used by the offline tests).
    """

    name = "chronos_bolt"
    zero_shot = True
    family = "tsfm"
    backend = "chronos_bolt"

    def __init__(
        self,
        name: str | None = None,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        backend: Any = None,
        max_context: int = 512,
    ) -> None:
        if name is not None:
            self.name = str(name)
        weights = check_chronos_model_id(str(model_id))
        if int(max_context) < 1:
            raise ValueError("max_context must be at least 1")
        self.model_id = weights.model_id
        self.revision = str(revision) if revision else weights.revision
        self.license = weights.license
        self.license_source = weights.source
        self.verified_on = weights.verified_on
        self._max_context = int(max_context)
        self._backend = backend
        self._context_values: list[float] | None = None

    def _ensure_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        key = (self.model_id, self.revision)
        cached = _BACKEND_CACHE.get(key)
        if cached is not None:
            self._backend = cached
            return self._backend
        weights = CHRONOS_VERIFIED[self.model_id]
        require_weights_allowed(self.model_id, weights)
        pipeline_class = _load_pipeline_class()
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "ChronosBolt needs torch; install it in the GPU session "
                "(docs/colab-handoff.md)"
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._backend = pipeline_class.from_pretrained(
            self.model_id,
            revision=self.revision,
            device_map=device,
            torch_dtype=torch.float32,
        )
        _BACKEND_CACHE[key] = self._backend
        return self._backend

    def fit(self, y: Sequence[float] | Any, **kwargs: Any) -> ChronosBolt:
        """Store the zero-shot context (last ``max_context`` points)."""
        values = flatten_floats(y)
        if not values:
            raise ValueError("cannot fit on an empty series")
        self._context_values = values[-self._max_context :]
        return self

    def predict(self, h: int, **kwargs: Any) -> Any:
        """Forecast ``h`` steps zero-shot from the fitted context."""
        if self._context_values is None:
            raise RuntimeError("fit() must be called before predict()")
        horizon = int(h)
        if horizon < 1:
            raise ValueError("h must be at least 1")
        backend = self._ensure_backend()
        forecast = backend.predict(
            _context_tensor(self._context_values), prediction_length=horizon
        )
        points = median_point(forecast)
        if len(points) < horizon:
            raise ValueError(
                f"Chronos-Bolt returned {len(points)} points for horizon {horizon}"
            )
        return as_float_array(points[:horizon])

    def info(self) -> ModelInfo:
        weights = CHRONOS_VERIFIED[self.model_id]
        return ModelInfo(
            name=self.name,
            family=self.family,
            zero_shot=self.zero_shot,
            params=weights.params,
            license=self.license,
            revision=self.revision,
            extra={
                "backend": self.backend,
                "model_id": self.model_id,
                "license_verified": True,
                "license_source": self.license_source,
                "verified_on": self.verified_on,
                "download_env": DOWNLOAD_ENV,
                "download_allowed": download_allowed(),
                "weights_loaded": self._backend is not None,
                "context_len": len(self._context_values or []),
                "max_context": self._max_context,
            },
        )

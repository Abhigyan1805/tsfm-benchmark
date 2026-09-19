"""Zero-shot TimesFM 2.5 wrapper (license-gated, TimesFM 2.5 line only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tsbench.models.tsfm import (
    DOWNLOAD_ENV,
    TIMESFM_APPROVED,
    Forecaster,
    ModelInfo,
    as_float_array,
    check_timesfm_model_id,
    download_allowed,
    flatten_floats,
    require_weights_allowed,
)

__all__ = ["TimesFM25"]

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
_INPUT_PATCH = 32
_OUTPUT_PATCH = 128


class TimesFM25(Forecaster):
    """Zero-shot TimesFM 2.5 forecaster.

    ``fit`` only stores the context (no weights are touched); ``predict``
    loads the pinned checkpoint behind the license/download gate and compiles
    the decoder for the requested horizon. Passing ``backend`` injects an
    already-loaded object (used by the offline tests) and skips loading.
    """

    name = "timesfm25"
    zero_shot = True
    family = "tsfm"
    backend = "timesfm"

    def __init__(
        self,
        name: str | None = None,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        backend: Any = None,
        max_context: int = 512,
        max_horizon: int = 128,
        use_quantile_head: bool = False,
    ) -> None:
        if name is not None:
            self.name = str(name)
        weights = check_timesfm_model_id(str(model_id))
        if int(max_context) < 1:
            raise ValueError("max_context must be at least 1")
        if int(max_horizon) < 1:
            raise ValueError("max_horizon must be at least 1")
        self.model_id = weights.model_id
        self.revision = str(revision) if revision else weights.revision
        self.license = weights.license
        self.license_source = weights.source
        self.verified_on = weights.verified_on
        context = max(_INPUT_PATCH, int(max_context))
        self._max_context = (
            (context + _INPUT_PATCH - 1) // _INPUT_PATCH
        ) * _INPUT_PATCH
        self._max_horizon = int(max_horizon)
        self._use_quantile_head = bool(use_quantile_head)
        self._backend = backend
        self._module: Any = None
        self._compiled_horizon: int | None = None
        self._context_values: list[float] | None = None

    def _ensure_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        weights = TIMESFM_APPROVED[self.model_id]
        require_weights_allowed(self.model_id, weights)
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "TimesFM25 needs the timesfm package; install it with "
                "pip install 'timesfm[torch]' in the GPU session "
                "(docs/colab-handoff.md)"
            ) from exc
        model_class: Any = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if model_class is None:
            try:
                from timesfm.timesfm_2p5.timesfm_2p5_torch import (
                    TimesFM_2p5_200M_torch,
                )
            except ImportError as exc:
                raise ImportError(
                    "installed timesfm package does not expose "
                    "TimesFM_2p5_200M_torch (TimesFM 2.5 line); TimesFM 3.0 "
                    "weights are non-commercial and unsupported"
                ) from exc
            model_class = TimesFM_2p5_200M_torch
        self._backend = model_class.from_pretrained(
            self.model_id,
            revision=self.revision,
            force_download=False,
        )
        self._module = timesfm
        return self._backend

    def _compile_for(self, horizon: int) -> None:
        target = max(int(horizon), self._max_horizon)
        target = ((target + _OUTPUT_PATCH - 1) // _OUTPUT_PATCH) * _OUTPUT_PATCH
        if self._compiled_horizon is not None and self._compiled_horizon >= target:
            return
        config = self._module.ForecastConfig(
            max_context=self._max_context,
            max_horizon=target,
            normalize_inputs=True,
            use_continuous_quantile_head=bool(
                self._use_quantile_head and target <= 1024
            ),
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
        self._backend.compile(config)
        self._compiled_horizon = target

    def fit(self, y: Sequence[float] | Any, **kwargs: Any) -> TimesFM25:
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
        if self._module is not None:
            self._compile_for(horizon)
        inputs = [as_float_array(self._context_values)]
        points, _quantiles = backend.forecast(horizon=horizon, inputs=inputs)
        flat = flatten_floats(points)
        if len(flat) < horizon:
            raise ValueError(
                f"TimesFM returned {len(flat)} points for horizon {horizon}"
            )
        return as_float_array(flat[:horizon])

    def info(self) -> ModelInfo:
        weights = TIMESFM_APPROVED[self.model_id]
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
                "max_horizon": self._max_horizon,
                "compiled_horizon": self._compiled_horizon,
            },
        )

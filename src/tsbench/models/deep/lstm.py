"""Seeded LSTM forecaster (one-step-ahead, autoregressive decoding)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tsbench.models.deep import (
    Forecaster,
    ModelInfo,
    as_float_array,
    make_windows,
    normalize,
    require_torch,
    resolve_device,
    set_seed,
    to_float_list,
)

__all__ = ["LSTMForecaster"]


def _build_net(torch_module: Any, hidden: int, layers: int) -> Any:
    """Build a tiny LSTM + linear head (defined after the lazy torch import)."""
    nn = torch_module.nn

    class _LSTMNet(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: Any) -> Any:
            outputs, _ = self.lstm(x)
            return self.head(outputs[:, -1, :])

    return _LSTMNet(hidden, layers)


class LSTMForecaster(Forecaster):
    """Small seeded LSTM trained for a fixed budget of epochs.

    Training is one-step-ahead on z-normalized windows; :meth:`predict`
    decodes autoregressively for any horizon. ``fit`` and ``predict`` import
    torch lazily, so the class is importable on a CPU-only, stdlib-only host.
    """

    name = "lstm"
    zero_shot = False
    family = "deep"
    backend = "torch"

    def __init__(
        self,
        name: str | None = None,
        *,
        context: int = 32,
        hidden: int = 32,
        layers: int = 1,
        epochs: int = 50,
        lr: float = 1e-2,
        batch_size: int = 64,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        if name is not None:
            self.name = str(name)
        if int(context) < 2:
            raise ValueError("context must be at least 2")
        if int(hidden) < 1:
            raise ValueError("hidden must be at least 1")
        if int(layers) < 1:
            raise ValueError("layers must be at least 1")
        if int(epochs) < 1:
            raise ValueError("epochs must be at least 1")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0.0 < float(lr) <= 1.0:
            raise ValueError("lr must be in (0, 1]")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        self._context = int(context)
        self._hidden = int(hidden)
        self._layers = int(layers)
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._device_spec = device
        self._device: str | None = None
        self._model: Any = None
        self._stats: tuple[float, float] | None = None
        self._history: list[float] | None = None
        self._n_train: int | None = None
        self._param_count: int | None = None

    def _config(self) -> dict[str, Any]:
        return {
            "context": self._context,
            "hidden": self._hidden,
            "layers": self._layers,
            "epochs": self._epochs,
            "lr": self._lr,
            "batch_size": self._batch_size,
            "seed": self._seed,
        }

    def fit(self, y: Sequence[float] | Any, **kwargs: Any) -> LSTMForecaster:
        """Train on a univariate series with an explicit, fixed budget."""
        torch = require_torch()
        set_seed(self._seed)
        series = to_float_list(y)
        if len(series) < self._context + 1:
            raise ValueError(
                f"need at least context + 1 = {self._context + 1} points, "
                f"got {len(series)}"
            )
        normalized, mean, std = normalize(series)
        inputs, targets = make_windows(normalized, self._context, 1)
        device = resolve_device(torch, self._device_spec)

        net = _build_net(torch, self._hidden, self._layers).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self._lr)
        loss_fn = torch.nn.MSELoss()
        x_tensor = torch.tensor(inputs, dtype=torch.float32, device=device).unsqueeze(
            -1
        )
        y_tensor = torch.tensor(
            [target[0] for target in targets], dtype=torch.float32, device=device
        )
        generator = torch.Generator().manual_seed(self._seed)
        n_windows = len(inputs)
        for _ in range(self._epochs):
            permutation = torch.randperm(n_windows, generator=generator).to(device)
            for start in range(0, n_windows, self._batch_size):
                batch = permutation[start : start + self._batch_size]
                optimizer.zero_grad()
                loss = loss_fn(net(x_tensor[batch]).squeeze(-1), y_tensor[batch])
                loss.backward()
                optimizer.step()
        net.eval()

        self._model = net
        self._device = device
        self._stats = (mean, std)
        self._history = normalized[-self._context :]
        self._n_train = n_windows
        self._param_count = sum(parameter.numel() for parameter in net.parameters())
        return self

    def predict(self, h: int, **kwargs: Any) -> Any:
        """Forecast ``h`` steps autoregressively on the trained LSTM."""
        torch = require_torch()
        if self._model is None or self._history is None or self._stats is None:
            raise RuntimeError("fit() must be called before predict()")
        horizon = int(h)
        if horizon < 1:
            raise ValueError("h must be at least 1")
        mean, std = self._stats
        window = list(self._history)
        decoded: list[float] = []
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(
                    [window[-self._context :]],
                    dtype=torch.float32,
                    device=self._device,
                ).unsqueeze(-1)
                step = float(self._model(x).squeeze().item())
                decoded.append(step)
                window.append(step)
        return as_float_array([value * std + mean for value in decoded])

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            family=self.family,
            zero_shot=self.zero_shot,
            params=self._param_count,
            license="Apache-2.0",
            revision="repo-local:0.0.1",
            extra={
                "backend": self.backend,
                "device": self._device or self._device_spec or "auto",
                "trained": self._model is not None,
                "n_train_windows": self._n_train,
                "config": self._config(),
            },
        )

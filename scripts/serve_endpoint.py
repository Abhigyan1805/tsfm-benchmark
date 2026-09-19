#!/usr/bin/env python3
"""Live forecast endpoint for the Colab GPU route (see docs/colab-handoff.md).

Runs a stdlib-only HTTP JSON server inside a GPU session and reuses
``scripts/colab_run.py`` for job execution and content-keyed caching, so a
repeated request (or a retried one after a tunnel drop) never re-measures a job
whose result already exists on disk.

Endpoints:

* ``GET  /healthz``  -- liveness/GPU/gate summary; no auth, no side effects.
* ``POST /forecast`` -- run one job object or ``{"jobs": [...]}``. Requires the
  bearer token. Returns ``{"results": [...]}`` with the same payload the batch
  route writes to ``<cache-key>.json``.

The endpoint is intentionally small: the operator starts it in Colab, exposes it
through a tunnel (``cloudflared``/``ngrok``), and drives it from the D-drive
machine. All the weight-download policy still lives in the model wrappers; this
server only lifts the transport.

Examples:
    python scripts/serve_endpoint.py --out results/serve --token "$TOKEN"
    curl -s -H "Authorization: Bearer $TOKEN" localhost:8000/healthz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ENDPOINT_TOKEN_ENV = "TSBENCH_ENDPOINT_TOKEN"
DEFAULT_MAX_BODY = 8 * 1024 * 1024


def _load_colab_run() -> Any:
    """Load the sibling ``colab_run`` module without a package import."""
    spec = importlib.util.spec_from_file_location(
        "colab_run", SCRIPT_DIR / "colab_run.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT_DIR / 'colab_run.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


colab_run = _load_colab_run()


def extract_jobs(payload: Any) -> list[dict[str, Any]]:
    """Accept a single job, ``{"jobs": [...]}`` or a bare list of jobs."""
    if isinstance(payload, list):
        jobs = payload
    elif isinstance(payload, dict) and "jobs" in payload:
        jobs = payload["jobs"]
    elif isinstance(payload, dict) and ("model" in payload or "entry" in payload):
        jobs = [payload]
    else:
        raise ValueError("body must be a job, a 'jobs' list, or a list of jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("job list is empty")
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("every job must be a JSON object")
    return jobs


def auth_ok(header: str | None, token: str) -> bool:
    """True when ``Authorization: Bearer <token>`` carries the shared secret."""
    if not token:
        return False
    if not header:
        return False
    scheme, _, value = header.partition(" ")
    return scheme.lower() == "bearer" and value.strip() == token


def resolve_token(cli_token: str | None) -> str:
    """Token from ``--token`` or the env contract; empty means fail closed."""
    return (cli_token or os.environ.get(ENDPOINT_TOKEN_ENV) or "").strip()


def run_job(job: dict[str, Any], out: Path, force: bool = False) -> dict[str, Any]:
    """Run (or reuse) a single job against the content-keyed cache."""
    series = colab_run.load_series(job)
    key = colab_run.cache_key(job, series)
    path = out / f"{key}.json"
    if path.exists() and not force:
        return {"cache_key": key, "status": "cached", "path": path.name}
    result = colab_run.execute_job(job, series, key)
    colab_run.write_json_atomic(path, result)
    return {"cache_key": key, "status": "ok", "path": path.name, "result": result}


def run_payload(
    payload: Any,
    out: Path,
    force: bool = False,
    max_jobs: int | None = None,
) -> list[dict[str, Any]]:
    """Execute every job in the request body, honoring the per-job cache."""
    jobs = extract_jobs(payload)
    if max_jobs is not None and len(jobs) > max_jobs:
        raise ValueError(f"request has {len(jobs)} jobs; max is {max_jobs}")
    out.mkdir(parents=True, exist_ok=True)
    return [run_job(job, out, force=force) for job in jobs]


def health_payload(out: Path) -> dict[str, Any]:
    """Minimal, secret-free liveness payload."""
    import importlib.util as ilu

    payload: dict[str, Any] = {
        "status": "ok",
        "service": "tsbench-forecast-endpoint",
        "runner_version": colab_run.RUNNER_VERSION,
        "python": platform.python_version(),
        "download_allowed": os.environ.get(colab_run.DOWNLOAD_ENV) == "1",
        "torch": ilu.find_spec("torch") is not None,
        "out": str(out),
    }
    if payload["torch"]:
        import torch

        payload["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["gpu"] = torch.cuda.get_device_name(0)
    return payload


class _Handler(BaseHTTPRequestHandler):
    server_version = "tsbench-endpoint/1"

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write(f"[endpoint] {self.address_string()} {fmt % args}\n")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?", 1)[0] != "/healthz":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._send_json(HTTPStatus.OK, health_payload(self.server.out))  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        server = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]
        if path != "/forecast":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not auth_ok(self.headers.get("Authorization"), server.token):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "bad or missing token"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > server.max_body:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "bad content length"}
            )
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
            return
        force = "force=1" in self.path
        try:
            results = run_payload(
                payload, server.out, force=force, max_jobs=server.max_jobs
            )
        except Exception as exc:  # noqa: BLE001 - surface any job failure as JSON
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._send_json(HTTPStatus.OK, {"count": len(results), "results": results})


def build_server(
    host: str,
    port: int,
    token: str,
    out: Path,
    max_body: int = DEFAULT_MAX_BODY,
    max_jobs: int | None = None,
) -> ThreadingHTTPServer:
    """Create the HTTP server with the request policy attached."""
    if not token:
        raise ValueError(
            f"refusing to start without a token; set {ENDPOINT_TOKEN_ENV} "
            "or pass --token"
        )
    server = ThreadingHTTPServer((host, port), _Handler)
    server.token = token  # type: ignore[attr-defined]
    server.out = out  # type: ignore[attr-defined]
    server.max_body = max_body  # type: ignore[attr-defined]
    server.max_jobs = max_jobs  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tsbench live forecast endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", required=True, help="result cache directory")
    parser.add_argument("--token", default=None, help=f"or set {ENDPOINT_TOKEN_ENV}")
    parser.add_argument("--max-body", type=int, default=DEFAULT_MAX_BODY)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token = resolve_token(args.token)
    server = build_server(
        args.host, args.port, token, out, args.max_body, args.max_jobs
    )
    health = health_payload(out)
    print(
        f"endpoint: http://{args.host}:{args.port} out={out} "
        f"download_allowed={health['download_allowed']} torch={health['torch']}",
        flush=True,
    )
    print("expose with: cloudflared tunnel --url http://localhost:8000", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nendpoint: stopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

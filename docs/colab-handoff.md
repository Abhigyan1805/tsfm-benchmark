# GPU compute handoff: deep + TSFM tiers

The GPU tiers run off the D-drive machine. Two compute routes are supported and
both are resumable:

| Route | Transport | When to use | Runner |
|---|---|---|---|
| **A — Colab live endpoint** | HTTP JSON over a tunnel | interactive sweeps, one-off jobs, quick iteration | `scripts/serve_endpoint.py` |
| **B — Kaggle batch kernel** | `kaggle` CLI push/poll/fetch | unattended batch runs, long jobs, no attached browser | `scripts/kaggle_run.py` |

Both routes execute the same job files through `scripts/colab_run.py`, so they
share the content cache key, the result JSON shape, and the license gate.

Tiers covered here are the GPU-slice registry keys: trained, seeded, fixed-budget
`lstm` and `transformer`, plus zero-shot `timesfm25` and `chronos_bolt`.
`configs/models.yaml` is the single owner of each key's entrypoint, license,
revision, and pinned `weights`; the verified TimesFM / Chronos-Bolt checkpoint
tables live in `src/tsbench/models/tsfm/__init__.py`.

## Job files

One JSON file with a `jobs` list. Each job names a registry key (or an explicit
`entry`), the series (inline or a path), the horizon, and constructor params:

```json
{
  "jobs": [
    {
      "model": "lstm",
      "series": [0.0, 0.1, 0.2, 0.3],
      "horizon": 12,
      "params": {"context": 32, "hidden": 16, "epochs": 50, "seed": 0}
    },
    {
      "model": "transformer",
      "series_path": "data/prepared/example.json",
      "horizon": 24,
      "params": {"context": 32, "max_horizon": 32, "epochs": 50, "seed": 0}
    },
    {"model": "timesfm25", "series_path": "data/prepared/example.csv", "horizon": 24, "params": {}},
    {"model": "chronos_bolt", "entry": "tsbench.models.tsfm.chronos:ChronosBolt",
     "series_path": "data/prepared/example.csv", "horizon": 24,
     "params": {"model_id": "amazon/chronos-bolt-base"}}
  ]
}
```

Series files are `.json` (a list, or `{"series": [...]}`) or `.csv` (numeric
column; pass `"column": "<name>"` for a named header column). The last
`horizon` points are held out; the model fits everything before them and the
runner records MAE/RMSE against the holdout.

Every job gets a **content cache key** (runner version + job spec + series
content). A result already on disk with the same key is skipped, so a re-run
after a disconnect, preemption, or runtime restart only fills the gap. Partial
writes never survive: `*.json.tmp` is replaced atomically.

## Env contract

| Variable | Required | Meaning |
|---|---|---|
| `TSBENCH_ALLOW_MODEL_DOWNLOAD=1` | yes for `timesfm25` / `chronos_bolt` | Explicit opt-in to load pretrained weights. Unset/`0` makes `predict` raise `DownloadNotAllowed` before any backend import. The batch kernel sets it for you; the endpoint inherits it from the session. |
| `TSBENCH_ENDPOINT_TOKEN` | yes for route A | Shared secret the endpoint requires as `Authorization: Bearer <token>`. The server refuses to start without it. |
| `HF_HOME` | optional | Relocate the Hugging Face cache (weights are ~1–2 GB total). |
| `CUDA_VISIBLE_DEVICES` | optional | Pin the GPU when a runtime has several. |

Trained deep models take `seed` in their constructor params; the runner records
it in the result and in the cache key.

---

## Route A — Colab live endpoint

### A1. Open the Colab GPU runtime (captain)

1. Open a new Colab notebook, then **Runtime → Change runtime type → T4 GPU**
   (an L4/A100 is better for the transformer tier; T4 handles all four models).
2. Confirm the accelerator is attached: `!nvidia-smi`.

### A2. Clone, install, start the endpoint (in Colab)

```sh
!git clone https://github.com/Abhigyan1805/tsfm-benchmark.git
%cd tsfm-benchmark
!git log --oneline -1            # record the commit in the run notes
!pip install -q 'timesfm[torch]' chronos-forecasting
```

```sh
import os
os.environ["TSBENCH_ALLOW_MODEL_DOWNLOAD"] = "1"
os.environ["TSBENCH_ENDPOINT_TOKEN"] = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
```

```sh
!python scripts/serve_endpoint.py --host 0.0.0.0 --port 8000 \
    --out results/serve-01 > /tmp/endpoint.log 2>&1 &
!sleep 3 && curl -s localhost:8000/healthz
```

`/healthz` returns `status`, `download_allowed`, `torch`/`cuda_available`, and
the runner version. `/forecast` is the job endpoint; it is token-gated.

### A3. Expose it through a tunnel

Colab has no public port, so tunnel `localhost:8000`. `cloudflared` quick
tunnels need no account:

```sh
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
!cloudflared tunnel --url http://localhost:8000 --no-autoupdate > /tmp/cloudflared.log 2>&1 &
!sleep 8 && grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cloudflared.log | head -1
```

The printed `https://<name>.trycloudflare.com` is the base URL. `ngrok http 8000`
works the same way when an ngrok token is configured.

### A4. Drive it from the D-drive machine

```sh
export TOKEN=REPLACE_WITH_A_LONG_RANDOM_SECRET
export BASE=https://<name>.trycloudflare.com
curl -s "$BASE/healthz"
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  --data @jobs.json "$BASE/forecast" > results/serve-01-response.json
```

The response is `{"count": N, "results": [...]}`; each entry carries the cache
key, status (`ok`/`cached`), and the full result. To persist the same
per-key files the batch route writes, use the stdlib client below (it also
dedupes against results you already ingested):

```python
import json, os, urllib.request
from pathlib import Path

base = os.environ["BASE"]
token = os.environ["TOKEN"]
jobs = json.loads(Path("jobs.json").read_text())
request = urllib.request.Request(
    f"{base}/forecast",
    data=json.dumps(jobs).encode(),
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
)
response = json.load(urllib.request.urlopen(request))
out = Path("results/serve-01")
out.mkdir(parents=True, exist_ok=True)
for entry in response["results"]:
    if entry["status"] == "ok":
        (out / f"{entry['cache_key']}.json").write_text(
            json.dumps(entry["result"], indent=2, sort_keys=True)
        )
print("saved", sum(e["status"] == "ok" for e in response["results"]), "results")
```

Re-POST the same `jobs.json` after a tunnel drop: jobs already cached in the
Colab session return `status: "cached"` without re-measuring. The endpoint also
keeps its own `<cache-key>.json` files under `results/serve-01/` in Colab, which
you can download with `google.colab.files` if you prefer the batch-style
artifacts.

### A5. Stop

`kill %1` stops the endpoint; ending the Colab runtime tears down the tunnel.

---

## Route B — Kaggle batch kernel

The `kaggle` CLI is installed and authenticated on the D-drive machine
(`kaggle config view` reports the username). Internet-enabled kernels are
verified working, so the kernel clones the repo and pip-installs the TSFM
packages at run time.

### B1. Preflight and plan (resumable, no network push)

```sh
kaggle config view                       # confirm username + auth
python scripts/kaggle_run.py preflight --jobs jobs.json --out results/kaggle-01
python scripts/kaggle_run.py plan      --jobs jobs.json --out results/kaggle-01 --ref HEAD
```

`plan` prints the pending/cached split. Jobs whose `<cache-key>.json` is already
in `--out` are cached and are not packed into the kernel.

### B2. Run the full route

```sh
REF="$(git rev-parse HEAD)"              # pin the exact commit the kernel clones
python scripts/kaggle_run.py run \
    --jobs jobs.json --out results/kaggle-01 --ref "$REF" \
    --pip 'timesfm[torch]' --pip chronos-forecasting
```

`run` executes, in order:

```sh
# what the runner prints and invokes (shown here for transparency)
kaggle kernels push -p results/kaggle/tsbench-gpu-<hash>
kaggle kernels status <owner>/tsbench-gpu-<hash>     # polled to a terminal state
kaggle kernels output <owner>/tsbench-gpu-<hash> -p results/kaggle/tsbench-gpu-<hash>/output
# then: ingest the fetched bundle into results/kaggle-01/<cache-key>.json
```

`run` never discards a partial batch: if the kernel ends in an error state it still
fetches and ingests whatever result JSON was written, reports the missing keys, and
exits non-zero, so a failed job costs only its own measurement. A pending-job timeout
aborts before fetching.

The generated kernel is self-contained: it base64-embeds the pending jobs,
clones `--repo` at `--ref` into `/tmp`, runs `scripts/colab_run.py run` against
`/kaggle/working/<run-name>`, and bundles the run as
`/kaggle/working/<run-name>.tar.gz` for download. The kernel metadata enables a
GPU (`--cpu` to disable) and internet (`--no-internet` to disable).

### B3. Individual steps (when you want manual control)

```sh
python scripts/kaggle_run.py build --jobs jobs.json --out results/kaggle-01 --ref "$REF" \
    --pip 'timesfm[torch]' --pip chronos-forecasting
python scripts/kaggle_run.py push   --workdir results/kaggle/tsbench-gpu-<hash>
python scripts/kaggle_run.py poll   --kernel <owner>/tsbench-gpu-<hash> --timeout 3600
python scripts/kaggle_run.py fetch  --kernel <owner>/tsbench-gpu-<hash> --dest results/kaggle/tsbench-gpu-<hash>/output
python scripts/kaggle_run.py ingest --dest results/kaggle/tsbench-gpu-<hash>/output --out results/kaggle-01
```

`run` is just those steps plus the pending-job filter, so a re-run after a
preemption skips cached jobs and re-pushes only the remainder. Add `--dry-run`
to any shelling step to print the exact commands without executing them.

### B4. Series data in the kernel

Inline `series` arrays need no extra setup. For `series_path` jobs, attach the
data as a Kaggle dataset and pass `--dataset-source owner/dataset`, then point
`series_path` at `/kaggle/input/<dataset>/...` (the kernel cannot read the
D drive). `--dataset-source` is repeatable.

### B5. Artifacts back in the repo

`ingest` copies every result JSON into `results/kaggle-01/` and writes
`results/kaggle-01/ingest-manifest.json` (ingested/skipped/missing keys). It
exits non-zero if any expected key is missing, so a partial download is caught.
`results/*` is gitignored by policy: the extracted run stays on the D drive as
raw evidence, and only the aggregates/summaries that the evaluation slice
produces are committed.

---

### B6. Experiment runs (the GPU tier's real sweep)

The job route above scores one holdout per job. The benchmark's GPU tier instead
runs the same rolling-origin experiment runner the CPU families used, so its
rows land on the exact frozen windows. The `experiments` subcommand packs
pending experiment configs into one self-contained kernel:

```sh
REF="$(git rev-parse HEAD)"          # pin the exact commit the kernel clones
python scripts/kaggle_run.py experiments \
    --config configs/experiments/gpu_probe.yaml \
    --out results/kaggle-gpu-probe --ref "$REF" \
    --pip 'timesfm[torch]' --pip chronos-forecasting --timeout 2400
```

Pin `--ref` to the branch tip that carries the process-level TSFM backend
cache (the wrappers cache the loaded checkpoint and the TimesFM compile per
process); the kernel asserts the clone contains it and refuses to run
otherwise. The default `$(git rev-parse HEAD)` is only safe when executed on
that branch, never on `main`. The GPU configs set `warmup: true`, so the
one-time checkpoint load and compile are primed before the first scored window
and do not skew its `latency_ms`.

The kernel clones `--repo` at `--ref`, installs the extra packages, fetches and
checksum-verifies the frozen dataset through `python -m
tsbench.data.build_catalog --materialize`, runs each pending config with
`python -m tsbench run`, and bundles the run directories as
`/kaggle/working/<run-name>/results.tar.gz`. A config that fails does not stop
the rest: every config that completed still lands, so a partial GPU run is
recoverable.

Resume is by `config_hash`: a config whose `run.json` is already under `--out`
is skipped and never packed, so a re-run after a preemption only measures the
gap. `--dry-run` builds and prints the kernel without pushing.

Recommended sequence for the primary sweep (after the probe succeeds):

```sh
python scripts/kaggle_run.py experiments \
    --config configs/experiments/gpu.yaml \
    --config configs/experiments/gpu_h48.yaml \
    --config configs/experiments/gpu_h96.yaml \
    --config configs/experiments/gpu_h192.yaml \
    --out results --ref "$REF" \
    --pip 'timesfm[torch]' --pip chronos-forecasting --timeout 10800
```

`make report` then merges `results/` with the committed `docs/telemetry/`
store, so the combined CPU + GPU summary and figures regenerate from committed
evidence.

## License policy (enforced in code)

- **TimesFM 2.5 only.** `google/timesfm-2.5-200m-pytorch` is Apache-2.0 and is
  the single approved TimesFM checkpoint. TimesFM 3.0 weights
  (`google/timesfm-3.0-pytorch`) ship under the
  `timesfm-non-commercial-license-v1.0` and are refused with a
  `UnverifiedWeights` error; any other TimesFM id is refused too.
- **Chronos-Bolt only, verified permissive.** The four `amazon/chronos-bolt-*`
  checkpoints are Apache-2.0 per their model cards and pinned by revision in
  `tsbench/models/tsfm/__init__.py`. Any other Chronos id (including
  `amazon/chronos-2`) is refused until its license is verified and added to the
  table.
- Even for approved checkpoints, weights are never loaded unless
  `TSBENCH_ALLOW_MODEL_DOWNLOAD=1`. `info()` always reports `model_id`,
  `revision`, `license`, and `license_source`.

## Cost note

GPU time is amortized, not metered per token. Record the runtime type and
session duration in the run manifest (the runner captures the environment per
run) so the cost/accuracy comparison can charge the deep + TSFM tiers honestly.

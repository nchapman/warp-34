# Automated four-engine quality suite — FP16 KV

`run_suite.py` is the checked-in process supervisor. `suite.toml` defines the
27B deployments. It runs **warp-34 → vLLM → llama.cpp → SGLang**, one server at a
time on one GPU, invoking the **unmodified upstream lm-eval CLI** separately for
IFEval and GSM8K. There is no custom prompt construction, answer extraction or
scoring. The default is **10 examples per task per engine (80 evaluated requests)**.

## One-time setup

Use Linux with Python 3.12, NVIDIA drivers/`nvidia-smi`, the
[locked evaluator environment](README.md#install-an-isolated-environment), and
separate installations of vLLM **0.31.0**, SGLang **0.5.19**, and llama.cpp
**v0.9.0** (`cd959bae99ff4bed22bef2604f45241c09bbb8c8`). Build warp-34 with CUDA.
The runner does not install engines, download weights, or modify their sources.

Set these paths once in the shell (or your own untracked environment file):

```sh
export PATH="/opt/cuda/bin:/usr/local/cuda/bin:$PATH"
export LM_EVAL="$PWD/benchmarks/quality/.venv/bin/lm-eval"
export W34="$PWD/target/release/w34"
export VLLM=/absolute/path/to/vllm-venv/bin/vllm
export SGLANG_PYTHON=/absolute/path/to/sglang-venv/bin/python
export LLAMA_SERVER=/absolute/path/to/llama.cpp/build/bin/llama-server
export NLTK_DATA="$HOME/nltk_data"

# Local snapshot directories, not repository names; use the exact revisions below.
export HGPTQ=/absolute/path/to/hgptq/snapshot
export GPTQ=/absolute/path/to/gptq/snapshot
export GGUF=/absolute/path/to/unsloth/snapshot
```

| Variable | Hugging Face repository | Revision |
|---|---|---|
| `HGPTQ` | `nchapman/Qwen3.8-27B-W4A16-HGPTQ` | `1de743eedefdb7422e6b171174c7080204f5b005` |
| `GPTQ` | `nchapman/Qwen3.8-27B-W4A16-GPTQ` | `0df81eadcfd59601cb82ccece2f16ea7d39244bf` |
| `GGUF` | `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |

The Unsloth directory must contain `Qwen3.8-27B-UD-Q4_K_XL.gguf` and
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`. Download pinned snapshots with the upstream
Hugging Face CLI if needed. Revisions in the TOML are declared provenance;
point the paths at the correct immutable snapshots. The runner hashes binaries
and small config files, **not every weight shard**. It cannot authenticate a
checkpoint merely from its path or the declared revision.

## Run

From the repository root:

```sh
# Validate interpolation and inspect manifest.json without starting a process.
mise run quality-run -- --dry-run \
  --output-dir benchmarks/quality/results/fp16-plan

# All four engines, 10 examples per task, GPU 0.
mise run quality-run -- \
  --output-dir "benchmarks/quality/results/$(date -u +%Y%m%dT%H%M%SZ)-fp16-smoke10"

# Select engines, a different GPU or port, or a larger native prefix.
mise run quality-run -- --engines warp llama --gpu 1 --port 18035 --limit 50 \
  --output-dir benchmarks/quality/results/fp16-smoke50

# Deliberate full-dataset evaluation: 541 IFEval + 1,319 GSM8K per engine.
mise run quality-run -- --full \
  --output-dir benchmarks/quality/results/fp16-full
```

Without mise, use `python3 benchmarks/quality/run_suite.py` with the same arguments.
Only selected engines' path variables are required. GPU UUIDs are also accepted.
Every output directory must be new, including dry runs; failed runs are never
silently overwritten or resumed. `--full` and `--limit` are mutually exclusive.
A dry run validates command expansion, not executable availability, model fit,
engine compatibility or GPU health.

## Controlled settings and remaining differences

All deployments use **FP16 attention K and V**, a **4,096-token context**, one
client request at a time, and the same native task/generation settings:

| Engine | KV configuration | Decoding mode |
|---|---|---|
| warp-34 | `--kv-format fp16 --kv-pages 256` | MTP-3 |
| vLLM | `--dtype half --kv-cache-dtype float16` | MTP-3 |
| llama.cpp | explicit `f16` K/V for both target and draft | Native MTP, max 3 draft tokens |
| SGLang | `--dtype float16 --kv-cache-dtype auto` | **Plain**, explicitly configured |

SGLang's pinned version documents `auto` as the model dtype; with explicit
`float16` model dtype this selects FP16 KV. Its native MTP configuration did not
fit the single 24GB GPU in the prior checks. The runner does **not** try MTP then
silently fall back. Its plain mode is visible in the manifest and summary.

Persistent recurrent SSM state is FP32: native warp-34 policy, explicit donor
flags, and llama.cpp's hybrid recurrent allocation. This is separate from
attention KV precision. Recurrent convolution storage/implementation is not
identical across engines (llama.cpp uses FP32 recurrent buffers, whereas the
other recipes use FP16 convolution storage). Weight quantization still differs:
HGPTQ vs GPTQ vs Unsloth GGUF. This controls **KV precision**, not every numerical
or implementation difference; it is not a pure engine-parity test.

No second GPU, transformer-layer CPU offload, reduced context or lower-precision
KV is introduced after a failure. llama.cpp uses `--fit off`, explicit all-layer
offload, and verbose startup logs; its ordinary CPU-mapped model buffers remain
visible. Inherited `LLAMA_ARG_*` variables are removed to prevent hidden overrides.
TQ4 remains warp-34's production policy, but belongs in a separately labeled KV
ablation, not this shared baseline. Historical mixed-KV outputs remain unchanged.

## Lifecycle, failures and evidence

- Reject an occupied localhost port or selected GPU before launching each server.
  A failed `nvidia-smi` check is an error, not permission to proceed.
- Wait for HTTP health **and** the expected model alias, with a bounded startup
  deadline; fail immediately if the server exits.
- Run native task commands serially. Check exit codes, the expected number of
  distinct examples for **each native filter**, and nonempty responses. GSM8K
  naturally emits two sample rows per document, one per filter.
- Stop only owned process groups, including remaining descendants, on success,
  failure, client timeout, Ctrl-C or SIGTERM. No global `pkill` or detached jobs.
  If cleanup cannot be established, stop the suite. SIGKILL/power loss cannot be
  handled; inspect GPU processes before restarting after those events.
- A failed engine remains failed; continue to the next engine only after cleanup.
  The command exits nonzero if any engine fails. Read `summary.json` for partial
  completion; `not_run` is not a pass.

The run directory contains the expanded command/environment manifest, copied
runner/TOML/native YAML/lock, per-engine server/client logs and receipts, native
results and samples, and `summary.json` with **unchanged native metrics**. It also
collects best-effort package identities, GPU allocation snapshots, cached dataset
refs and NLTK resource hashes. These diagnostics may include unavailable-field
errors; they do not replace evaluation success criteria. Dataset/NLTK bytes are
not pinned by the Python lock. Only a small environment whitelist is recorded,
not a full environment dump containing credentials.

Package versions and model snapshots must be installed as specified; the runner
records their identity but does not replace a package manager or prove that an
installed package has no local source edits. There is no aggregate quality score,
throughput ranking or statistical significance claim from a smoke run.

## Completed smoke qualification

`quality-fp16-suite-smoke10-001` ran the four-engine sequence without manual
intervention in 10m25s on the pod's single selected RTX 3090. All four engines
completed both tasks: **80 evaluated requests**, no empty responses, eight native
result bundles, and established cleanup for every engine. The GPU process list
was empty afterward. Local GPUs were not used.

| Engine | IFEval strict prompt | GSM8K strict | GSM8K flexible |
|---|---:|---:|---:|
| warp-34, FP16 KV | 10/10 | 8/10 | 8/10 |
| vLLM, FP16 KV | 9/10 | 8/10 | 9/10 |
| llama.cpp / Unsloth, FP16 KV | 10/10 | 9/10 | 9/10 |
| SGLang plain, FP16 KV | 9/10 | 8/10 | 9/10 |

These aggregate smoke scores match the previous mixed-KV checks; this tiny
prefix does not establish TQ4 parity. Native results, manifest, summary and
receipts are tracked in `qualification/quality-fp16-suite-smoke10-001/`.
`evidence.sha256` covers 47 files in the raw bundle retained under `results/`
and `/workspace/quality-fp16-suite-smoke10-001` on the pod. The bundle includes
its executed runner snapshot. While the GPU run was underway, review prompted
an optional-diagnostics read-error handling fix, covered by a failing-then-passing
CPU regression test; launch commands, task invocation and scoring did not change.

vLLM logged `kv_cache_dtype=float16`; SGLang logged allocated KV dtype
`torch.float16`; llama.cpp logged F16 K/V in both target and draft contexts.
warp-34 was launched with explicit `--kv-format fp16`. These are configuration
and startup checks, not tensor-level verification of every cache allocation.

## Tests

`mise run quality-check` runs CPU-only configuration tests and runner lifecycle
integration tests with a real local fake HTTP server and fake native-artifact
producer. These exercise orchestration, not model quality. GPU smoke qualification
is a separate invocation of the same checked-in runner, not a temporary pod script.

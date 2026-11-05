# Flash-Next HTTP serving campaign

Configuration: [flash-next-serving.toml](flash-next-serving.toml). This is a
standalone `w34perf` HTTP campaign, not a server launcher or a results report.
Qualification attempts are complete, including failed cells. Frozen profiles
retain their original **unverified** metadata; assessed outcomes are recorded
separately rather than retroactively changing profile digests. Both specialized
vLLM overlays and Strata have actual HTTP measurements and quality evidence. NInfer's known Flash artifact is a different, IQ2_XS-derived
precision cohort, excluded here rather than labeled architecturally unsupported.
See `qualification/reference-pins.json` for build identities and
[completion-results.md](completion-results.md) for the evidence ledger and
[standings.md](standings.md) for the practical performance assessment.

## Workloads and protocol

Default: **three repetitions**, one warmup request, concurrency/inflight one,
seed 34, temperature zero, thinking disabled in chat-template kwargs, and a
1,800-second request timeout. Confirm each server honors the template setting.
Output lengths are ceilings, not guaranteed lengths; EOS can stop generation.

| Case | Work per repetition | Output ceiling |
|---|---|---:|
| `short-chat` | 4 requests, text fixture context once | 256 |
| `prefill-context-r16` | 4 requests, context repeated 16 times | 64 |
| `code-context-r64` | 4 requests, context repeated 64 times | 256 |
| `prefix-primed-r16` | 4 measured requests with explicit priming | 64 |
| `coding-agent-replay` | 2 sequential four-turn recorded sessions | 128/turn |

The text fixture mixes technical explanations, incident review, and coding
prompts. Its repeated context is service-design prose, **not a source-code
repository**; `code-context-r64` is a code/technical-context proxy, not a pure
code corpus. The separate coding-agent fixture contains code, diffs, and recorded
tool observations. Replay executes no tools and is not an autonomous agent or a
correctness benchmark.

All context repetition counts are **uncalibrated**. In particular, these are not
128-token, 4K, 8K, 128K, or 256K input buckets, nor exact reproductions of the
reference's published benchmark shapes. Tokenization, chat templates, unique
salts, and accumulated agent turns change input length. Retain server-reported
prompt/completion usage and inspect context fit before drawing comparisons.
The native 4096-page allocation is a requested capacity, not proof of usable
long-context capacity once QSA side caches and hybrid state are included.

## Pinned artifacts and implementation evidence

Native checkpoint (already cached):

```
albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE@ef554143369a706525336f6b42a09094835dc077
/home/nchapman/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/ef554143369a706525336f6b42a09094835dc077
```

Native version: `66974d8`. Expected `target/release/w34` SHA-256:
`f203e7efd883893192cd6262871d278997b1380253e3874053f8d7947cb8948f`.
Check `sha256sum target/release/w34` before launch; these identifiers are pinned
provenance, not an assertion that this campaign qualified that binary.

Recipes were checked against the actual option parser in
`crates/w34-host/src/main.rs`: `serve --model`, `--offline`, `--tp`,
`--kv-format`, `--kv-pages`, `--host`, `--port`, `--speculative mtp`, and
`--mtp-k 3`. Target-only means omitting speculation entirely. No invented
`--moe-offload-mode` or hot-cache tuning CLI is used. Earlier design sections in
`docs/QWEN_FLASH_NEXT.md` describe plans as well as implementation; their targets
and historical results do not qualify these campaign profiles.

Source residency defaults in `model/rank.rs` are target hot48 (TP1) and hot112
(TP2) per layer/rank, staging 32. `mtp.rs` selects hot64 (TP1) or hot128 (TP2),
staging 32. These are implementation defaults, not measured allocation receipts.
The runtime discovers compact MTP under `<model_dir>/runtime/mtp-int4-g32`
when root MTP tensors are absent. Rankings are discovered at
`static_hot_cache_rankings.json` or `runtime/static_hot_cache_rankings.json`;
retain these assets and record which was used. No external draft-model flag or
weight conversion is required by these recipes. Verify the runtime assets exist
and record hashes; missing assets or startup failure must remain failures rather
than silently substituting a model.

Native HTTP model IDs deliberately equal the full snapshot path passed to
`--model`: no alias is set. `crates/w34-host/src/server/http.rs` advertises the configured model name
at `/v1/models`. The reference explicitly serves `Qwen3.8-Flash-Next`, matching
its TOML request model. Always inspect `/v1/models` before running; do not send
the upstream base-model ID to a server advertising a different ID.

## Launch and measure, one profile at a time

Each runnable TOML deployment contains its complete launch command. Execute
native commands from the repository root. The reference command changes to its
checkout. `w34perf` does not launch or stop these servers. They deliberately
share port 18034: stop one before starting another and wait for readiness.

```sh
cd /home/nchapman/Code/warp-34/benchmarks/perf
python3 -m w34perf validate campaigns/flash-next-serving.toml
# After manually starting the chosen deployment:
curl -fsS http://127.0.0.1:18034/v1/models
python3 -m w34perf run campaigns/flash-next-serving.toml \
  --engine warp34-flash-tp1-target --gpus 1 \
  --out results/flash-next-tp1-target
python3 -m w34perf report results/flash-next-tp1-target
```

Use fresh output paths and repeat for `warp34-flash-tp1-mtp3`,
`warp34-flash-tp2-target`, `warp34-flash-tp2-mtp3`, and
`qfn-2x3090-mtp3`; use `--gpus 2` for two-card deployments.
`--gpus` filters by card count, not device IDs; select physical devices when
launching the server with `CUDA_VISIBLE_DEVICES` (or Docker's device selection). Preserve startup logs,
request/SSE records, manifests, usage, errors, and GPU telemetry.

**Separate sweeps:** compare target-only against MTP3 *within TP1*, then do the
same *within TP2*. Compare TP1 versus TP2 separately: GPU count, expert ownership,
hot-cache capacity, and collectives all change. MTP3 adds draft weights/cache and
verification state and may alter available memory; do not attribute every
cross-profile gain to speculation. Retain acceptance/tokens-per-round telemetry
if available, alongside TTFT, TPOT, aggregate output throughput, and agent-session
latency. HTTP output throughput including prefill/overhead is not a decode-kernel
rate. No claim of MTP parity, speedup, concurrency, or long-context support follows
from successful TOML validation.

## Reference recipe and its prerequisites

Reference checkout: `/tmp/w34-benchmark-refs/qfn-2x3090`, pinned commit
`4dca58f7243f3910ba072a97e7f29fcf6f5636f3`. Read inputs are
`configs/2x3090-128gb.env`, `scripts/docker_serve.sh`,
`scripts/serve-container.sh`, `docker/Dockerfile`, and `repro.lock.json`.
The lock pins the base container:

```
vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
```

The **base image alone is not the recipe**. The Dockerfile installs the repository
vLLM overlay, humming-kernels 0.1.12, compiled extensions, rankings, and the
`qwen38-serve` entrypoint. The launch command requires an already prepared
`qwen38-flash-next-2x3090:locked` image built from that pinned checkout/Dockerfile.
Image preparation, if needed, is an operator prerequisite outside this campaign
(`docker build -f docker/Dockerfile -t qwen38-flash-next-2x3090:locked .` from the
reference checkout); no build was executed here. Record the resulting image ID
and inspect it before using the mutable local tag. Do not replace it with the
unmodified pinned base image in `docker_serve.sh`.

The recipe mounts the same native snapshot read-only at `/model`; compact MTP is
`/model/runtime/mtp-int4-g32`. The reference 128GB profile uses TP2/EP2,
BF16 activations/KV, hot84, dynamic LRU/mixed VMM expert backing, 30GB UVA CPU
expert offload, CPU PLE, 4,429,185,024 KV bytes, one sequence, 4096-token prefill
chunks, prefix retention interval 1600, approximate QSA, and MTP depth 3.
Custom all-reduce is disabled by default. Do not conflate this with the distinct
64GB hot-only CPU-expert/mmap-PLE recipe or a tuned hot104 headline result.
The wrapper uses `--gpus all`; run it only on the intended two-card host or
explicitly qualify a changed wrapper. Do not assume native CUDA visibility
settings constrain this Docker wrapper.

## Residency, topology, and cache interpretation

Native intended host: one or two 24GiB RTX 3090 Ti, Ryzen 7950X3D, approximately
124GiB RAM with NVMe and swap. The documented dual-card topology is PCIe Gen4 x8,
PHB, P2P disabled, pinned-sysmem collectives. Native dense/hot weights and main
TQ4 KV are device-resident; cold experts have mapped pinned host backing, while
FP8 PLE uses mmap/NVMe plus the Linux page cache. FP32 GDN state, FP16 convolution
history, QSA raw/compressed side caches, and radix snapshots also consume memory.
Do not assume the whole checkpoint fits in VRAM or that CPU-hybrid expert compute
is enabled by a design document. Verify actual loader/runtime behavior.

The reference's published hardware was 2x RTX 3090, Threadripper PRO 5975WX,
PCIe Gen4 x16, 128GB RAM, not this 3090 Ti x8 host. Its 128GB wrapper imposes no
memory limit and notes that loading can rely on swap. Record GPU UUIDs, PCIe
width/generation, P2P/NUMA topology, CPU affinity, RAM/swap use, storage, clocks,
power, temperatures, driver/CUDA versions, startup time, peak VRAM/RSS, faults,
and swap I/O for each run. Ordinal GPU IDs in TOML are selections, not inventory.

Warmups and explicit priming are separate from measured requests. `unique`
prompt salts discourage exact reuse but do not flush prefix state, expert LRU,
PLE pages, JIT kernels, CUDA graphs, or OS caches. `natural` agent sessions grow
prefixes across turns; `primed` intentionally tests reuse. Later repetitions are
potentially warmer than earlier ones. Report each repetition and case order;
do not label the default run cold. For a separate cold-process experiment,
restart between trials and record startup separately, with warmups disabled by
an explicit run configuration. A process restart still does not cold-clear the
OS page cache or persistent JIT caches. Do not drop system caches implicitly.

Validation performed from `benchmarks/perf`:
`python3 -m w34perf validate campaigns/flash-next-serving.toml` →
`valid: 11 deployments, 5 cases`. This validates schema/fixtures only; no serving
measurements or support qualification are claimed.

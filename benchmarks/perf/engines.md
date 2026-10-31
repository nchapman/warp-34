# Engine qualification and deployment matrix

## Primary lane: approximately equivalent 4-bit weights

The common ground is **4-bit model weights**, not FP16 weights. KV storage is an
independent axis. Do not compare a 4B GPTQ run to a 27B `.ninfer` run, or imply
that all 4-bit formats use exactly four bits/parameter or have equal quality.

1. Choose one **source checkpoint and immutable revision** for a campaign.
   Start the full five-engine campaign with a mutually supported **27B dense**
   Qwen revision; verify ninfer artifact provenance before picking the exact one.
   The 4B GPTQ entries in `suite.toml` are practical local smoke examples, **not**
   a qualified match for the ninfer/placeholder GGUF entries.
2. Use the same GPTQ INT4 artifact for warp-34/vLLM/SGLang where supported.
   Target symmetric group-size 128 if compatible. Pin every shard, scale, config,
   tokenizer, template, and any native MTP/draft sidecar with a SHA-256 inventory.
3. For llama.cpp choose the nearest **Q4_K_M or Q4_0** GGUF candidate based on
   measured quality and artifact size, not the format name alone. Q4_K_M mixes
   precision. Record per-tensor exceptions, imatrix/calibration dataset, converter
   revision and command, and effective weight bytes/parameter.
4. For ninfer choose a compatible v3 groupwise integer 4-bit artifact for `sm_86`.
   The NVFP4/FP8 Blackwell artifact is **not** an RTX 3090 Ti candidate.
   NVFP4 is not GPTQ INT4. Record artifact source/conversion, scale
   precision, activation route, embedding/lm_head exceptions, GDN state precision,
   and any fused/dequantization tradeoffs. No private grafts or hidden adapters.
5. Independently qualify each **weight × KV × speculation** deployment on public
   functional tasks/long-context retrieval. This must not import development
   oracle outputs or grant a quality pass from matching a format label.
6. Publish size and quality alongside speed. Equivalent 4-bit deployments answer
   "what can users run on this hardware?"; a same-GPTQ subset can additionally
   isolate implementation differences. Keep these claims distinct.

`model_revision` is the quantized artifact revision; `source_model_revision` is
the common pre-quantization source. `quantization_details` describes all the
above. `quality_evidence` names retained independent evidence; it is a manual
release-audit input, not a self-certification automatically verified by Python.

## Pins and recipe status

The checked-in launch commands are **candidate recipes, not proof of execution**.
Update them with `--help` output, launch logs, and binary/container hashes during
qualification. Dev-oracle donor pins are not an obligation for this suite: update
release pins deliberately and retain older reports for longitudinal comparisons.

| Engine | Initial pin / source | 1 card | 2 cards | Current qualification |
|---|---|---|---|---|
| warp-34 | current local commit, record exact binary hash | native | tensor parallel 2 | 4B runner smoke and repeated 27B 0df FP16 one-card probes pass; full campaign pending |
| vLLM | 0.31.0 (existing installed donor) | native | `--tensor-parallel-size 2` | 27B eager c11 smoke and compiled/CUDA-graph 0df one-card probes pass; quality/two-card pending |
| SGLang | 0.5.19 / image below | native | `--tp-size 2` | 27B 0df **BF16** one-card probes pass; c11 Marlin and 0df FP16 startup failures retained |
| llama.cpp | `5047748f114564fcf271e6a648f60e2d2eae6264` | all layers GPU | layer split (`--split-mode layer --tensor-split 1,1`) | Candidate source pin; artifact/build/runtime pending |
| Don-Chad/ninfer-3090 | `d8c29253f2c0845611a794d87057950107a4956b` | native | Linux pipeline `--devices 0,1` | Single-card artifact/build pending; dual-card **experimental/unqualified** |

On this machine two 3090 **Ti** cards have PCIe Gen4 x8, PHB topology, no NVLink,
and P2P disabled. Never describe layer/pipeline splitting as tensor parallelism.
Record exact topology, physical GPU UUIDs, CPU/RAM, clocks/power limits, driver,
CUDA, container digest, KV/prefix pool sizes, context/admission limits, and offload
behavior. A two-card capacity win is not necessarily a latency win. Do not quietly
CPU-offload a model and compare it to fully resident GPU execution. Capture host
RAM and PCIe behavior for offload lanes separately.

### ninfer-specific verified documentation

At the pinned source, `docs/serving.md` starts `ninfer-serve <model.ninfer>` with
`--model-id`, `--host`, `--port`, `--max-context`, `--kv-capacity`,
`--max-concurrency`, and `--kv-dtype`. `docs/cli.md` and
`src/serve/serve_options.cpp` define `--devices A,B` and `--stage-layers A,B`.
The README explicitly says the Linux multi-GPU pipeline route has **not** been
requalified on the current context engine. Earlier release notes explicitly
exclude tensor parallelism and multi-GPU DFlash/DFlash2. Do not infer dual-card
speculative support from a parser accepting flags.

Candidate baseline, after selecting a real compatible 27B artifact:

```sh
CUDA_VISIBLE_DEVICES=0 ninfer-serve /models/selected-4bit.ninfer \
  --model-id qwen-4bit --host 127.0.0.1 --port 8000 \
  --max-context 32768 --kv-capacity 32768 --max-concurrency 8 --kv-dtype int8
# Experimental Linux pipeline probe: same command with CUDA_VISIBLE_DEVICES=0,1
# and --devices 0,1. No speculation until independently qualified.
```

Source references (read at the pin, not floating `master`):

- https://github.com/Don-Chad/ninfer-3090/tree/d8c29253f2c0845611a794d87057950107a4956b
- `README.md`, `docs/serving.md`, `docs/cli.md`, `src/serve/serve_options.cpp`
- https://github.com/ggml-org/llama.cpp/tree/5047748f114564fcf271e6a648f60e2d2eae6264

## 27B candidate mapping (2026-10-30)

Machine-readable evidence: [`qualification/27b-candidates.json`](qualification/27b-candidates.json).
This is a candidate shortlist, **not a frozen common-source or quality-qualified
campaign**. No large model downloads were performed. HF API metadata was queried
for repository revisions, sizes and LFS SHA-256s; remote hashes below are
advertised values, not verification of local GGUF/ninfer bytes.

| Artifact | Immutable revision | Bytes | Status |
|---|---|---:|---|
| `nchapman/Qwen3.8-27B-W4A16-GPTQ` | `c11a884fe4b659eccdce66761f26d3775cf7696c` | 19,438,009,793 (11 visible files incl. MTP/tokenizer/template) | Cached; full streaming inventory + re-verification pass |
| `WarlaxZ/Qwen3.8-27B-NInfer-3090/qwen3_8_27b.ninfer` | `ea90e8766354c723e6bd0cf0ed731d5d71ce5fa1` | 18,982,458,624 | 3090-specific candidate, not downloaded |
| `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | 16,464,440,224 | Candidate, not downloaded |
| Same repo, `Qwen3.8-27B-UD-Q4_K_XL.gguf` | same revision | 17,559,178,144 | Candidate, not downloaded |

The 3090 NInfer card/manifest at the pinned source explicitly declares base
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, making that
the **source candidate**. It uses importance-weighted scale-search mixed Q4/Q5,
Q4 embeddings, Q6 output head, and extra vision/MTP/DFlash/proposal components.
It is not the generic `neroued/Qwen3.8-27B-NInfer` artifact or the Blackwell
NVFP4 variant. Its documented quality/speed results are upstream claims, not
independent suite qualifications. Its imatrix came from the pinned Unsloth repo;
that does **not** prove the GGUF's exact base revision or equal quantized weights.

The cached GPTQ README/recipe/config name the base repo but **do not record its
source commit**. Do not fill `source_model_revision` from the model name, a current
HF ref, or the NInfer card. The newer cached GPTQ `0df81ea...` is a different
artifact with changed model bytes/index and an extra Q4 MTP sidecar; don't silently
interchange them. Snapshot `c11a88...`'s index references both present safetensors
files. Its tokenizer and froggeric v22.5 template differ from the official base
metadata and need explicit alignment/audit. A same-artifact GPTQ smoke is valid
without claiming cross-format source/quality equivalence.

Local inventory evidence is `/tmp/w34-perf-27b-gptq-inventory.json`, stable hash
`cfef8c491913394b330a74c92a1738a474e9e04e09d4f2cd422b0ad7cd8f2cf2`.
The evidence JSON records principal file hashes; retain the full inventory with
future run bundles. Inventory includes sidecars, so its total size is **not**
text-only loaded weight bytes or VRAM usage.

Runtime inventory found installed vLLM **0.31.0**, SGLang **v0.5.19** image
`sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`,
and a llama.cpp server CUDA image
`sha256:b2497f8834f5ecb4e38530f6bf2734b8e0be107ff48e4720145911c86930f2ce`.
An image being cached does not qualify its model support or associate it with
the candidate source pin. No host llama.cpp/NInfer binaries or local 27B GGUF/
`.ninfer` artifacts were found. Freeze builds and retain startup evidence before
using their recipe rows in results.

### vLLM 27B single-card smoke (not a speed result)

Installed vLLM 0.31.0 successfully served the **exact cached GPTQ `c11a884...`**
on one RTX 3090 Ti, using `--dtype half --language-model-only --enforce-eager`,
`--max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.90`,
`--cpu-offload-gb 0`, offline HF mode and no speculation. Startup logs confirm
compressed-tensors Marlin weights, FP16 activation/KV auto mode, and Triton GDN.
They explicitly report eager mode disables torch compilation/CUDA graphs; this
is **not** the optimized publication recipe. `/v1/models` advertised the expected
`qwen38-gptq-smoke` ID. Two sequential + two concurrent (C2) requests and a
four-turn agent replay all succeeded: **8/8 measured requests**, valid server
usage, one repetition and 32-token output ceilings. The reduced `batch-8` case
was **C2, not C8**; resolved values are in its manifest.

Evidence: `/tmp/w34-perf-vllm27-eager-smoke/` (full resolved config, raw SSE,
coverage, summary, checksums), `/tmp/w34-perf-vllm27-startup.log`,
`/tmp/w34-perf-vllm27-models.json`, and
`/tmp/w34-perf-vllm-serve-help-all.txt`. Principal evidence hashes are in the
candidate JSON. The server was stopped; quality, optimized graph mode, long
contexts, and vLLM two-card runs are not qualified by this smoke.

Next gates: recover or reproduce GPTQ base-revision provenance, establish GGUF
source/converter provenance, acquire/check candidate bytes, independently test
quality, and qualify each real Linux runtime/KV mode. Two-card NInfer stays
**disabled/unverified** until an explicit successful pipeline probe; use
`--gpus 1` for a single-card-only campaign without deleting matrix entries.

### Repeated 27B one-card operational probes

The [probe preset](qualification/27b-one-card.toml), [full inventory](qualification/27b-gptq-0df.inventory.json),
and [machine-readable launch/probe ledger](qualification/27b-one-card.json) retain
this follow-up. It deliberately selects **newer snapshot `0df81ea...` for all
three runnable profiles**, rather than pretending SGLang used `c11a884...`.
All **13 files / 19,610,855,960 bytes** were inventoried and re-verified, stable
hash `82675bcf7919243f8a373072ddcd73d36fc76070b1182dfc8f531c49cd683d03`.
The newer snapshot leaves small GDN a/b projections unquantized and adds a Q4
MTP sidecar. Source provenance remains unresolved. No model downloads or engine
source patches were needed.

Retained SGLang v0.5.19 failures, not zero-throughput rows:

1. Original `c11a884...` / FP16: Marlin repack rejects width 96 (tile width 64).
   The Docker client exited **0 despite worker failure**; logs, not exit alone,
   establish failure before HTTP startup.
2. New `0df81ea...` plus `--language-model-only`: this flag accepts only
   MuseGlimmer in this image, not Qwen3.5. Its generic help is insufficient proof.
3. New `0df81ea...` / FP16 without that flag: weights load, then Triton GDN
   causal-convolution compilation fails with BF16/FP16 branch-type mismatch
   during prefill graph initialization. Runtime also caps requested active 8 to
   3 (16 Mamba slots / five slots per request) and KV capacity to 11,739 tokens.

An **explicitly different BF16** SGLang profile passes preflight and **42/42
measured requests / nine trials**, with FP32 SSM, BF16 KV (16,384 tokens), active
limit 2, 12 Mamba slots, zero CPU offload, and both prefill/decode graph capture.
Offered C8 therefore includes queueing behind two active requests; it is **not**
eight simultaneously GPU-active sequences or a successful FP16 qualification.
Vision remains loaded because the text-only flag is unsupported for this model.

vLLM 0.31.0 on the **same `0df81ea...` bytes**, FP16 activation/auto KV, FP32 SSM,
text-only, max context 8192, active limit 8, zero CPU offload and no speculation
passes preflight and **42/42 requests / nine trials**. Startup confirms **21.25 s
of torch.compile and piecewise/full CUDA graph captures**, not `--enforce-eager`.
Its profiled KV pool holds 17,347 tokens; pools and memory are not matched to
SGLang. This is the graph-enabled qualification recipe, not exhaustive tuning.

Warp-34 also passes preflight and **42/42 requests / nine trials** on those exact
bytes with FP16 activation/KV, FP32 SSM, target-only and 1024 requested KV pages
(16,384 tokens). Its CLI leaves CUDA graphs enabled; this HTTP suite does not
independently profile graph execution. The exact binary hash is in the ledger.
All three servers were stopped before full project verification.

The preset uses three repetitions of short C1 (two requests), **offered C8**
(eight requests), and a four-turn C1 coding-agent replay, with 64-token output
ceilings. Raw resolved configs, SSE, coverage, summaries, checksums, startup logs
and preflight paths/hashes are in the ledger. Inventory, model ID and indexed
shards pass the new `preflight` command; this is **not remote loaded-byte
attestation**. Warp-34 uses its own ChatML renderer rather than executing the
artifact Jinja: matching stored template hashes do not establish token-aligned
prompts. Different dtypes, rendering and admission must remain visible.

No speed leaderboard, quality pass, tail-latency conclusion, long-context,
saturation or two-card claim follows from these operational probes. Archive
raw `/tmp` evidence and independently anchor/sign hashes before publication.

## KV-cache sweep (same 4-bit weights)

For **each GPU topology**, make a deployment profile per actual KV setting.
Change the launch command, restart the server, and record `kv_format`. Keep
context/batching/speculation and weight artifact fixed within a KV comparison.
Do not use a field label alone to pretend the server changed storage precision.

| Engine | Baseline KV | Quantized KV candidates | Important caveats |
|---|---|---|---|
| warp-34 | `--kv-format fp16` | `q8_0`, `tq4`, `tq4_all` | TQ4 boundary policy is model dependent; record effective bytes/token, not just nominal bits |
| vLLM | `--kv-cache-dtype auto` with `--dtype half` | FP8/other formats only if pinned release + hybrid model + sm_86 kernels actually support them | No presumed sm_86 native FP8; unsupported is a retained result, not FP16 silently substituted |
| SGLang | auto with float16 | Quantized candidates supported by pinned hybrid-model backend | Same hardware/model qualification requirement; do not assume generic CLI support proves kernel support |
| llama.cpp | `--cache-type-k f16 --cache-type-v f16` | K/V `q8_0`, `q4_0`, mixed K8/V4 where supported | Quantized V/attention backend and hybrid model support must be tested; Q4_0 != TQ4 |
| ninfer | `--kv-dtype bf16` | `int8`, `rk8v4`, `rk4v4`; other advertised formats only after qualification | Baseline BF16 != FP16; rotation/state precision/host cache are separate knobs |

The sample ninfer entry uses INT8 because that is a documented practical serving
profile, **not** an FP16-equivalent baseline. Add the BF16 entry for the common
baseline lane. Do not publish its INT8 result in an FP16-KV column.

For warp-34 the same one-card example can be restarted with each KV format:

```sh
# Run separately, not concurrently:
CUDA_VISIBLE_DEVICES=0 ./target/release/w34 serve \
  --model nchapman/Qwen3.5-4B-W4A16-GPTQ \
  --kv-format q8_0 --kv-pages 4096 --port 8000
# Copy the endpoint table to id=warp34-1-q8, kv_format=q8_0, and record this command.
```

Required KV measurements: short-context overhead, long-context decode, prefill,
concurrency pressure, largest successful context and active sessions under a
fixed memory budget, actual peak VRAM, and retrieval/functional quality. Evaluate
both equal-workload latency and equal-hardware maximum capacity. The runner
currently supplies latency/throughput shapes and retains failures; automatic
capacity search, memory sampling, exact token buckets, and a quality battery are
tracked in `PLAN.md`, not claimed complete.

## Speculation and dual-card lanes

Always run target-only first. Then separate native MTP and DFlash2 profiles,
recording proposer artifact, precision, token/block count, memory and batch
policy. Never pool speculative and target-only rows. Use natural prose/code and
agent histories: acceptance on repeated synthetic text is not representative.

For each 4B/9B/27B dense and 35B-A3B MoE campaign where supported:

- one card; two cards with the engine's real supported parallel strategy;
- target-only + baseline KV; target-only + each qualified compressed KV;
- best practical speculation with the **same** KV matrix where supported;
- optional two independent one-card replicas as a separate **data-parallel**
  capacity lane, not mislabeled TP=2 (routing support is a follow-up);
- explicit `unsupported`, `not-tested`, `startup-error`, `OOM`, `timeout`, or
  `quality-failed` campaign ledger entries. Unsupported rows are not zero tok/s.

Do not edit engine source to improve a benchmark score or reuse development
oracle instrumentation. Save tuning trials separately and freeze settings before
recording the publication campaign.

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
4. For ninfer choose a compatible v3 4-bit artifact (groupwise INT4 or NVFP4 as
   available). NVFP4 is not GPTQ INT4. Record artifact source/conversion, scale
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
| warp-34 | current local commit, record exact binary hash | native | tensor parallel 2 | Local runner smoke; full campaign pending |
| vLLM | 0.31.0 (existing installed donor) | native | `--tensor-parallel-size 2` | Recipe only in this suite |
| SGLang | 0.5.19 | native | `--tp-size 2` | Recipe only in this suite |
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

# Short upstream evaluation tooling qualification

## Outcome

The direct lm-eval workflow completed on **three servers and two model families**
on the dual-RTX-3090 NVLink pod. Each deployment evaluated **10 IFEval documents
and 10 GSM8K documents**, using upstream `--limit 10`. All eight task invocations
completed and produced native results JSON and sample JSONL. All evaluated
responses were nonempty. This is tooling coverage, not quality baselines,
rankings, or proof of speculative parity.

| Server | Target | Decode | Evaluated documents | Result |
|---|---|---|---:|---|
| warp-34 `b7f5fdd` | Qwen3.8-27B HGPTQ | TP2/MTP-3/TQ4 | 10 + 10 | Passed |
| warp-34 `b7f5fdd` | Qwen3.8-Flash-Next HGPTQ | TP2/MTP-3/TQ4 | 10 + 10 | Passed |
| vLLM 0.31.0 | Qwen3.8-27B GPTQ | TP2/MTP-3/native KV | 10 + 10 | Passed |
| SGLang 0.5.19 | Qwen3.8-27B GPTQ | TP2/native MTP, 3 steps/native KV | 10 + 10 | Passed |

All use the same `lm-eval==0.4.13` client, native tasks and non-thinking setting.
The different checkpoint quantizations and KV policies make these unsuitable
for a controlled cross-server quality comparison. No 9B or 35B check was run.

### Checkpoint revisions

- `nchapman/Qwen3.8-27B-W4A16-HGPTQ`:
  `1de743eedefdb7422e6b171174c7080204f5b005`.
- `nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ`:
  `6ee06d2936b111f19eb5eb331044cb8b8b28562b`.
- `nchapman/Qwen3.8-27B-W4A16-GPTQ`:
  `0df81eadcfd59601cb82ccece2f16ea7d39244bf`.

Dataset cache refs: IFEval `966cd89545d6b6acfd7638bc708b98261ca58e84`;
GSM8K `740312add88f781978c0658806c59bc2815b9866`.

## Serving and evidence checks

- Each task reports native `n-samples.effective=10`, with original dataset counts
  541/1,319. Sample logs contain 10 distinct document IDs per task.
- GSM8K's two filters produce **20 sample rows for 10 documents**. These are not
  duplicate inference calls or a custom scoring pipeline.
- Each warp-34 deployment reports 20 completed requests, zero failed/cancelled
  requests, and zero HTTP client/server errors.
- vLLM reports 20 successful requests: 18 stop finishes, two length finishes,
  zero abort/error finishes. Token-budget finishes remain scored unchanged.
- SGLang logs 21 HTTP-200 chat requests: one built-in startup warmup before
  evaluator startup, followed by the 20 evaluation requests. No empty evaluated
  responses. Its shutdown emitted NCCL cleanup and asyncio cancellation warnings
  after both tasks and metrics capture finished; these logs are preserved.
- All servers stopped afterward; pod GPU process inventory was empty.

## Operational lessons and preserved interruptions

The earlier full Flash-Next/plain and 27B/MTP runs were stopped at the user's
request before scoring. They remain incomplete, not successful full evaluations:
`results/quality-flash-001/` and `results/quality-27b-mtp3-001/`, with explicit
interruption receipts. No partial outputs were mixed into smoke results.

The temporary vLLM process supervisor missed its `Application startup complete`
message and waited even though the API was ready. Readiness was independently
verified through `/v1/models`; both native evaluator commands then ran directly
and exited successfully. The original supervisor was terminated (exit 143),
not reclassified as a successful launcher. `direct-client-receipt.json` records
the successful client execution separately. The readiness detector was corrected
for SGLang. These temporary supervisors are not the evaluation interface; use
the direct CLI commands in [README.md](README.md) and [servers.md](servers.md).

Startup/compilation can dominate these tiny checks: the 27B warp-34 check took
about 65 seconds including startup, Flash-Next about 3.5 minutes, and SGLang about
7 minutes including first startup/compilation. vLLM's direct client execution
took about 64 seconds, excluding startup and the supervisor delay. These are
operational observations, **not comparative performance measurements**.

## Retained artifacts

Native results JSON is copied unchanged into `qualification/<run-id>/<task>/`.
No custom score aggregation or success threshold is applied. Raw bundles,
including sample JSONL, package freezes, metadata, server logs and metrics, are
retained locally under gitignored `results/` and on the pod under `/workspace/`:

- `quality-27b-mtp3-smoke10-001`
- `quality-flash-mtp3-smoke10-001`
- `quality-vllm-27b-mtp3-smoke10-001`
- `quality-sglang-27b-mtp3-smoke10-001`

`qualification/evidence.sha256` records hashes of the retained smoke bundles.
The small checked-in results are not a substitute for those raw bundles.

## Verification

`mise run quality-check` exercises native config parsing, both smoke limits,
chat-message wrapping and generation-payload settings. `mise run perf-check`
covers the existing performance tooling. No Rust/CUDA engine changes were made;
engine build/GPU regression suites were not rerun for this evaluator-only work.
Review had no blocking findings. Suggestions to enforce the optional dependency
suite in the core check and avoid a private adapter payload method were declined:
the evaluator stays isolated, and its exact pinned adapter is deliberately tested
at the payload seam so dependency upgrades trigger revalidation.

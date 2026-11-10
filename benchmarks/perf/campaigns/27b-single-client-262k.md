# Single-client 27B HGPTQ: restoring 262k capacity

## Scope

One RTX 3090 Ti (24 GB), pinned
`nchapman/Qwen3.8-27B-W4A16-HGPTQ@ae51987e3339a2a70070702548571e10fe8bcbc2`,
TQ4 (not `tq4_all`), one active client, native MTP-3. The desired context is
262,144 tokens; a modest reduction is acceptable for measured operating headroom.
This supersedes the **default recommendation**, not the recorded observations, in
[the earlier concurrent capacity investigation](27b-single-gpu-capacity.md).

## Defect and regression

Before the fix, both plain and MTP serving with `--kv-pages 16384
--max-num-seqs 1 --max-model-len 262144` failed at target-pool admission:
`insufficient VRAM for serving state pool and headroom`.

Embedding placement considered boot allocations plus a fixed 2 GiB slack preference,
but not the requested target KV/state budget. The untied 248,320-by-5120 FP16 input
embedding table could remain in VRAM even when that crowded out requested context.
The existing mapped-host path recovers approximately **2.368 GiB**. This moves
embedding storage, not transformer computation or the output `lm_head`, to host memory.

The placement fix checks boot and steady-state requirements separately, reusing the
same target KV/state geometry as final live-memory admission. Repack staging is
transient and is not charged simultaneously with target pools. Tied embeddings and
tiered-MoE placement retain their existing behavior; state clamping remains authoritative.

Regression `worker_capacity::single_client_27b_hgptq_serving_allocates_262k_tq4_context`
failed before the fix at the expected admission check and passed afterward.

## Boundary run after target-placement correction

```sh
export PATH="/opt/cuda/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 CUDA_DISABLE_PTX_JIT=1 ./target/release/w34 serve \
  --model nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --revision ae51987e3339a2a70070702548571e10fe8bcbc2 --offline \
  --speculative mtp --mtp-k 3 --kv-format tq4 --kv-pages 16384 \
  --max-num-seqs 1 --max-model-len 262144 --port 18834
```

Two sequential synthetic service-design prompts, with the first actual answer
carried into the follow-up. Temperature zero, thinking disabled, `ignore_eos=true`,
64 output tokens per response. This is a **capacity and execution probe**, not an
intact SWE trajectory, quantization-quality test or controlled embedding-placement
performance A/B benchmark.

| Turn | Prompt tokens | Output tokens | First output | Completion | Reported cached tokens |
|---|---:|---:|---:|---:|---:|
| Initial | 261,750 | 64 | 700.64 s | 702.32 s | 0 |
| Follow-up | 261,840 | 64 | 703.34 s | 705.13 s | 0 |

Both responses succeeded; no server errors/cancellations. Actual request totals were
261,814 and 261,904 tokens, below the configured 262,144 limit. This does not test
an exactly full last page or all possible speculative stopping boundaries.
MTP telemetry recorded **47 rounds, 136 drafted tokens and 79 accepted tokens**.
All active pools drained; the final cache retained 16,368 of 16,384 KV pages and
three of eight GDN slots. Zero reported reuse on the continuation remains an
unresolved cache/admission observation; no root-cause or interactive-readiness claim.

GPU used memory rose from **24,029 to 24,067 MiB**, leaving only **99 MiB free**
at peak/end (`nvidia-smi` reserved memory is separate). The ~23.5-minute probe
therefore establishes near-262k execution, **not comfortable production headroom**.
The server stopped cleanly afterward.

## Follow-on MTP budgeting issue

Reducing to `--kv-pages 14336 --max-model-len 229376` after only the target-placement
fix caused startup `CUDA error 2 (out of memory) in w34_malloc`: the smaller target
pool allowed GPU-resident embeddings again, but subsequent MTP allocations were
not included in the placement/state-pool plan. Lower context is not a monotonic
memory improvement while placement and optional allocations can change.

The subsequent dense-MTP correction reserves its mandatory weight arena, FP16 KV,
scratch, proposer IO and initialization staging before target placement and live
pool admission. Its shared formulas are also used by the actual loader. Optional
quantized draft-head allocation retains its existing policy. GPU regressions now
successfully load target plus MTP at both 14,336 and 16,384 pages. The 99-MiB
boundary measurement above predates this correction and must not be presented as
the final allocator's measured 262k headroom.

## Selected next-qualification configuration: 192 Ki tokens

Use the same pinned model and MTP-3/TQ4 command, changing only:

```text
--kv-pages 12288 --max-num-seqs 1 --max-model-len 196608
```

This is a **single-client, text-only starting point**, not a concurrency or production
SLO qualification. The corrected 229,376-token server also started, but only had
275 MiB free at startup; allocation choices, including the optional MTP draft-head
copy, mean free memory is not simply proportional to KV capacity.

With the corrected MTP budget and 196,608 configured tokens:

- **196,155 prompt tokens + 64 output tokens = 196,219 total**, one cold request.
- Client first output **428.13 s**; completion **429.47 s**; successful terminal
  response and usage, no request errors.
- MTP: **24 rounds, 67 drafted tokens, 39 accepted tokens**.
- GPU memory **23,219 → 23,257 MiB**; free VRAM **947 → 909 MiB**.
- Target capacity 12,288 pages; GDN pool eight slots. Final active counts zero;
  retained cache 12,260 KV pages and three GDN slots. No follow-up was submitted
  in this run; it does not test warm-cache behavior.
- Server stopped cleanly afterward. One near-capacity request is not a sustained
  soak, exact-final-page test, or proof of a leak-free plateau.

After the DFlash serving-entrypoint changes, a fresh-server MTP smoke at the same
196,608-token allocated capacity completed two shorter turns (3051 / 3333 prompt
tokens, 256 output tokens each). Cold / follow-up TTFT was **2.204 / 0.092 s**;
E2E **4.664 / 2.356 s**, with **3264 cached follow-up tokens**. GPU peak remained
23,257 MiB, confirming the same observed headroom through the final CLI path.

This keeps approximately 0.89 GiB free at measured peak while preserving the
normal MTP execution policy. It is **not** a retraction of restored 262k capacity:
262k remains the demonstrated boundary, while 192 Ki is the chosen headroom
starting point for further work.

## DFlash2-aware serving preflight

Serving now accepts the actual speculative configuration instead of only an MTP
boolean. For DFlash2, it resolves the selected draft checkpoint and validates
allocation metadata before placing target embeddings or sizing target state/KV
pools. Activation uses the resolved draft directory from preflight.

The draft reserve includes its checkpoint-dependent weight arena, replicated
selector codebooks, FP32 RoPE, separate quantized KV projection when required,
**BF16 draft KV** for committed capacity plus the proposal suffix, both 32 MiB
scratch/graph arenas even with graphs disabled, and target-feature/proposer buffers.
Target TQ4 does not change draft KV precision. State requirements are charged in
the target pool, not duplicated in the draft reserve. The preferred 24-slot pool
can still clamp: one client's required floor is **block size + 3** (11 for block 8,
19 for block 16), covering three private states, verification extras, and one
locked radix checkpoint. Treating all 24 preferred slots as mandatory incorrectly
rejected the pinned 27B/DFlash2 32k startup and is covered by a GPU regression.
Checked arithmetic and
metadata validation reject unsupported or impossible estimates instead of silently
reserving zero bytes.

A final live smoke used the pinned target above plus
`nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ@46b87ea125d24924affa8da9c3e981a937aab216`,
`--kv-pages 2048 --max-num-seqs 1 --max-model-len 32768`, TQ4 target KV and the
unchanged BF16 draft KV. The actual GDN pool was **12 slots**, validating why the
preferred 24-slot count cannot be a hard floor.

- Two responses completed, **3050 / 3332 prompt tokens**, 256 output tokens each.
- Cold / follow-up TTFT **2.196 / 0.086 s**; E2E **4.773 / 2.126 s**.
- Follow-up reused **3264 prompt tokens**; no request errors.
- GPU used memory **22,685 → 22,723 MiB**, with **1443 MiB free** at peak/end.
- DFlash telemetry: 163 rounds, 1124 drafted tokens, 347 accepted tokens.

This is a short-prompt smoke with a 32k **allocated capacity**, not a 32k full-prompt
qualification, matched MTP performance comparison, or evidence that DFlash2 fits
at 262k. DFlash2 must be qualified with its own selected draft and context budget;
the live near-capacity measurements above use MTP only. Dense MTP uses the new
explicit reserve; MTP's existing MoE/QSA allocation contracts are not rewritten here.

## Final verification

`PATH="/opt/cuda/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1 W34_STRICT_TESTS=1
CUDA_DISABLE_PTX_JIT=1 mise run check` passed after the final state-floor fix:
**505 Rust tests passed, 0 failed, 13 ignored**, plus the non-GPU checks, in
**373.40 seconds**. The five worker-capacity tests include plain/MTP long-context
admission and the failing-before/passing-after 27B DFlash2 32k regression.

Code review findings were addressed. A separate state-lifetime investigation
identified the preferred-versus-required DFlash pool distinction; its correction
was verified by the regression and the live cached-prefix smoke above.

Log: `/tmp/w34-speculative-budget-final-check-2.log`; SHA-256
`56487354157aafe1d23ee87e025196b6cb5b7871add888f831ca6ecae7e8d484`.

## Local evidence

- Before/after regression logs: `/tmp/w34-262k-regression-before.log` and
  `/tmp/w34-262k-regression-after.log`.
- Boundary probe: `/tmp/w34-27b-262k-probe.py` and
  `/tmp/w34-27b-262k-probe.log`.
- `/tmp/w34-27b-mtp-262k-fixed/`: request journal, one-second GPU/RSS samples,
  one-second server metrics, before/after summary.

- `/tmp/w34-27b-mtp-192k-reserved/`, `/tmp/w34-27b-192k-probe.py` and
  `/tmp/w34-27b-192k-probe.log`: selected-capacity evidence.
- 192 Ki `summary.json` SHA-256:
  `c4ab0eab0abfe052aa6c647ae19a659303a3e2cc11960d91e59e50c361ae3a6f`.

- Final MTP CLI smoke: `/tmp/w34-27b-mtp-final-smoke/` and matching `.py` / `.log`.
  Summary SHA-256: `1b5726fd3492a90eadffcd4fe6ba45a1480c2c6ba09c94e8b6de444000971e91`.
- DFlash smoke: `/tmp/w34-27b-dflash-budget-smoke/` and matching `.py` / `.log`.
  Summary SHA-256: `8f03829411dac227c4ff0ec7003817ea4d56f19e7168e6f593d5508794619f0e`.

These are local temporary artifacts, not durable release receipts. No eight-hour
27B soak, full-context quality adjudication or concurrency qualification is implied.

# Native MTP FP16 parity investigation

> Historical investigation notes. The temporary `W34_MTP_TRACE_DIR` switch has
> been removed from the runtime before landing; commands below using it document
> how the archived captures were obtained, not a supported serving option.
> Current parity is checked by `phase4_mtp`; the earlier rowwise failures below
> were resolved by batched ingest without widening tolerances. TP2 lossless
> verification also passes in `phase4_gates`.

## Batched ingest update

Native committed ingest now executes one contiguous causal batch (one request), bounded to `min(capacity, 256)` rows. All predictor and shared-head GEMMs preserve M; proposals remain autoregressive M1. This required an MTP-local dense projection helper: shared `LinearProj::forward` intentionally splits M=2..8 into M1 GEMMs. No shared kernels or shared projection policy were modified. Pre-FC norms are concatenated by row, packed QKV is split into contiguous Q/gate, K and V batches, and attention metadata covers all rows. `batch_hidden` exposes all live normalized rows (`batch_hidden_rows`); `hidden` retains only the last row.

The scoped GPU1 test passes both tests, including namespace/state isolation, truncate/replay, k=1/2/3 and both true lossless target-verification rounds. Donor replay now ingests actual call 0 as M5, then calls 1 and 2 as M1. Original fixtures and strict maxabs < .05 / relative L2 < .005 are unchanged.

| Call / row | maxabs | relative L2 |
|---|---:|---:|
| 0 / 0 | .03125 | .0011582343 |
| 0 / 1 | .03125 | .0007692212 |
| 0 / 2 | .03125 | .0007892193 |
| 0 / 3 | .015625 | .0006600141 |
| 0 / 4 | .015625 | .0007080698 |
| 1 / 0 | .03125 | .0008753962 |
| 2 / 0 | .015625 | .0008061633 |

This passes the parity contract, but is not bitwise parity. Full-batch diagnostics identify one pre-FC norm difference among 25,600 elements: row 2, concatenated column 2892 is .93212890625 versus donor .9326171875 (maxabs .00048828125). Full FC maxabs is .00048828125. Row 0 concat, FC, input norm and QKV are now **bitwise identical**; the next captured row-0 divergence is attention output projection (maxabs .00390625, 1465 unequal elements). The existing capture does not distinguish attention-core versus output-projection causes. Full-batch final output maxabs is .03125, relative L2 .0008387233. No numerical policy was changed to eliminate these remaining differences.

Validation (GPU1 only):

```sh
CUDA_VISIBLE_DEVICES=1 cargo test --release -p w34-host --features cuda --test phase4_mtp -- --nocapture
CUDA_VISIBLE_DEVICES=1 W34_MTP_TRACE_DIR=/tmp/mtp-batched-trace cargo test --release -p w34-host --features cuda --test phase4_mtp -- --nocapture
CUDA_VISIBLE_DEVICES=1 python tools/oracle/mtp_diagnostics/compare.py tools/oracle/mtp_diagnostics/call_0_traced.safetensors /tmp/mtp-batched-trace
```

Trace filenames still identify starting cache position, but now contain all rows of that forward. The comparison tool supports both historical row-only and current batch dumps. Raw new dumps remain outside the repository. TP2 execution was not tested.

## Historical rowwise investigation

The original strict parity test remains failing; no tolerance or original expected tensor was changed. Geometry passes, and the k=1/2/3 and both lossless greedy rounds complete before the parity assertion. All GPU work used `CUDA_VISIBLE_DEVICES=1`, TP1. No shared kernels were changed.

The installed vLLM donor was captured again with intermediate `forward` wrappers, not a substitute implementation. Its first three calls' original inputs/outputs are **bitwise identical** to the existing goldens. `call_0_traced.safetensors` is this new actual donor capture (with source hash metadata). `row0/` contains W34's first-row intermediate FP16 dumps on the original captured inputs. These dumps were taken on the failing replay, which overwrites earlier row-zero dumps from the lossless test.

## Measured first-row boundaries

| Boundary | maxabs | relative L2 |
|---|---:|---:|
| concatenated pre-FC norms | 0 | 0 |
| FC | .015625 | .0003204204 |
| input norm | .015625 | .0004486912 |
| QKV projection | .015625 | .0003098840 |
| attention output projection | .0078125 | .0003043860 |
| post-attention norm | .046875 | .0018680213 |
| residual after attention | .015625 | .0019973479 |
| gate/up projection | .015625 | .0011170371 |
| SiLU multiply | .0234375 | .0017991331 |
| down projection | .0078125 | .0017301631 |
| final hidden | .0703125 | .0018632932 |

The first divergence is FC, **not pre-FC normalization**. A CUDA PyTorch counterfactual using exactly the captured FC input and pinned checkpoint weight gives:

* Default reduced-precision FP16 matmul with M=1 reproduces W34 FC **bitwise**, all 2560 elements.
* The same matmul with M=5 reproduces donor FC **bitwise**. Versus W34: maxabs .015625, 869 unequal elements.
* Disabling `allow_fp16_reduced_precision_reduction` changes both results. M=1 differs from donor by .0078125 and from W34 by .015625 (823 elements). M=5 differs from donor by .0078125 and W34 by .015625 (819 elements).
* FP32 matmul rounded to FP16 also differs from donor by .0078125 and W34 by .015625 (823 elements).

Thus FC's discrepancy is concretely reproduced by the M-dependent FP16 GEMM/reduction path: W34 runs one row at a time whereas donor prefill runs five. It is not evidence of a transpose or weight-loading bug. Subsequent amplification occurs at cancellation: donor FC norm=74.77919, attention projection norm=80.57445, sum norm=16.80212. The residual's reference max is only 2.125, versus FC's 37.84375 and attention projection's 39.125. Normalizing this cancellation-sensitive residual propagates errors into the MLP. This is evidence for a numerical cause, not proof that every downstream kernel is identical.

## Full live replay

Call 0 rows 0..4 maxabs: `.0703125, .01586914, .03125, .046875, .016601563`.
Relative L2: `.0018632935, .0009026130, .0009256921, .0009753561, .0008494160`.
Calls 1 and 2: maxabs `.015625, .03125`; relative L2 `.0008899196, .0009388944`.
Only the first row violates the original `.05` absolute cap. Call 3 is not replayed because it contains padding.

## Recommendation / remaining work

Do not patch shared GEMM precision globally just to make this MTP test pass: disabling reduced-precision reduction does not reproduce the donor either. Exact prefill parity would require a batched MTP ingest path matching donor M and attention execution, rather than this eager row-at-a-time component. Shared-kernel work, if pursued, should expose an explicit accumulation policy and test both M=1 and batched M; it should not silently change the parent BF16 DFlash work.

For the intentionally rowwise FP16 implementation, consider a **separately reviewed** scale-aware criterion combining relative L2 <= .005 with maxabs / max(reference maxabs, 1) <= .0025, plus finite checks and greedy verification. First-row normalized max error is .00164721 (reference max=42.6875). This candidate is a recommendation, **not an implemented or validated general acceptance contract**: more prompts, later recurrent calls, cancellation-sensitive rows and donor batch sizes are needed to calibrate it. No parity claim is made.

## Reproduction

```
HF_HUB_OFFLINE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 CUDA_VISIBLE_DEVICES=1 \
  python tools/oracle/vllm_mtp_dump.py --out /tmp/mtp-donor-trace
CUDA_VISIBLE_DEVICES=1 W34_MTP_TRACE_DIR=/tmp/mtp-w34-trace \
  cargo test --release -p w34-host --features cuda --test phase4_mtp -- --nocapture
python tools/oracle/mtp_diagnostics/compare.py \
  tools/oracle/mtp_diagnostics/call_0_traced.safetensors tools/oracle/mtp_diagnostics/row0
CUDA_VISIBLE_DEVICES=1 python tools/oracle/mtp_diagnostics/fc_precision.py \
  tools/oracle/mtp_diagnostics/call_0_traced.safetensors tools/oracle/mtp_diagnostics/row0 \
  ~/.cache/huggingface/hub/models--nchapman--Qwen3.5-4B-W4A16-GPTQ/snapshots/d5874365f7093b28b69379722d2f45d660fcbca6/model_mtp.safetensors
```

The oracle defaults to that pinned local snapshot; never substitute a mutable Hub model ID. The trace switch is diagnostic only, synchronizes and writes raw little-endian FP16 after each boundary, and overwrites files indexed by cache position. Disable it for normal execution. The replay test now collects all live-row mismatches before asserting, without weakening the limits.

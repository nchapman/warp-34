# HGPTQ integration

## Status

Development branch: `feat/hgptq`. **Checkpoint execution is not supported yet.**
The first implementation increment provides a standalone CUDA input-transform
primitive, warmup registration, correctness tests, and a Hadacore comparison
probe. It does not yet attach transforms to loaded projections or replace the
dense vision path.

## Pinned checkpoint inspection

`nchapman/Qwen3.5-2B-W4A16-HGPTQ-g64` at
`0fbb9a38f631ea63678f4b51a491e7ecb26c681c`:

- `quant_method=compressed-tensors`, `format=pack-quantized`, symmetric INT4,
  `group_size=64`, `actorder=static`.
- `model.safetensors`: 248 packed linears = **150 text + 98 vision**.
- `model_mtp.safetensors`: **7 packed MTP linears**.
- The safetensors index references both files, so the existing indexed archive
  loader already discovers the MTP sidecar; no special sidecar merge is needed.
- Neither file contains `g_idx` or transform tensors. Static indices must remain
  optional, while any supplied indices must be validated.
- `transform_config.config_groups.v`: `type=hadamard`, `head_dim=64`,
  `randomize=false`, `precision=torch.float32`, with paired `weight_input`
  (inverse) and `input` (forward) applications. Weight transforms have already
  been applied offline and must not run again at load time.
- The target regexes contain negative lookbehind. Rust's `regex` crate does not
  implement that syntax; target resolution must not silently drop these rules
  or treat unsupported patterns as matches.

The announced 9B/27B variant additionally needs channelwise W4 vision FC2 with
`head_dim=16`, logical K=4304, and FC1 logical N=4304. It has not been qualified
against a real checkpoint here.

## Input-transform contract

`w34_block_hadamard_f16` in
`crates/w34-kernels/csrc/ops/op_hadamard.cu`:

- Contiguous FP16 `[rows, K]` input; separate contiguous FP16
  `[rows, padded_K]` output, with no overlapping byte ranges.
- Normalized, deterministic Sylvester H16 or H64 independently on each logical
  input block. FP32 butterflies, normalization by 1/4 or 1/8, final FP16 rounding.
- Zero-fill the output tail after the logical transform, including the future
  vision K=4304 → 4352 case. Both widths must be block-aligned.
- No allocation, synchronization, transform-weight storage, or random signs.
  Both specializations are registered in `prepare_execution_plan()`.
- Empty batches do not launch. Invalid geometry, null nonempty buffers,
  misalignment, and overlapping buffers are rejected before launch.

FP32 butterflies are a mathematical baseline, **not a bit-exact reproduction of
Hadacore's tensor-core rounding**. The installed vLLM Hadamard module delegates
`type=hadamard` to Hadacore using the activation dtype; its `precision` descriptor
does not make this path FP32 end-to-end. Model-level numerical and greedy gates
remain necessary before accepting this arithmetic policy for inference.

## Execution integration still required

1. Resolve per-linear quantization and transform rules at load time. Preserve
   grouped/channelwise distinctions and reject unsupported transform semantics.
   The current config representation collapses linear groups to one scheme.
2. Carry input-transform metadata through host weight preparation, sharding,
   upload, and projection dispatch. An untransformed dense fallback of rotated
   weights is not valid. Reject TP K-slices that split Hadamard blocks.
3. Allocate explicit stream-safe transform scratch. Preserve unrotated inputs
   for GDN `in_proj_a/b` and other shared consumers. Share transformed inputs
   only when source, shape, block size, and lifetime all agree.
4. Add native channelwise W4 Marlin support (currently only W8 exposes the
   channelwise specialization) and distinguish logical from physical dimensions.
   Zero-valued padding in U4B8 packing is nibble 8, not nibble 0.
5. Replace quantized vision linears' dense-only load/dispatch with biased
   quantized projections, retaining the original unquantized patch embedding.
   Mask/crop padded FC1 output and transform logical FC2 blocks before K padding.
6. Verify text, MTP, vision, graphs, and TP=1/2 against the pinned checkpoint;
   then profile shared-transform reuse and producer fusion before modifying
   Marlin's inner loop.

The donor oracle must account for the user-reported vLLM vision fixes: allow bias
for input-only transforms and iterate `self.input_tfms` for unfused QKV. The
installed vLLM `HadamardTransform` also explicitly rejects TP > 1, so TP=2 needs
comparison with a TP=1 oracle plus engine cross-TP checks, not an assumed vLLM
TP=2 baseline.

## Reproduce the primitive checks

```sh
export PATH="/opt/cuda/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda \
  --test hgptq_hadamard -- --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --release --features cuda \
  --test hgptq_hadamard benchmark_block_hadamard -- --ignored --nocapture
```

`tools/oracle/hgptq_hadamard_probe.py` documents how to build a standalone shared
library and compare it with the installed vLLM Hadacore implementation. Both
paths are timed with CUDA graphs; the native H16 path includes K padding while
the donor transform alone does not. These are op timings, not model speedups.

Initial RTX 3090 Ti (`sm_86`) probe, random FP16 activations:

| Block / shape | Native µs | Hadacore µs | Relative L2 error |
|---|---:|---:|---:|
| H64, M=1, K=2048 | 1.259 | 1.135 | 3.59e-4 |
| H64, M=16, K=2048 | 1.326 | 1.211 | 3.72e-4 |
| H64, M=512, K=2048 | 5.345 | 3.641 | 3.72e-4 |
| H64, M=4096, K=6144 | 122.452 | 112.306 | 3.72e-4 |
| H16, M=1, K=4304 → 4352 | 1.151 | 3.285 | 2.58e-4 |
| H16, M=512, K=4304 → 4352 | 17.072 | 8.519 | 2.69e-4 |

Across all 15 probe shapes, max absolute error was at most 0.00390625 and relative
L2 at most 3.77e-4. The baseline is **not yet performance-qualified**: H64 and
large-M H16 need optimization or fusion. No `sm_89` runtime measurements or
end-to-end HGPTQ results have been obtained.

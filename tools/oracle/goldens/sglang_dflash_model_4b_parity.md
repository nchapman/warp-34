# Native DFlash BF16 parity evidence

## Oracle integrity and scope

`sglang_dflash_model_4b.safetensors` is the original actual SGLang v0.5.19
`DFlash2DraftModel` capture with an explicit FlashInfer FA2 attention adapter.
It is **not** an end-to-end DFlashWorkerV2 serving oracle. Its tensors and
provenance were not changed to resolve numerical differences.

The independent `sglang_dflash_model_4b_operators.safetensors` capture contains
real checkpoint operands and outputs; all 52 shared tensors match the original
capture bitwise. Norm operands were captured before in-place mutation in the
separate `sglang_dflash_model_4b_norms.safetensors` capture.

## Verified full-model and operator results

- `phase4_dflash actual_sglang_full_model_hidden_parity` checks original input
  tokens, anchor, and bitwise target taps, then compares final hidden states.
  It now runs by default and measured **relL2=0, maxabs=0** for this 4B capture.
  This is not a claim of bitwise parity for every intermediate or every prompt.
- `phase4_bf16_attention` measured bit-exact outputs for each of five layers
  when replaying the donor's attention operands at head dimension 256.
- All 20 real-shape grouped-convolution replays and the captured norm outputs
  measured zero error. QK/RoPE replays pass the unchanged 1e-4 relative-L2 bound;
  they are not all bit-exact. The four layer-0 GEMMs and five SiLU replays match
  their captured outputs bitwise.

Mechanisms corrected:

1. Production RoPE uses an FP32 cos/sin cache even with BF16 activations.
   `w34_dflash_qk_norm_rope_bf16_f32_table` is declared in the shared CUDA header
   and Rust FFI and registered in execution-plan warmup. The BF16-table variant
   remains available for its separate component contract.
2. FA2 head-dimension-256 attention uses 16-key tiles, BF16-rounded softmax
   probabilities, Tensor Core QK/PV/rowsum contractions, and approximate PTX
   exponentials and reciprocals.
3. Grouped convolution requires separately rounded multiply and add; norm
   reductions match the donor's thread geometry and ascending butterfly order.

## Head-dimension-128 and split-KV adjudication

A fresh replay in `lmsysorg/sglang:v0.5.19` on RTX 3090 Ti reproduced all four
`sglang_dflash_attention.safetensors` outputs bitwise. CUDA profiling showed
`KernelTraits<...,64,1,4,8,8,4,1,...>`: **NUM_MMA_KV=4**, hence 64-key KV tiles,
not the 16-key tiles of HD=256. Native attention now uses that tile width and
16-key contractions within each tile. The short and full strict fixture cases
then passed without changing their expected outputs or tolerance.

The 2,064-key case additionally launches a split-KV merge over 17 partitions.
FlashInfer's own auto-split vs explicitly unsplit outputs differ by
**maxabs=0.0009765625, relL2=0.00378025**; 70 elements exceed the gate's
`0.008*abs(expected)+0.0002` bound. This is partial-output BF16 rounding, not a
mask mismatch. Native attention is unsplit, so its strict comparison uses a
new actual donor capture generated with `disable_split_kv=True` by
`sglang_dflash_unsplit_dump.py`. Original auto-split tensors remain unchanged
and retain their separate coverage in `phase4_draft_kernels`.

`sglang_dump.py` now bundles only actual captures, fails on missing inputs or
provenance, and preserves their hashes and metadata. It no longer manufactures
the incorrect Phase 0 block-end-mask reference. The bundle contains original
attention outputs and explicitly named `*.unsplit_out` outputs.

The focused `sglang_dflash2_fixtures_operator_dump_diff_parity` gate passes with
the original tolerance. This does not establish bitwise default split-KV
serving parity, full-suite success, or a performance improvement.

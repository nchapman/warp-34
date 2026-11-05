# Native DFlash BF16 parity evidence

## Oracle integrity and scope

`sglang_dflash_model_4b.safetensors` is the actual SGLang v0.5.19
`DFlash2DraftModel` capture with an explicit FlashInfer FA2 attention adapter,
including pre-mutation RMSNorm operands (`norms.*`) and real-shape operator
operands (`ops.*`). It is **not** an end-to-end DFlashWorkerV2 serving oracle.
Its tensors and provenance were not changed to resolve numerical differences.

## Verified full-model and operator results

- `dflash_proposer actual_sglang_full_model_hidden_parity` checks original input
  tokens, anchor, and bitwise target BF16 taps (`w34_tap_add_f16_to_bf16`), then
  compares final hidden states against the `< 0.005` relative-L2 gate (`relL2=0,
  maxabs=0` measured on this 4B capture).
- `dflash_bf16_attention` measured bit-exact outputs for each of five layers
  when replaying the donor's attention operands at head dimension 256.
- All 20 real-shape grouped-convolution replays, captured norm outputs,
  QK/RoPE replays, layer-0 GEMMs, and five SiLU replays pass their `< 1e-4`
  relative-L2 bounds (with V columns in QK/RoPE verified bitwise).

Mechanisms verified:

1. Production RoPE uses an FP32 cos/sin cache even with BF16 activations.
   `w34_dflash_qk_norm_rope_bf16_f32_table` and
   `w34_dflash_qk_norm_rope_bf16_f32_table_qout` are declared in the shared CUDA
   header and Rust FFI and registered in execution-plan warmup. The BF16-table
   variant remains available for its separate component contract.
2. FA2 head-dimension-256 attention uses 16-key tiles, BF16-rounded softmax
   probabilities, Tensor Core QK/PV/rowsum contractions, and approximate PTX
   exponentials and reciprocals.
3. Grouped convolution requires separately rounded multiply and add; norm and
   QK-norm reductions use single-warp FP32 butterfly reductions.

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
and retain their separate coverage in `dflash_draft_kernels`.

`sglang_dump.py` now bundles only actual captures, fails on missing inputs or
provenance, and preserves their hashes and metadata. It no longer manufactures
synthetic block-end-mask references. The bundle contains original
attention outputs and explicitly named `*.unsplit_out` outputs.

The focused `sglang_dflash2_fixtures_operator_dump_diff_parity` gate passes with
the original tolerance. This does not establish bitwise default split-KV
serving parity, full-suite success, or a performance improvement.

# Vision FA2 dependency closure

This is donor code, not an independently authored numerical implementation.

- FA2 headers and torch-free compatibility stubs: `warp3090/vendor/vllm.cpp`,
  subsystem `src/vt/cuda/flash_attn`, donor project pin `50d425bec`.
  The donor annotates its upstream as `vllm-project/flash-attention@2c839c33`;
  that upstream revision was not independently checked out here.
- CUTLASS/CUTE: NVIDIA CUTLASS 4.5.0, checkout
  `e406c186f510a15091cce01f782020ceb7ba8eb5`.
- Selected using CUDA 13.4 `nvcc -M` on the proven attention-only adapter
  including `flash_fwd_launch_template.h`, the compatibility prelude, and
  CUTLASS headers. Only dependencies under those two donor roots were copied;
  system/CUDA toolkit headers remain toolchain dependencies. No vendor source
  translation units or tests are compiled.
- Original notices retained in every header. FA2 BSD-3-Clause LICENSE/AUTHORS
  were supplied from the existing mistral.rs flash-attention distribution;
  CUTLASS LICENSE.txt comes from the pinned CUTLASS checkout.
- `SOURCE_MANIFEST.sha256` records copied source contents (relative paths).

Production `op_vision_fa2.cu` invokes the unchanged donor forward kernel directly:
D64 traits `<64,128,128,4,false,false,half_t>`, even-K=true; logical D72 traits
`<96,128,64,4,false,false,half_t>`, even-K=false. All other boolean parameters
are false. This is exactly the donor launch-template selection for noncausal,
non-dropout varlen vision attention. No fragment-order rewrite. CUDA errors
are checked directly; the donor's unused launch-wrapper stubs are not called.
No Torch or external native library is linked.

The build exposes these includes only to op_vision_fa2.cu. All vendor file
changes (including .hpp) participate in that object's dependency epoch.

Verification: tools/oracle/vision_fa2_probe.py compares packed post-RoPE input
against actual installed vLLM FA2 and a full captured D72 native output, with
strict half-bit equality, including segmented images and CUDA graph replay.
The capture is an external test artifact, not distributed here.

# Collectives & Cold-Start `TP=2` Initialization Contract

## 1. Hardware Substrate (`2× RTX 3090 Ti`, PCIe Gen4 x8, `topo PHB`, No P2P)

On our target dual-3090 Ti platform (`docs/HARDWARE.md` §1, §11):
- NVLink is absent on RTX 3090 Ti.
- `cudaDeviceCanAccessPeer` is `0` in both directions (`topo PHB` on AMD Ryzen root complex).
- Cross-GPU communication uses a **graph-capturable pinned-sysmem two-hop exchange** (`cudaHostAllocMapped | cudaHostAllocPortable`) with system-scope release/acquire fences (`__threadfence_system()`, `ld.global.acquire.sys`, `st.global.release.sys`) and monotonically increasing epoch parity flags.

## 2. Cold-Start `TP=2` Initialization State Machine (`prepare_execution_plan()`)

CUDA 13 lazy module loading (`CUDA_MODULE_LOADING=LAZY`) acquires a global runtime/context mutex the first time a kernel specialization's cubin/fatbin is JIT-loaded on a device. If Rank 0 enters a spin-waiting collective kernel (`all_reduce`) while Rank 1 is still lazily loading an upstream kernel cubin, Rank 0 hogs the GPU/PCIe bus or blocks driver progress, causing a deterministic cold-start deadlock.

Every `TP=2` engine instance MUST transition through the 6-stage `InitPhase` state machine via `prepare_execution_plan()` before serving any request:

```
UNINITIALIZED
  -> (1) ContextsCreated       : Bind device 0 & device 1, create compute/copy/comm streams & cuBLAS handles
  -> (2) ArenasAllocated       : Preallocate DeviceArena on each GPU + shared mapped pinned-sysmem comm arena
  -> (3) ModulesWarmedPerRank  : Independently on each GPU (NO cross-rank sync/collectives), launch 1 dummy
                                 invocation of every reachable kernel specialization (Marlin, Norms, RoPE,
                                 Attn, GDN conv/chunk/recurrent, Sampler, Tap copy) to force cubin loading
  -> (4) DevicesSynchronized   : cudaDeviceSynchronize() on BOTH GPUs + host barrier (all cubins now resident)
  -> (5) CollectivesWarmed     : Launch paired FP16, BF16, FP32, and INT32 sysmem collectives across both ranks
                                 and synchronize both streams
  -> (6) GraphsCaptured / READY: Capture composite GraphKey families and enter READY state
```

Calling a multi-GPU collective before `InitPhase >= DevicesSynchronized` returns `Err(CudaError::ColdStartNotWarmed)` in Rust and `W34_ERR_COLD_START_NOT_WARMED` across FFI.

## 3. Typed Collective Entrypoints (No `FP16` Casts for Integer or Loss-Sensitive Data)

`w34-kernels` exposes four distinct typed collective operations over the mapped pinned-sysmem exchange buffer:

1. **`w34_tp2_all_reduce_f16` (`__half`):**
   - Sum-reduction across Rank 0 and Rank 1 for target model hidden states (`o_proj`, `down_proj`, `in_proj` outputs), accumulating in `FP32` registers before writing `FP16` output.
   - Also provides fused **`w34_tp2_all_reduce_residual_rms_norm_f16`** (all-reduce + residual add + `GemmaRMSNorm` in one pass for small-batch decode/verify).
2. **`w34_tp2_all_reduce_bf16` (`__nv_bfloat16`):**
   - Sum-reduction in `FP32` with `BF16` I/O for `BF16` drafter layers (`4B DFlash2`).
3. **`w34_tp2_all_gather_top1_f32_i32` / `w34_tp2_all_reduce_f32` (`float` + `int32_t`):**
   - Used for distributed argmax / logprob normalization / top-K candidate selection across vocabulary-sharded `ParallelLMHead` (`(local_max_f32, global_vocab_id_i32)` per token row).
4. **`w34_tp2_broadcast_i32` (`int32_t`):**
   - Broadcasts sampled token IDs, accepted draft counts (`a in 0..=k`), and scheduler control words from Rank 0 to Rank 1 with exact 32-bit integer fidelity.

**Hard Prohibition:** Token IDs (`vocab_size = 248,320 > 65,504` on Qwen3.5/3.8) and counters MUST NEVER be cast to `FP16` (`__half` saturates at `65,504` and loses integer precision above `2048`).

## 4. System-Scope Memory Ordering & CUDA Graph Capture Compatibility

- Each collective call uses a dedicated monotonic step counter (`uint32_t* d_epoch` in device memory, incremented inside the kernel at completion) so the exact same kernel launch sequence can be captured once in a `cudaGraph_t` and replayed arbitrarily many times without host intervention.
- Writers issue `__threadfence_system()` after writing payload lines to mapped sysmem, then write the current `epoch` tag to the peer-visible flag word using `st.global.release.sys.b32`.
- Readers spin on the peer's flag word using `ld.global.acquire.sys.b32` until `peer_flag == epoch`, then read the payload lines.

# Collectives & Cold-Start `TP=2` / `TP=4` Initialization Contract

## 1. Hardware Substrate (`Pinned-Sysmem PCIe` & `NVLink P2P VRAM`)

`Tp2Collectives` (`crates/w34-host/src/cuda/collectives.rs` & `crates/w34-kernels/csrc/ops/op_collectives.cu`) supports both `TP=2` and `TP=4` across two transport substrates:
- **Pinned-Sysmem Two-Hop (`CommBuffer::MappedHost`):** On non-P2P PCIe platforms (such as `2× RTX 3090 Ti` on AMD Ryzen `topo PHB`, where NVLink is absent and `cudaDeviceCanAccessPeer == 0`), communication uses graph-capturable mapped pinned host memory (`cudaHostAllocMapped | cudaHostAllocPortable`) with system-scope release/acquire fences (`__threadfence_system()`, `ld.global.acquire.sys`, `st.global.release.sys`) and monotonically increasing epoch counters.
- **1-Hop NVLink / P2P VRAM (`CommBuffer::DeviceVram`):** When `w34_device_can_enable_p2p` confirms bidirectional P2P access across all peer pairs (e.g. `2× RTX 3090` over bonded `NV4` NVLink `112.5 GB/s`), `Tp2Collectives::new_for_tp_with_p2p` allocates signal and staging buffers in device VRAM (`DeviceBuffer`) so peer kernels perform direct 1-hop NVLink VRAM reads/writes.

## 2. Cold-Start `TP=2` / `TP=4` Initialization State Machine (`prepare_execution_plan()`)

CUDA 13 lazy module loading (`CUDA_MODULE_LOADING=LAZY`) acquires a global runtime/context mutex the first time a kernel specialization's cubin/fatbin is JIT-loaded on a device. If Rank 0 enters a spin-waiting collective kernel (`all_reduce`) while another rank is still lazily loading an upstream kernel cubin, Rank 0 hogs the GPU/PCIe bus or blocks driver progress, causing a deterministic cold-start deadlock.

Every multi-GPU engine instance MUST transition through the 6-stage `InitPhase` state machine via `prepare_execution_plan()` before serving any request:

```
UNINITIALIZED
  -> (1) ContextsCreated       : Bind all TP devices, create compute/copy/comm streams & cuBLAS handles
  -> (2) ArenasAllocated       : Preallocate DeviceArena on each GPU + shared comm arena (MappedHost or DeviceVram)
  -> (3) ModulesWarmedPerRank  : Independently on each GPU (NO cross-rank sync/collectives), launch 1 dummy
                                 invocation of every reachable kernel specialization (Marlin, Norms, RoPE,
                                 Attn, GDN conv/chunk/recurrent, MoE, QSA, PLE, HC, Sampler, Tap copy)
  -> (4) DevicesSynchronized   : cudaDeviceSynchronize() on ALL GPUs + host barrier (all cubins now resident)
  -> (5) CollectivesWarmed     : Launch paired FP16, BF16, FP32, and INT32 collectives across all ranks
                                 and synchronize all streams
  -> (6) GraphsCaptured / READY: Capture CUDA graph families and enter READY state
```

Calling a multi-GPU collective before `InitPhase >= DevicesSynchronized` returns `Err(CudaError::ColdStartNotWarmed)` in Rust and `W34_ERR_COLD_START_NOT_WARMED` across FFI.

## 3. Typed Collective Entrypoints (No `FP16` Casts for Integer or Loss-Sensitive Data)

`w34-kernels` exposes seven typed collective operations (each with `w34_tp2_*` and `w34_tp4_*` variants):

1. **`w34_tp{2,4}_all_reduce_f16` (`__half`):**
   - Sum-reduction across all TP ranks for target model hidden states (`o_proj`, `down_proj`, `in_proj` outputs), accumulating in `FP32` registers before writing `FP16` output.
2. **`w34_tp{2,4}_all_reduce_residual_gemma_rms_norm_f16` (`__half`):**
   - Fused all-reduce + residual add + `GemmaRMSNorm` (`1.0f + w`) in a single kernel launch across all hidden-size regimes (`H <= 2560`, `H == 4096`, `H == 5120`, and looped `H > 5120`).
3. **`w34_tp{2,4}_all_reduce_bf16` (`__nv_bfloat16`):**
   - Sum-reduction in `FP32` with `BF16` I/O for `BF16` drafter layers (`4B DFlash2`).
4. **`w34_tp{2,4}_all_reduce_f32` (`float`):**
   - Exact `FP32` sum-reduction across ranks.
5. **`w34_tp{2,4}_all_gather_top1_f32_i32` (`float` + `int32_t`):**
   - Distributed argmax across vocabulary-sharded `ParallelLMHead` (`(local_max_f32, global_vocab_id_i32)` per token row), breaking ties deterministically toward the lowest global token ID.
6. **`w34_tp{2,4}_broadcast_i32` (`int32_t`):**
   - Broadcasts sampled token IDs, accepted draft counts (`a in 0..=k`), and scheduler control words from Rank 0 to peer ranks with exact 32-bit integer fidelity.
7. **`w34_tp{2,4}_gather_vocab_f16` (`__half`):**
   - Gathers vocabulary-sharded `[num_rows, local_vocab]` `FP16` logits into a full `[num_rows, total_vocab]` tensor on Rank 0 for stochastic sampling (`SamplerV1`) and `RejectionSampler`.

**Hard Prohibition:** Token IDs (`vocab_size = 248,320 > 65,504` on Qwen3.5/3.8) and counters MUST NEVER be cast to `FP16` (`__half` saturates at `65,504` and loses integer precision above `2048`).

## 4. System-Scope Memory Ordering & CUDA Graph Capture Compatibility

- Each collective call uses a dedicated monotonic step counter (`uint32_t* d_epoch` in device memory, incremented inside the kernel at completion) so the exact same kernel launch sequence can be captured once in a `cudaGraph_t` and replayed arbitrarily many times without host intervention.
- Writers issue `__threadfence_system()` after writing payload lines to mapped sysmem, then write the current `epoch` tag to the peer-visible flag word using `st.global.release.sys.b32`.
- Readers spin on the peer's flag word using `ld.global.acquire.sys.b32` until `peer_flag == epoch`, then read the payload lines.

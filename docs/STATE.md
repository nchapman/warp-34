# warp-34 — Session Orientation & Current State

## Status: Phase 0 Ready (`main`)

`warp-34` is initialized with a compiling Rust + CUDA (`sm_86`) workspace (`crates/w34-kernels`, `crates/w34-host`), `mise` task runner, and an SME-verified engineering roadmap (`docs/ROADMAP.md`).

## Active Phase: Phase 0 (`docs/ROADMAP.md` §4)

Start immediately on **Phase 0**:
1. **0.1 Foundational Seam Contracts (`docs/contracts/`):**
   - `execution-contract.md` (4-length transaction `processed_len`/`emitted_len`/`reserved_len`/`draft_context_len`, token flattening, `valid_token_mask`, composite `GraphKey`, static `TapPlan`)
   - `cache-layout.md` (`KvLayerLayout`, `AttentionBatch`, logical `page_id`s, cache namespaces, COW/ownership rules)
   - `state-transaction.md` (GDN `StateBundle`, `state_in_indices`, `spec_state_indices`, `checkpoint_dst_indices`, chunk geometry `B=16`, `BT=64`, `C=k*64`)
   - `collectives.md` (`prepare_execution_plan()` cold-start `TP=2` init state machine, system-scope fences, typed payloads `FP16`/`BF16`/`FP32`/`INT32`)
   - `precision-policy.md` (`FP16` target spine, `BF16` drafter preservation for 4B DFlash2, `FP32` persistent `ssm_state`/reductions)
2. **0.2 `vLLM` & `SGLang` Oracle & `nsys` Diff Harness (`tools/oracle/`):**
   - `vllm_dump.py` & `sglang_dump.py` (per-op/per-layer tensor dumps, `DFlash2` target hidden taps, partial-rejection GDN states)
   - `vllm_greedy.py` (12-prompt greedy token + top-5 logprob generator)
   - `profile_diff.py` (automated side-by-side `nsys` SQLite kernel category scoreboard vs `vLLM 0.31.0` and `SGLang v0.5.19`)
3. **0.3 CUDA Runtime, `prepare_execution_plan()`, & `CapacityPlanner` (`w34-kernels`, `w34-host::cuda`)**
4. **0.4 Multi-Format Safetensors & HF Hub Loader (`w34-host::loader`)**

## Reference Repositories & Cached Models on Disk

- **Prior-gen reference repo:** `/home/nchapman/Code/warp3090` (donor for `TQ4` in `crates/w3090-kernels/csrc/ops/op_tq4kv.cu`, `fattn-vec-tq.cuh`, `op_fattn_mma.cu`, `docs/contracts/kv-tq.md`, `docs/contracts/dflash2.md`, and pinned-sysmem `TP=2` AllReduce in `op_all_reduce.cu`).
- **`vLLM 0.31.0` (`8352b2427`):** Installed at `~/.local/share/mise/installs/python/3.12.12/lib/python3.12/site-packages/vllm/` and `/home/nchapman/Code/warp3090/vendor/vllm`.
- **`vllm.cpp` (`50d425bec`):** `/home/nchapman/Code/warp3090/vendor/vllm.cpp` (donor for native CUDA C++ GDN kernels in `src/vt/cuda/cuda_gdn.cu`).
- **`SGLang v0.5.19` (`0bcd822377`):** `/home/nchapman/Code/warp3090-dflash/vendor/sglang` and Docker image `lmsysorg/sglang:v0.5.19`.
- **Cached HF Checkpoints (`~/.cache/huggingface/hub/`):**
  - `models--nchapman--Qwen3.5-4B-W4A16-GPTQ`
  - `models--nchapman--Qwen3.5-9B-W4A16-GPTQ`
  - `models--nchapman--Qwen3.8-27B-W4A16-GPTQ` (`0df81ea`)
  - `models--incoai--Qwen3.8-27B-DFlash2` (`015e795`)
  - `models--llmfan46--Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4` (`fb685a8`)

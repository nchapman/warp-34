# cuBLAS packaging experiment

**Status: Ampere validation completed; RTX 4090 validation pending.** This report
covers the proposed Docker packaging change, not full 30-/40-series release
qualification. The Dockerfile uses pruned static cuBLAS; normal local Cargo builds
still use the shared libraries unless `W34_CUBLAS_STATIC_DIR` is set. The user
approved committing this change with the known unrelated test failure documented;
a separate agent will perform the 4090 qualification.

## Prior art and pruning boundary

- [NVIDIA cuBLAS: nvprune](https://docs.nvidia.com/cuda/cublas/index.html#nvprune)
  requires keeping the selected architecture and all earlier minor architectures
  in the same major architecture.
- [Hugging Face TEI's CUDA Dockerfile](https://github.com/huggingface/text-embeddings-inference/blob/main/Dockerfile-cuda)
  prunes `libcublas_static.a` and keeps `sm_80` alongside its selected 8.x target.
- [NVIDIA static-library support](https://docs.nvidia.com/cuda/cublas/index.html#static-library-support)
  documents the `culibos` dependency. This engine also directly uses cuBLASLt.
- [llama.cpp issue #15620](https://github.com/ggml-org/llama.cpp/issues/15620)
  records a static-link failure when `cublasLt_static` was omitted.

For our pinned CUDA 13.4.1 / cuBLAS 13.8.0.4 installation, keep **sm_80, sm_86,
and sm_89** in both archives. Also preserve the six compute_75 PTX fallbacks in
cuBLASLt. Classic cuBLAS has no sm_89 cubins; Ada relies on compatible earlier
8.x code. Do not prune only to the application's sm_86/sm_89 targets, remove
relocatable ELF, or use whole-archive linkage.

The pruned image is restricted to the intended RTX 30-/40-series targets. The
application's PTX fallback does not make the pruned cuBLAS library suitable for
other GPU generations. Re-inventory cuBLAS architectures after toolkit updates.

## Size measurements

Fresh same-source images, with the preceding CUDA binary compression enabled:

| Packaging | Executable bytes | Runtime filesystem bytes |
|---|---:|---:|
| Shared cuBLAS baseline (`77699cb`) | 39,614,600 | 775,577,600 |
| Unpruned static cuBLAS | 577,632,848 | 711,024,640 |
| Pruned static cuBLAS | 442,053,712 | 575,447,040 |

Runtime filesystem measurements use `du -sx -B1 /` inside each container, not
Docker's local storage accounting or compressed registry download sizes.
Static linking alone saves ~64.6 MB; pruning saves another ~135.6 MB. Combined:
**~200.1 MB / 25.8% less runtime filesystem space** versus shared cuBLAS.

`ldd` confirms the pruned executable no longer needs either cuBLAS shared library.
It still uses shared cudart and the normal C++/glibc runtime. Final cubin inventory:
328 sm_80, 155 sm_86, 83 sm_89 entries (application plus linked library code).
Archive-member elimination explains why these counts are below the input archives.

## Initial GPU measurements

RTX 3090 Ti, driver 615.71.09; 4B GPTQ, TQ4 KV, CUDA graphs, 128 prompt tokens and
64 generated tokens. Three alternating runs per image/profile, with
`CUDA_DISABLE_PTX_JIT=1`. Plain, MTP-3, and native-BF16 DFlash2 generated token IDs
match between shared and pruned images across all runs. The vision benchmark
passes cold/warm lossless parity and image-content cache isolation in all runs.

| Profile | Shared median decode tok/s | Pruned median decode tok/s |
|---|---:|---:|
| Plain | 210.44 | 210.53 |
| MTP-3 | 528.22 | 528.45 |
| DFlash2 | 1,280.81 | 1,289.72 |
| Vision, warm | 196.73 | 196.65 |

Plain/MTP/DFlash2 median TTFT remained ~20.3–20.7 ms. Vision median cold TTFT
was 34.87 → 34.31 ms; warm TTFT was 17.35 → 17.42 ms. Treat these small differences
as short-run variability, not a speedup claim. Process wall times include Docker
startup and do not isolate cold CUDA module-loading overhead.

Full static-link verification completed with `W34_CUBLAS_STATIC_DIR` set and
`CUDA_DISABLE_PTX_JIT=1`: **438 passed, 1 failed, 11 ignored**, including the
single-/dual-GPU, vision bias/rounding, and BF16 drafter tests. The sole failure
is the same pre-existing `qwen4_flash_next.rs:1093` hot-cache assertion seen with
shared cuBLAS (actual 128, expected 120), being fixed separately. Formatting,
Clippy in both feature modes, rustdoc, dashboard (18), oracle tooling (69), and
performance tooling (91) checks passed. `readelf` on the vision test executable
confirms these tests used static cuBLAS rather than accidentally linking the
shared libraries. These short benchmark runs do not establish broad model
performance coverage or Ada performance parity.
No actual RTX 40-series or minimum-driver qualification has been performed.

## Reproduce

```sh
# Experimental pruned image (current Dockerfile).
docker build --platform linux/amd64 -t warp-34:cublas-pruned-experiment .

# Isolate static-link savings from pruning savings.
docker build --platform linux/amd64 \
  --build-arg W34_CUBLAS_STATIC_DIR=/usr/local/cuda/lib64 \
  -t warp-34:cublas-static-unpruned .

# Equivalent local static-link checks. Set CUDA_HOME to your CUDA 13.4.1 toolkit.
export CUDA_HOME=/opt/cuda
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p /tmp/w34-cublas-pruned
for lib in cublas cublasLt; do
  nvprune --generate-code code=sm_80 --generate-code code=sm_86 \
    --generate-code code=sm_89 --generate-code code=compute_75 \
    "$CUDA_HOME/lib64/lib${lib}_static.a" \
    -o "/tmp/w34-cublas-pruned/lib${lib}_static.a" || exit 1
done
# culibos comes from CUDA_HOME/lib64. Use CUDA_VISIBLE_DEVICES=0 on one GPU.
W34_CUBLAS_STATIC_DIR=/tmp/w34-cublas-pruned \
  CUDA_VISIBLE_DEVICES=0,1 CUDA_DISABLE_PTX_JIT=1 mise run check
```

The local static directory is validated before linkage and its archive timestamps
are tracked by Cargo. Unset `W34_CUBLAS_STATIC_DIR` to return local builds to shared
cuBLAS. An empty or invalid directory intentionally fails rather than silently
falling back to shared libraries, which the experimental Docker runtime omits.
The packaging change deliberately leaves Ubuntu, the existing non-root runtime,
healthcheck, model cache, and driver capability requirement unchanged.

## RTX 4090 handoff / remaining qualification

Compare this branch's pruned image against the shared-library baseline at
`77699cb`, built in a separate worktree with the same pinned toolchain. Do not use
an older published image as the comparator: its engine code may differ. Record
GPU model, driver version, image IDs, and source commits with the results.

1. Build both images and inspect `ldd` / `cuobjdump` output. The pruned executable
   must not depend on shared cuBLAS, and sm_80/sm_86/sm_89 code must remain present.
2. On the 4090, repeat the 4B GPTQ/TQ4 benchmark with `--prompt-len 128 --gen-len 64`:
   plain `--stage decode`, MTP `--stage decode --speculative mtp --mtp-k 3`,
   DFlash2 `--stage decode --speculative dflash2 --draft-model nchapman/Qwen3.5-4B-DFlash2`,
   and `--stage vision`. Use at least three alternating runs per image/profile,
   `--offline`, and `CUDA_DISABLE_PTX_JIT=1` in the container. Compare generated
   token IDs, acceptance counts, TTFT, decode throughput, and vision parity.
3. Run the static-link suite using the commands above. Record skipped multi-GPU
   tests if only one 4090 is available. Exercise cuBLAS FP16/BF16, cuBLASLt vision
   bias/rounding, and graph capture/replay; check both GPUs and TP=2 if available.
4. Confirm HTTP health, dashboard, chat completion, and named-volume model-cache
   access using the README's container recipes. Distinguish first-use/module-load
   latency from warm throughput if startup differs.
5. Update this report and `STATE.md` with actual Ada results before declaring
   40-series qualification. Investigate any new numerical failures or repeatable
   regressions; do not explain them away using the unrelated hot-cache assertion.

The existing hot-cache failure is expected only while its separate fix is absent.
If integrating that fix for testing, record the resulting source revision rather
than relabeling the original measurements. Minimum-driver qualification remains
separate from the 4090 hardware check.

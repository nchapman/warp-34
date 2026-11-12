# cuBLAS packaging experiment and decision

## Decision: keep shared cuBLAS

The default Docker image uses **shared cuBLAS/cuBLASLt in a separate reusable
layer**, followed by the application executable. CUDA device-code compression,
Ubuntu, non-root execution, cache setup, and the other Docker improvements remain.
The experimental static-link build option has been removed; there are no separate
static/shared release variants to maintain.

Pruned static linkage saved ~200 MB of unpacked runtime space, but grew the
executable from ~40 MB to ~442 MB. Docker downloads entire changed layers, not
binary patches within them. With shared libraries and stable CUDA/runtime layers,
ordinary application updates reuse the large library layer and replace only the
small executable layer. Multiple installed versions can also share those layers.
Actual downloads are compressed; executable and filesystem sizes below are not
transfer-size measurements.

For frequent engine updates, this is a better trade than minimizing first-install
size. Shared cuBLAS also avoids architecture-pruning maintenance and its added
GPU qualification burden. Keep the CUDA libraries separate from `/w34` and avoid
unnecessary base/package churn, while still rebuilding for security updates.
Ubuntu's ~115 MB unpacked base was explicitly accepted and remains unchanged.

The final shared-image rebuild after main integration confirmed the intended
layer separation: its first five filesystem layer digests match the earlier
shared image, including the ~603 MB CUDA-library layer; only the final ~39.6 MB
executable layer changed. Shared-library linkage and CPU-only CLI startup passed.
This confirms stable filesystem layers across this application update, not a
measurement of compressed registry transfer size.

Final shared-library verification on both RTX 3090 Ti GPUs with PTX JIT disabled
passed the full `mise run check`: **441 Rust tests passed, 0 failed, 11 ignored**,
plus formatting, both Clippy modes, rustdoc, dashboard, oracle tooling, and
performance tooling. An earlier attempt was interrupted by another worktree's
GPU workload exhausting available VRAM; the uncontended retry passed. Review
found no actionable issues. Actual RTX 4090 qualification remains pending.

## Experiment history and prior art

The static experiment is preserved in commit `114218f`; `3088850` integrates
main's `d9ecc01` fixes and records successful merged-tree static validation.
Use a detached worktree at `3088850` to reproduce the old static build rather
than passing the removed `W34_CUBLAS_STATIC_DIR` option to current sources.

- [NVIDIA cuBLAS: nvprune](https://docs.nvidia.com/cuda/cublas/index.html#nvprune)
  requires keeping the selected architecture and all earlier minor architectures
  in the same major architecture.
- [Hugging Face TEI's CUDA Dockerfile](https://github.com/huggingface/text-embeddings-inference/blob/main/Dockerfile-cuda)
  prunes `libcublas_static.a` and keeps `sm_80` alongside its selected 8.x target.
- [NVIDIA static-library support](https://docs.nvidia.com/cuda/cublas/index.html#static-library-support)
  documents the `culibos` dependency. This engine also directly uses cuBLASLt.
- [llama.cpp issue #15620](https://github.com/ggml-org/llama.cpp/issues/15620)
  records a static-link failure when `cublasLt_static` was omitted.

The experiment used CUDA 13.4.1 / cuBLAS 13.8.0.4 and retained **sm_80, sm_86,
and sm_89** in both archives, plus the six compute_75 PTX fallbacks in cuBLASLt.
Classic cuBLAS has no sm_89 cubins; Ada relies on compatible earlier 8.x code.
It did not remove relocatable ELF or use whole-archive linkage. Shared cudart
remained unchanged.

The pruned library was restricted to the intended RTX 30-/40-series targets.
The application's PTX fallback did not make that library suitable for other
GPU generations. Any future pruning experiment must re-inventory architectures
after toolkit updates.

## Historical size measurements

Same-engine-source images, with CUDA device-code compression enabled:

| Packaging | Executable bytes | Runtime filesystem bytes |
|---|---:|---:|
| Shared cuBLAS baseline (`77699cb`) | 39,614,600 | 775,577,600 |
| Unpruned static cuBLAS | 577,632,848 | 711,024,640 |
| Pruned static cuBLAS | 442,053,712 | 575,447,040 |

Runtime filesystem measurements used `du -sx -B1 /` inside each container, not
Docker's local storage accounting or compressed registry download sizes.
Static linking alone saved ~64.6 MB; pruning saved another ~135.6 MB. Combined:
**~200.1 MB / 25.8% less runtime filesystem space** versus shared cuBLAS.

`ldd` confirmed that the pruned executable had no cuBLAS shared-library dependency.
Its final cubin inventory was 328 sm_80, 155 sm_86, and 83 sm_89 entries
(application plus linked library code). Archive-member elimination explains why
these counts were below the input archives.

## Historical GPU measurements

RTX 3090 Ti, driver 615.71.09; 4B GPTQ, TQ4 KV, CUDA graphs, 128 prompt tokens and
64 generated tokens. Three alternating runs per image/profile, with
`CUDA_DISABLE_PTX_JIT=1`. Plain, MTP-3, and native-BF16 DFlash2 generated token IDs
matched between shared and pruned images across all runs. The vision benchmark
passed cold/warm lossless parity and image-content cache isolation in all runs.

| Profile | Shared median decode tok/s | Pruned median decode tok/s |
|---|---:|---:|
| Plain | 210.44 | 210.53 |
| MTP-3 | 528.22 | 528.45 |
| DFlash2 | 1,280.81 | 1,289.72 |
| Vision, warm | 196.73 | 196.65 |

Plain/MTP/DFlash2 median TTFT remained ~20.3–20.7 ms. Vision median cold TTFT
was 34.87 → 34.31 ms; warm TTFT was 17.35 → 17.42 ms. These small differences
are short-run variability, not a speedup claim. Process wall times included
Docker startup and did not isolate cold CUDA module-loading overhead.

Initial static verification had 438 passing tests and the unrelated hot-cache
assertion failure. After integrating `d9ecc01`, the full static-link
`mise run check` with PTX JIT disabled passed: **441 Rust tests passed, 0 failed,
11 ignored**, plus formatting, both Clippy modes, rustdoc, dashboard, oracle,
and performance-tooling checks. The merged static Docker image rebuilt and
passed CLI startup. Those are historical static-build results, not a substitute
for checking subsequent shared-build changes.

Size and throughput tables retain their original pre-merge measurements; they
have not been relabeled for newer engine sources. No actual RTX 40-series or
minimum-driver qualification was performed during the static experiment.

## RTX 4090 (`sm_89`) qualification (completed)

Verified on **2× NVIDIA GeForce RTX 4090 (`24,564 MiB`, `sm_89`, driver `595.84`, CUDA `13.2` / `libcudart.so.13.2.86`, `libcublas.so.13.4.0.1`, `libcublasLt.so.13.4.0.1`)** with `CUDA_DISABLE_PTX_JIT=1` across both:
1. The exact `/usr/local/bin/w34` executable from `warp-34:shared-cublas-final` (`sha256:9260b1c51a9c2f7874993a6ff4a9f6560e83cfe4aaefc9d37609f909891f70f7`, `39,615,656` bytes, source `a3237c6`), and
2. The merged `main` (`95a7c86`) build with `W34_ARCH=sm_86,sm_89 W34_NVCC_EXTRA=--compress-mode=size` (`39,592,816` bytes, `25 × sm_86` + `25 × sm_89` native cubins via `cuobjdump --list-elf`).

- **`Qwen3.5-4B-W4A16-HGPTQ` (`128` prompt / `64` decode, `tq4`, CUDA Graphs, `CUDA_DISABLE_PTX_JIT=1`):**
  - Plain (`TP=1`): `220.49 tok/s` (`9260b1c51a9c`, `12.17 ms` TTFT) / `220.77 tok/s` (`main`, `12.09 ms` TTFT).
  - `MTP-3` (`TP=1`): `562.58 tok/s` / `560.21 tok/s` (`2.9375` accepted drafts/round), 100% lossless token match with Plain.
  - `DFlash2` (`TP=1`, native `BF16` + `cuBLASLt`): `1,398.04 tok/s` / `1,409.28 tok/s` (`11.60` accepted drafts/round), 100% lossless token match.
  - `DFlash2` (`TP=2`, 2× RTX 4090): `1,394.05 tok/s` (`29.60 ms` TTFT) / `1,476.03 tok/s` (`15.69 ms` TTFT), 100% lossless token match.
- **Vision (`--stage vision`, `TP=1`, `CUDA_DISABLE_PTX_JIT=1`):**
  - `lossless_parity: true`, `content_hash_isolated: true`, cold TTFT `31.21` / `31.31 ms` → warm TTFT `18.23` / `18.26 ms` (`1.71×` speedup), warm decode `212.85` / `212.82 tok/s`.
- **HTTP Serving (`w34 serve --offline`, `CUDA_DISABLE_PTX_JIT=1`):**
  - `GET /health` (`{"status":"ok"}`), `GET /` (embedded dashboard HTML/JS), `GET /v1/models` (cached snapshot via `HF_HUB_CACHE`), and `POST /v1/chat/completions` all passed.
- Minimum-driver qualification remains separate from the 4090 hardware check.

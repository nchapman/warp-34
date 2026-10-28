# Real DFlash RMSNorm / GEMM investigation (physical GPU1 only)

## Outcome

No native arithmetic fix made: the proposed BF16-rounded residual sum is disproven by actual donor operands. Native full-model parity **still fails** at relL2 0.01666940361054917, maxabs 0.21875. The checked-in test currently uses **0.005**, not the 0.008 mentioned in the request; it was left unchanged. No expected tensors or tolerances changed. Attention, shared headers/FFI, and unrelated work were not edited. No commits or full verification.

Capture uses genuine SGLang 0.5.19 DFlash2DraftModel, checkpoint 8d8bb62873a4fe6165e644163e9ff76e06ab78ba, original exported target inputs. Original goldens remain untouched. New captures' overlapping tensors are **bitwise identical** to the original full-model golden (see diagnostics JSON). Existing post-forward input hooks are insufficient for in-place RMSNorm; added pre-forward snapshots, weight/epsilon captures, invocation IDs, and output snapshots. New `*_norms` and `*_gemms` safetensors and JSON record provenance. The latter includes the four layer-0 projection weights (213 MiB); the former is 27 MiB.

## Isolated actual-operand replay

- 17 standalone/fused RMSNorm invocations: 16 normalized outputs bitwise, one output differs by one BF16 element (layer 2 post-attention norm, relL2 3.8606281e-5, maxabs 0.0078125). All fused residual outputs bitwise. All pass strict relL2 < 1e-4.
- Largest captured fused operand magnitude: 4,879,082,848,256.
- CPU FP32-sum normalization reproduces the same single discrepancy as native; rounding the sum to BF16 first worsens donor agreement, up to relL2 0.00285377 at final norm. Therefore changing residual-sum rounding would introduce a bug, not fix one.
- Layer-0 QKV, output, gate/up, and down GEMMs with donor operands/weights: **bitwise** across all four outputs.
- Existing 20 real grouped-convolution, five QK/RoPE, and five SiLU replays still pass strict bounds. This does not establish bitwise agreement everywhere.

## Native stagewise error and donor-attention counterfactual

Diagnostic injection replaces attention output with the original donor tensor at **every layer**, after recording native attention. It defaults off, has no environment-variable behavior, uses existing memcpy API, and is explicitly NOT native parity. Injection also generates separate traces in `.pi/dflash-model-inputs/trace-donor-attention`. Ordinary traces remain in `trace`. The following values are relL2:

| Stage | Native | Donor attention injected |
|---|---:|---:|
| Layer 0 input norm | 0 | 0 |
| Layer 0 pre-QKV convolution | 0 | 0 |
| Layer 0 QKV GEMM | 0 | 0 |
| Layer 0 Q after norm/RoPE | 9.53688e-7 | 9.53688e-7 |
| Layer 0 native attention, before substitution | 5.15126e-5 | 5.15126e-5 |
| Layer 0 output GEMM | 1.77312e-4 | 0 |
| Layer 0 residual | 1.12981e-4 | 0 |
| Layer 0 post-attention norm | 4.39696e-4 | 0 |
| Layer 0 pre-MLP convolution | 0.00124726 | 0 |
| Layer 0 gate/up GEMM | 0.00150306 | 0 |
| Layer 0 SiLU | 0.00211704 | 0 |
| Layer 0 down GEMM | 0.00287118 | 0 |
| Layer 0 hidden | 0.00417760 | 0 |
| Layer 1 hidden | 0.0127305 | 0 |
| Layer 2 hidden | 0.0114106 | 0.000423461 |
| Layer 3 hidden | 0.0181666 | 0.00163604 |
| Layer 4 hidden | 0.0153370 | 0.00283976 |
| Final normalized hidden | **0.0166694** | **0.00373236** |

Final maxabs: native 0.21875; counterfactual 0.0625. Counterfactual clears the existing strict bound but cannot count as a native pass.

This localizes two measured issues rather than declaring inherent error:
1. The first ordinary-path mismatch is QK/RoPE (three Q elements), followed by 24 differing layer-0 attention elements. Even attention error below 1e-4 is amplified substantially by the real checkpoint. Attention implementation was not changed.
2. With donor attention injected, layers 0 and 1 hidden/residual are bitwise. Layer 2 post-attention norm has **bitwise input and residual**, but one normalized element differs (same discrepancy as isolated replay). Its pre-MLP convolution differs at 98 elements, gate/up at 4870, and final layer hidden relL2 grows to 0.000423461. Thus the remaining counterfactual divergence has a concrete normalization reduction/rounding boundary to investigate, not the proposed rounding of the residual sum. Layer 2 pre-QKV convolution also has a one-element discrepancy (relL2 2.10608e-7), but its attention path is cut by this injection.

## Commands run

All native CUDA invocations used physical GPU1 isolated as logical device 0:

```sh
CUDA_VISIBLE_DEVICES=1 cargo test -p w34-host --features cuda --test phase4_dflash real_shape_ -- --nocapture
# 5 tests pass: norms, GEMMs, convolution, QK/RoPE, SiLU
CUDA_VISIBLE_DEVICES=1 cargo test -p w34-host --features cuda --test phase4_dflash export_dflash_oracle_inputs -- --ignored --exact --nocapture
# passes, refreshes native diagnostic traces
CUDA_VISIBLE_DEVICES=1 cargo test -p w34-host --features cuda --test phase4_dflash actual_sglang_full_model_hidden_parity -- --ignored --exact --nocapture
# FAILS, unchanged strict bound
CUDA_VISIBLE_DEVICES=1 cargo test -p w34-host --features cuda --test phase4_dflash diagnostic_donor_attention_full_model -- --ignored --exact --nocapture
# passes only as explicitly labeled counterfactual
```

Capture command (run twice with distinct new output names, `norms` then `gemms`; never overwrite an existing capture):

```sh
docker run --rm --gpus device=1 --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -v "$PWD":/work -v "$HOME/.cache/huggingface":/hf:ro \
  --entrypoint python lmsysorg/sglang:v0.5.19 \
  /work/tools/oracle/sglang_dflash_model_dump.py \
  --inputs /work/.pi/dflash-model-inputs \
  --model /hf/hub/models--nchapman--Qwen3.5-4B-DFlash2/snapshots/8d8bb62873a4fe6165e644163e9ff76e06ab78ba \
  --diagnostics --out /work/tools/oracle/goldens/sglang_dflash_model_4b_gemms.safetensors
```

CPU-only `diagnose_dflash_real_norms.py --fixture ..._gemms.safetensors --original ..._4b.safetensors --trace .pi/dflash-model-inputs/trace` produces `*_norm_diagnostics.json`; use `trace-donor-attention` for `*_donor_attention_diagnostics.json`. Replay and full-model logs are alongside this report.

## Remaining blockers

No first-fail/fix arithmetic patch is justified for residual-sum rounding: actual replays reject that hypothesis. Next work should isolate donor reduction/inverse-RMS/multiply ordering for the single layer-2 norm element, with a regression targeting that measured discrepancy before any change. Then quantify the effect of the three QK/RoPE discrepancies separately from attention's isolated discrepancy (donor Q/K/V injection), coordinating with the attention owner. Four layer-0 GEMMs are ruled out; later-layer GEMMs have not been exhaustively replayed. No claim of unavoidable full-model error or completed native parity is made.

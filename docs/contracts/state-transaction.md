# State Transaction Contract — GDN `StateBundle`, Explicit Indexing, & Chunk Geometry

## 1. Donor & Precision Architecture

Qwen 3.5 / 3.6 / 3.8 hybrid models interleave 3× `"linear_attention"` (`QwenGatedDeltaNetAttention`, GDN) layers with 1× `"full_attention"` layer (`48` GDN + `16` full-attn on 27B; `24` GDN + `8` full-attn on 9B/4B; `18` GDN + `6` full-attn on 0.8B).

- **CUDA C++ Kernel Donor:** `vllm.cpp` (`src/vt/cuda/cuda_gdn.cu` at `50d425bec`), cross-checked against `vLLM 0.31.0` (`qwen_gdn_linear_attn.py` / FLA) golden dumps.
- **Per-Layer State Dtypes (`precision-policy.md`):**
  - `conv_state`: `FP16` (`__half`), shape `[num_slots, local_conv_dim, conv_kernel_size - 1]` where `conv_kernel_size = 4` (`d_conv - 1 = 3` history taps) and `local_conv_dim = (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim) / tp_size`.
  - `ssm_state`: **`FP32` (`float`) in global memory**, shape `[num_slots, local_v_heads, head_v_dim, head_k_dim]` (`head_k_dim = 128`, `head_v_dim = 128`). Never downcast persistent `ssm_state` to `FP16` or `BF16`.
  - `A_log` and `dt_bias`: `FP32` (`float`).
  - Layer activation I/O (`qkvz`, `ba`, `conv_out`, `recurrent_out`, `RMSNormGated` output): `FP16` (`__half`).

## 2. VRAM Footprint of One `StateBundle` Slot

One `StateBundle` slot represents the complete GDN state across all `L_gdn` linear-attention layers for a single sequence or snapshot:
- **Qwen3.8-27B (`L_gdn = 48`, `H_k = 16`, `H_v = 48`, `d_k = 128`, `d_v = 128`):**
  - Unsharded (`TP=1`): `ssm_state = 48 * 48 * 128 * 128 * 4 B = 150,994,944 B (144.00 MiB)`; `conv_state = 48 * 10240 * 3 * 2 B = 2,949,120 B (2.81 MiB)` → **`146.81 MiB / slot`**.
  - Sharded (`TP=2`): `ssm_state = 72.00 MiB / rank`; `conv_state = 1.41 MiB / rank` → **`73.41 MiB / rank / slot`**.
- **Qwen3.5-9B / 4B (`L_gdn = 24`, `H_k = 16`, `H_v = 32`, `d_k = 128`, `d_v = 128`):**
  - Unsharded (`TP=1`): `ssm_state = 24 * 32 * 128 * 128 * 4 B = 48.00 MiB`; `conv_state = 24 * 8192 * 3 * 2 B = 1.125 MiB` → **`49.13 MiB / slot`** (`24.56 MiB / rank` on `TP=2`).
- **Qwen3.8-Flash-Next (`qwen4_exp`: `L_gdn = 36`, `H_k = 16`, `H_v = 48`, `d_k = 128`, `d_v = 128`, `ple_layer_ids = [2]`):**
  - Includes `36` GDN layers (`ssm_state` `FP32` + `conv_state` `FP16`) plus `ple_conv_state` (`[num_ple_layers, hc_count * hidden_size, 9]` `FP16`, `184,320 B/slot` unsharded) and `ple_history_tokens` (`[ngram_size - 1]` `u32` token IDs per slot) → **`110.3 MiB / slot` on `TP=1`** (`55.2 MiB / rank / slot` on `TP=2`).

Because each 27B `StateBundle` is `~73.4 MiB/rank`, all state slots (active request slots, speculative verify scratch slots, ping-pong checkpoint track buffers, and `MambaRadixCache` snapshot slots) are pre-budgeted upfront by `CapacityPlanner`.

## 3. Three Distinct Chunk Geometries (`B = 16`, `BT = 64`, `C = k * 64`)

1. **KV Page Size (`B = 16` tokens):** Granularity of `KvPool` page allocation and `block_tables`.
2. **GDN Prefill Compute Tile (`BT = 64` tokens = `4` KV pages):** Intra-chunk parallel scan tile size used by `chunk_gated_delta_rule`.
3. **`MambaRadixCache` Checkpoint Interval (`C = k * 64` tokens, default `C = 64`):**
   - Must be an integer multiple of both `B = 16` and `BT = 64`, and satisfy `C >= max_spec_verify_rows` (`64 >= 16`).
   - Guarantees at most **one** radix checkpoint boundary can be crossed within a single speculative verify step (`block_rows <= 16`).

## 4. Explicit Input / Output / Spec / Checkpoint Indexing (`w34_gdn_batch`)

GDN kernels NEVER assume `state_out == state_in`. Every prefill, decode, and speculative verify launch accepts an explicit `w34_gdn_batch` descriptor:

```c
typedef struct {
    const int32_t* query_start_loc;            // [R_max + 1] token offsets per request
    const int32_t* state_in_indices;           // [R_max] source StateBundle slot per request (< 0 = trash/inactive)
    const int32_t* state_out_indices;          // [R_max] destination StateBundle slot for final step of request
    const int32_t* spec_state_indices;         // [R_max * max_verify_tokens] per-step candidate slots (NULL if decode/prefill)
    const int32_t* checkpoint_dst_indices;     // [R_max * max_checkpoints_per_req] radix/ping-pong snapshot dst slots (NULL if none)
    const int32_t* checkpoint_token_offsets;   // [R_max * max_checkpoints_per_req] 0-based query token offset triggering snapshot
    const uint8_t* has_initial_state;          // [R_max] 1 = load from state_in_indices[r], 0 = zero-initialize state
    const uint8_t* valid_req_mask;             // [R_max] 1 = active request, 0 = padded graph slot
    int32_t num_reqs;                          // Active requests R <= R_max
    int32_t total_tokens;                      // Active tokens N <= N_max
    int32_t max_verify_tokens;                 // Stride of spec_state_indices (e.g. 8 or 16; 0 if unused)
    int32_t max_checkpoints_per_req;           // Stride of checkpoint_* arrays (0 if unused)
} w34_gdn_batch;
```

### Transactional Semantics Across Modes

1. **Cold Prefill (`has_initial_state[r] == 0`):**
   - Starts from zero `conv_state` and zero `ssm_state`.
   - At each chunk boundary `checkpoint_token_offsets[r, e]` (multiples of `C = 64`), writes a snapshot of `(conv_state, ssm_state)` into `checkpoint_dst_indices[r, e]`.
   - Writes final state after `q_len[r]` tokens into `state_out_indices[r]`.
2. **Warm Suffix Prefill / Chunk Continuation (`has_initial_state[r] == 1`):**
   - Reads initial `(conv_state, ssm_state)` from `state_in_indices[r]` (which may be a restored `MambaRadixCache` snapshot slot while `state_out_indices[r]` is the request's private working slot — copy-on-write in one pass).
3. **`M = 1` Decode:**
   - Reads `state_in_indices[r]`, advances 1 token, writes `state_out_indices[r]` (may alias `state_in_indices[r]`).
4. **Speculative Verify (`block_rows = 1 + k`, `spec_state_indices != NULL`):**
   - Reads initial state prior to the anchor token from `state_in_indices[r]`.
   - For each verify step `s in 0..block_rows` (`s = 0` is the anchor, `s = 1..=k` are draft proposals), writes the post-step `(conv_state, ssm_state)` into `spec_state_indices[r * max_verify_tokens + s]`.
   - When `RejectionSampler` accepts `a in 0..=k` draft tokens, step `s = a` (`spec_state_indices[r * max_verify_tokens + a]`) is the exact committed GDN state for `processed_len + 1 + a`! The host/pool either rebinds the request's active slot to `spec_state_indices[r, a]` (zero-copy slot swap) or copies from `spec_state_indices[r, a]`, leaving `state_in_indices[r]` and rejected steps untouched.

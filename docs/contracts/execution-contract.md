# Execution Contract — V1 Token-Budget Forward Pass, 4-Length Transaction, `TapPlan`, & Composite `GraphKey`

## 1. Overview

Every model step in `warp-34` — prefill chunk, prefix-hit suffix prefill, `M=1` decode, batched decode, `MTP` `1+k` verify, and `DFlash2` `M=8` (`27B`) / `M=16` (`4B`) verify — executes through a single unified token-budget forward spine (`Qwen3_5Model::forward`).

## 2. Flattened Token-Budget Batch Layout

All scheduled tokens across active requests in a step (`R <= R_max`, `total_tokens = N <= N_max`) are packed into a 2D contiguous activation tensor:

- `hidden_states`: `[N_max, hidden_size]` (`FP16` for target spine; `BF16` for `BF16` drafter spine). Only rows `0..N` are logically active (`valid_token_mask[row] == 1`).
- `query_start_loc`: `[R_max + 1]` (`INT32`), prefix sum of per-request query token counts (`query_start_loc[0] = 0`, `query_start_loc[r + 1] = query_start_loc[r] + q_len[r]`). Padded request slots `r >= R` satisfy `query_start_loc[r + 1] == query_start_loc[r]`.
- `positions`: `[3, N_max]` (`INT32`), fixed plane stride `N_max` for 3D `mRoPE` (`temporal`, `height`, `width` planes). For text-only tokens, all 3 planes carry identical 1D logical token positions.
- `valid_token_mask`: `[N_max]` (`UINT8` / `INT32`), `1` for active token rows `0..N` and `0` for padded graph rows `N..N_max`.
- `valid_req_mask`: `[R_max]` (`UINT8` / `INT32`), `1` for active requests `0..R` and `0` for padded request slots `R..R_max`.

### Valid-Row Isolation Invariant

Every CUDA Graph-capturable kernel MUST guard side effects with `valid_token_mask` / `valid_req_mask` (or route padded indices to a dedicated trash sink page/state slot):
1. Inactive token rows (`valid_token_mask[t] == 0`) never write to live KV pages (`write_slot_mapping[t] < 0` or trash page), never overwrite `TapPlan` capture rows, and never update RNG state.
2. Inactive request slots (`valid_req_mask[r] == 0`) never mutate live GDN `StateBundle` slots (`state_in_indices`, `spec_state_indices`, `checkpoint_dst_indices`).

## 3. Four-Length Request Transaction (`processed_len`, `emitted_len`, `reserved_len`, `draft_context_len`)

Every active request tracks four distinct lengths (in tokens):

| Field | Invariant | Meaning |
|---|---|---|
| `processed_len` | `<= emitted_len <= reserved_len` | Number of tokens whose target KV cache and target GDN `StateBundle` have been computed and committed. |
| `emitted_len` | `processed_len <= emitted_len <= processed_len + 1` (at rest between steps) | Total prompt + committed output tokens visible to the caller/sampler history. After prefill or verify, `emitted_len = processed_len + 1` because the newly sampled/recovered token is emitted immediately and serves as the next step's unconsumed anchor token. |
| `reserved_len` | `>= processed_len + q_len` | Private writable KV page and state capacity reserved in the pool prior to launching prefill or speculative verify. |
| `draft_context_len` | `<= processed_len` | Number of tokens materialized inside the attached proposer's (`MTP` or `DFlash2`) own draft-context KV cache. |

### Speculative Verify Commit Rule (`1 + k` verify rows, `a in 0..=k` accepted drafts)

Given `block_rows = 1 + k` target verify rows (`[anchor, draft_1, ..., draft_k]`, where `block_rows = 8` on 27B `DFlash2` and `16` on 4B `DFlash2`):
- Before target verify: `anchor` is at logical position `processed_len` (`emitted_len == processed_len + 1`).
- Target forward evaluates `block_rows = 1 + k` tokens at positions `processed_len .. processed_len + 1 + k`, writing candidate KV rows into reserved slots and step-indexed GDN candidate states into `spec_state_indices[req, 0..=k]`.
- If `a in 0..=k` draft proposals are accepted by `RejectionSampler`:
  - `processed_len` advances by `1 + a` (the anchor + `a` accepted drafts now have committed target KV and committed GDN state copied/selected from `spec_state_indices[req, a]`).
  - `emitted_len` advances by `1 + a` (`a` accepted drafts + `1` recovered/bonus token sampled from verify row `a`), maintaining `emitted_len == processed_len + 1`.
  - Speculative KV writes at positions `processed_len .. reserved_len` and unchosen `spec_state_indices` are discarded without rollbacks to committed state.
  - Only chunk boundaries (`C = k_chunk * 64`) that fall strictly within the new `processed_len` are eligible to publish GDN checkpoints to `MambaRadixCache`.

## 4. Static `TapPlan` for Proposer Hidden-State Capture

To support `DFlash2` and `MTP` without dynamic graph branching or extra forward passes:
- `TapPlan` is an immutable descriptor configured at model load time:
  - `tap_plan_id: u8` (`0` = disabled, `1` = `DFlash2` tap set e.g. `[5, 19, 33, 47, 61]` on 27B, `2` = `MTP` final-layer hidden tap).
  - `layer_mask: u64`: bitmask of transformer layer indices `l` whose post-MLP residual output (`hidden_states` after layer `l`'s post-norm/residual addition) must be captured.
  - `layer_to_slot: [i8; 64]`: maps layer index `l` to tap buffer plane `0..num_tap_layers`.
- Output buffer layout: `tap_buffer: [num_tap_layers, N_max, hidden_size]` (`FP16`), written only for `valid_token_mask[t] == 1`.

## 5. Composite CUDA `GraphKey`

CUDA Graphs are keyed by the 6-tuple `GraphKey`:

```rust
pub struct GraphKey {
    pub execution_mode: ExecutionMode,   // DecodeM1 | SpecVerify { block_rows: u8 }
    pub req_bucket: u16,                 // Padded request count R_max (e.g. 1, 2, 4, 8, 16)
    pub tok_bucket: u16,                 // Padded token count N_max (e.g. 1, 2, 4, 8, 16, 64, 128)
    pub query_len_class: QueryLenClass,  // SingleToken | UniformVerify(u8) | Heterogeneous
    pub kv_layout_mode: KvLayoutMode,    // Fp16 | Tq4FirstLast2 | K8v4FirstLast2
    pub tap_plan_id: u8,                 // 0 = None, 1 = DFlash2, 2 = Mtp
}
```

Keying by `total_tokens` alone is forbidden because `8 reqs x 1 tok` (`DecodeM1`) and `1 req x 8 tok` (`SpecVerify { block_rows: 8 }`) have distinct `query_start_loc` strides, GDN recurrent scan lengths, and attention causal masks.

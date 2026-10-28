# Cache Layout Contract — `KvLayerLayout`, `AttentionBatch`, Logical Page IDs, & Namespaces

## 1. Core Principle: Logical Page IDs Decoupled from Physical Layout

Page allocation and radix prefix indexing operate exclusively on **logical `page_id`s** (`i32`) with fixed page block size `B = 16` tokens:
- Logical page index within a sequence: `logical_block_idx = token_pos / 16`
- Intra-page token offset: `page_offset = token_pos % 16` (`0..16`)
- Physical token slot: `slot = physical_page_id * 16 + page_offset` (`NULL_PAGE_ID = 0` or negative slot `< 0` denotes a padded/masked trash sink).

No scheduler, radix tree, or block-table manager may multiply a `page_id` or `slot` by a global "bytes per token" constant. Physical byte offsets are resolved per layer through `KvLayerLayout`.

## 2. Cache Namespaces (`KvNamespace`)

A single `KvPool` manages isolated logical page tables across three cache namespaces:
1. `TargetFullAttn` (`0`): Target model full-attention layers (e.g., 16 full-attn layers on 27B, 8 on 9B, 8 on 4B; `head_dim = 256`).
2. `MtpFullAttn` (`1`): Native `Qwen3_5MTP` 1-layer full-attention KV cache (`head_dim = 256`).
3. `DFlashDraftContext` (`2`): `DFlash2` proposer cross-attention draft-context KV cache (`head_dim = 128` on 27B with symmetric non-causal sliding window `2048`; `BF16` full attention on 4B).

## 3. Per-Layer Physical Descriptor (`KvLayerLayout`)

Mirrored identically in Rust (`crates/w34-kernels/src/ffi.rs`) and CUDA C (`crates/w34-kernels/csrc/w34_cuda.h`):

```c
typedef enum {
    W34_KV_NAMESPACE_TARGET_FULL_ATTN    = 0,
    W34_KV_NAMESPACE_MTP_FULL_ATTN       = 1,
    W34_KV_NAMESPACE_DFLASH_DRAFT_CTX    = 2,
} w34_kv_namespace;

typedef enum {
    W34_KV_FORMAT_FP16  = 0, // 2 B/elem (512 B/row at d=256, 256 B/row at d=128)
    W34_KV_FORMAT_BF16  = 1, // 2 B/elem (512 B/row at d=256, 256 B/row at d=128)
    W34_KV_FORMAT_TQ4   = 2, // block_tq4_0: 66 B per 128 elems (132 B/row at d=256)
    W34_KV_FORMAT_Q8_0  = 3, // block_q8_0:  68 B per 64 elems  (272 B/row at d=256)
} w34_kv_format;

typedef enum {
    W34_KV_TRANSFORM_IDENTITY           = 0,
    W34_KV_TRANSFORM_RADEMACHER_WHT256  = 1, // Seed-42 Rademacher sign + WHT-256
} w34_kv_transform;

typedef struct {
    uint64_t k_base;                // Device pointer to layer K slab base
    uint64_t v_base;                // Device pointer to layer V slab base
    uint64_t k_page_stride_bytes;   // Bytes between consecutive physical pages in K
    uint64_t k_head_stride_bytes;   // Bytes between KV heads within a page in K
    uint64_t k_token_stride_bytes;  // Bytes between tokens (u in 0..16) within a head in K
    uint64_t v_page_stride_bytes;   // Bytes between consecutive physical pages in V
    uint64_t v_head_stride_bytes;   // Bytes between KV heads within a page in V
    uint64_t v_token_stride_bytes;  // Bytes between tokens (u in 0..16) within a head in V
    int32_t  namespace_id;          // w34_kv_namespace
    int32_t  layer_id;              // Full-attn slot index within namespace
    int32_t  local_kv_heads;        // KV heads on this TP rank
    int32_t  head_dim;              // 128 or 256
    int32_t  block_size;            // Always 16
    int32_t  k_format;              // w34_kv_format
    int32_t  v_format;              // w34_kv_format
    int32_t  k_transform;           // w34_kv_transform (independent of v_transform)
    int32_t  v_transform;           // w34_kv_transform (independent of k_transform)
    int32_t  reserved;
} w34_kv_layer_layout;
```

### Physical Byte Address Formula & Alignment Invariant

For physical page `p`, local KV head `h in 0..local_kv_heads`, and intra-page token `u in 0..16`:
- `k_row_ptr = (const uint8_t*)k_base + p * k_page_stride_bytes + h * k_head_stride_bytes + u * k_token_stride_bytes`
- `v_row_ptr = (const uint8_t*)v_base + p * v_page_stride_bytes + h * v_head_stride_bytes + u * v_token_stride_bytes`

**Alignment Note:** For `TQ4` (`d = 256`), `k_token_stride_bytes = 132 B` (`2 * sizeof(block_tq4_0)`), which is `4`-byte aligned (`132 % 16 == 4`), while `k_head_stride_bytes = 16 * 132 = 2112 B` (`2112 % 16 == 0`). Kernels addressing `TQ4` rows MUST compute byte pointers using explicit byte strides and perform `block_tq4_0`-aligned loads (`__half d` + `uint32_t` / `uchar4` packed nibbles), never assuming `float4` (`16 B`) alignment on individual `TQ4` rows.

## 4. `AttentionBatch` Descriptor

```c
typedef enum {
    W34_ATTN_MASK_CAUSAL                  = 0,
    W34_ATTN_MASK_SYMMETRIC_WINDOW        = 1, // Non-causal within query chunk, sliding window
} w34_attn_mask_mode;

typedef struct {
    const int32_t* query_start_loc;      // [R_max + 1] device ptr
    const int32_t* context_lens_before;  // [R_max] committed prefix length before this step
    const int32_t* seq_lens_after;       // [R_max] context_lens_before[r] + q_len[r]
    const int32_t* block_tables;         // [R_max, max_blocks_per_req] physical page IDs
    const int32_t* write_slot_mapping;   // [N_max] physical_page_id * 16 + u (< 0 for masked/trash)
    const uint8_t* valid_token_mask;     // [N_max] 1 = active, 0 = padded
    const uint8_t* valid_req_mask;       // [R_max] 1 = active, 0 = padded
    int32_t num_reqs;                    // Active requests R <= R_max
    int32_t total_tokens;                // Active tokens N <= N_max
    int32_t max_blocks_per_req;          // Row stride of block_tables
    int32_t max_q_len;                   // Host-known upper bound on max_r(q_len[r])
    int32_t max_seq_len;                 // Host-known upper bound on max_r(seq_lens_after[r])
    int32_t mask_mode;                   // w34_attn_mask_mode
    int32_t sliding_window;              // 0 = full, >0 = window size (e.g. 2048)
    float   sm_scale;                    // 1.0f / sqrtf(head_dim)
} w34_attention_batch;
```

## 5. Copy-on-Write (COW) & Radix Ownership Rules

- Pages stored in `MambaRadixCache` with `full_lock_ref > 0` or shared across multiple requests are immutable with respect to existing tokens.
- Full pages (`16/16` tokens committed) in a radix prefix are referenced directly in `block_tables` without copying.
- If a prefix hit ends at a **partial tail page** (`committed_len % 16 != 0`) and that page is shared (`ref_count > 1` or radix-cached), the pool allocates a fresh private physical page, copies the `committed_len % 16` valid rows across all layers in the namespace before writing new suffix tokens, and places the private `page_id` into `block_tables`.

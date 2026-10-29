//! Phase 5 Verification Gates:
//! - Phase 5.1: Paged `TQ4` (`132 B`/row) & `K8V4` (`272 B` K + `132 B` V) with Seed-42
//!   Rademacher + WHT-256 rotation, `FirstLast2` (`26,016 B/token`, `2.52x`) and
//!   `K8V4FirstLast2` (`33,296 B/token`, `1.97x`) boundary layers, 0-ULP page-table
//!   permutation invariance, and long-context needle-in-a-haystack retrieval.
//! - Phase 5.2: Multimodal Vision (`smart_resize`, bicubic + normalize, 3D `mRoPE`,
//!   content-hashed `MambaRadixCache` keys, OpenAI `image_url`, `op_vision.cu` kernels,
//!   and `Qwen3VisionEncoder` + multimodal generation).
//! - Phase 5.3: Hybrid MoE (`op_moe.cu` top-k router, fused `W4A16` & `FP16` MoE
//!   gate+up+SiLU + down+shared_expert+gate kernels, `Qwen3.6-35B-A3B` config & engine).
#![allow(clippy::undocumented_unsafe_blocks)]

use w34_host::cache::{KvFormatPolicy, kv_format_row_bytes};
use w34_kernels::ffi::{
    W34_KV_FORMAT_FP16, W34_KV_FORMAT_Q8_0, W34_KV_FORMAT_TQ4, W34_KV_TRANSFORM_IDENTITY,
    W34_KV_TRANSFORM_RADEMACHER_WHT256,
};

#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::Mutex;
#[cfg(feature = "cuda")]
use w34_host::cache::KvPool;
#[cfg(feature = "cuda")]
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, f16_to_f32, f32_to_f16};
#[cfg(feature = "cuda")]
use w34_kernels::ffi::{self, W34_ATTN_MASK_CAUSAL, W34AttentionBatch};

#[cfg(feature = "cuda")]
static GPU_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_phase5_1_firstlast2_and_k8v4_byte_budgets() {
    // Qwen 27B: 16 full-attention layers, 4 KV heads total, head_dim = 256.
    let num_full = 16usize;
    let kv_heads = 4usize;
    let head_dim = 256usize;

    assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_FP16, head_dim), 512);
    assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_TQ4, head_dim), 132);
    assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_Q8_0, head_dim), 272);

    let fp16_bpt = KvFormatPolicy::Fp16.bytes_per_token(num_full, kv_heads, head_dim);
    let tq4_bpt = KvFormatPolicy::Tq4.bytes_per_token(num_full, kv_heads, head_dim);
    let k8v4_bpt = KvFormatPolicy::K8V4.bytes_per_token(num_full, kv_heads, head_dim);
    let fl2_bpt = KvFormatPolicy::FirstLast2.bytes_per_token(num_full, kv_heads, head_dim);
    let k8v4_fl2_bpt = KvFormatPolicy::K8V4FirstLast2.bytes_per_token(num_full, kv_heads, head_dim);

    assert_eq!(fp16_bpt, 65_536);
    assert_eq!(tq4_bpt, 16_896); // 3.88x raw compression
    assert_eq!(k8v4_bpt, 25_856); // 2.53x raw compression
    assert_eq!(fl2_bpt, 26_016); // 2.52x effective compression (3 FP16 + 13 TQ4)
    assert_eq!(k8v4_fl2_bpt, 33_296); // 1.97x effective compression (3 FP16 + 13 K8V4)

    for ord in 0..num_full {
        let (kf, vf, kt, vt) = KvFormatPolicy::FirstLast2.resolve_layer(ord, num_full);
        if ord == 0 || ord >= num_full - 2 {
            assert_eq!(kf, W34_KV_FORMAT_FP16);
            assert_eq!(vf, W34_KV_FORMAT_FP16);
            assert_eq!(kt, W34_KV_TRANSFORM_IDENTITY);
            assert_eq!(vt, W34_KV_TRANSFORM_IDENTITY);
        } else {
            assert_eq!(kf, W34_KV_FORMAT_TQ4);
            assert_eq!(vf, W34_KV_FORMAT_TQ4);
            assert_eq!(kt, W34_KV_TRANSFORM_RADEMACHER_WHT256);
            assert_eq!(vt, W34_KV_TRANSFORM_RADEMACHER_WHT256);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_phase5_1_wht256_outlier_protection_and_long_context_needle() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");
    let ex = stream.exec(0, 0);

    // 1. Verify WHT-256 rotation protects channel-outlier KV vectors and preserves
    //    0-ULP page-table permutation invariance across Tq4, K8V4, FirstLast2, K8V4FirstLast2.
    for policy in [
        KvFormatPolicy::Tq4,
        KvFormatPolicy::K8V4,
        KvFormatPolicy::FirstLast2,
        KvFormatPolicy::K8V4FirstLast2,
    ] {
        let layer_ids = [3, 7, 11, 15, 19];
        let pool = KvPool::new(0, 0, &layer_ids, 2, 256, 16, 16, policy).expect("KvPool");
        // Test interior layer (ordinal 1 -> rotated TQ4 or K8V4)
        let layout = pool.layers[1].layout;
        assert_eq!(layout.k_transform, W34_KV_TRANSFORM_RADEMACHER_WHT256);
        assert_eq!(layout.v_transform, W34_KV_TRANSFORM_RADEMACHER_WHT256);

        let seq_len = 48usize; // 3 full pages
        let q_len = 8usize; // multi-token verify batch (t <= 16)
        let mut k_host = vec![0u16; seq_len * 2 * 256];
        let mut v_host = vec![0u16; seq_len * 2 * 256];
        for t in 0..seq_len {
            for h in 0..2 {
                for c in 0..256 {
                    // Inject structured signal + sharp channel-0 outlier (which WHT-256 spreads evenly)
                    let base_k = (((t * 13 + h * 7 + c * 3) % 61) as f32 - 30.0) * 0.02;
                    let outlier_k = if c == 0 {
                        3.5 * ((t as f32 * 0.3).sin())
                    } else {
                        0.0
                    };
                    k_host[(t * 2 + h) * 256 + c] = f32_to_f16(base_k + outlier_k);

                    let base_v = (((t * 17 + h * 11 + c * 5) % 59) as f32 - 29.0) * 0.02;
                    let outlier_v = if c == 128 {
                        4.0 * ((t as f32 * 0.5).cos())
                    } else {
                        0.0
                    };
                    v_host[(t * 2 + h) * 256 + c] = f32_to_f16(base_v + outlier_v);
                }
            }
        }
        let q_host: Vec<u16> = (0..(q_len * 8 * 256))
            .map(|i| {
                let c = i % 256;
                let base = (((i * 19) % 67) as f32 - 33.0) * 0.02;
                let outlier = if c == 0 { 2.0 } else { 0.0 };
                f32_to_f16(base + outlier)
            })
            .collect();

        let d_k = DeviceBuffer::alloc(0, k_host.len() * 2).unwrap();
        let d_v = DeviceBuffer::alloc(0, v_host.len() * 2).unwrap();
        let d_q = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        let d_out_a = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        let d_out_b = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        d_k.copy_from_host(&k_host, &stream).unwrap();
        d_v.copy_from_host(&v_host, &stream).unwrap();
        d_q.copy_from_host(&q_host, &stream).unwrap();

        let run_pages = |pages: [i32; 3], d_out: &DeviceBuffer| -> Vec<u16> {
            let slots: Vec<i32> = (0..seq_len)
                .map(|pos| pages[pos / 16] * 16 + (pos % 16) as i32)
                .collect();
            let d_slots = DeviceBuffer::alloc(0, slots.len() * 4).unwrap();
            let d_bt = DeviceBuffer::alloc(0, 3 * 4).unwrap();
            let d_qs = DeviceBuffer::alloc(0, 8).unwrap();
            let d_cb = DeviceBuffer::alloc(0, 4).unwrap();
            let d_sa = DeviceBuffer::alloc(0, 4).unwrap();
            d_slots.copy_from_host(&slots, &stream).unwrap();
            d_bt.copy_from_host(&pages, &stream).unwrap();
            d_qs.copy_from_host(&[0i32, q_len as i32], &stream).unwrap();
            d_cb.copy_from_host(&[(seq_len - q_len) as i32], &stream)
                .unwrap();
            d_sa.copy_from_host(&[seq_len as i32], &stream).unwrap();

            let mut batch = W34AttentionBatch {
                block_tables: d_bt.addr() as *const i32,
                write_slot_mapping: d_slots.addr() as *const i32,
                query_start_loc: d_qs.addr() as *const i32,
                context_lens_before: d_cb.addr() as *const i32,
                seq_lens_after: d_sa.addr() as *const i32,
                valid_req_mask: core::ptr::null(),
                valid_token_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: seq_len as i32,
                max_blocks_per_req: 3,
                max_q_len: q_len as i32,
                max_seq_len: seq_len as i32,
                sliding_window: 0,
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sm_scale: 1.0 / (256.0f32).sqrt(),
                reserved: 0,
            };
            assert_eq!(
                unsafe {
                    ffi::w34_kv_cache_write(
                        &ex,
                        &layout,
                        &batch,
                        d_k.addr() as *const c_void,
                        d_v.addr() as *const c_void,
                    )
                },
                0
            );
            batch.total_tokens = q_len as i32;
            assert_eq!(
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex,
                        &layout,
                        &batch,
                        d_q.addr() as *const c_void,
                        d_out.addr() as *mut c_void,
                        8,
                    )
                },
                0
            );
            let mut out = vec![0u16; q_host.len()];
            d_out.copy_to_host(&mut out, &stream).unwrap();
            out
        };

        let out_a = run_pages([0, 1, 2], &d_out_a);
        let out_b = run_pages([14, 5, 9], &d_out_b);
        assert_eq!(
            out_a, out_b,
            "0-ULP page-table permutation invariance failed for {policy:?}"
        );
    }

    // 2. Long-context (4096 tokens = 256 pages, fragmented page table) needle-in-a-haystack gate
    //    for both TQ4 and K8V4.
    let ctx_len = 4096usize;
    let num_pages = ctx_len / 16;
    let needle_pos = 2713usize; // deep inside fragmented page table

    for policy in [KvFormatPolicy::Tq4, KvFormatPolicy::K8V4] {
        let pool = KvPool::new(0, 0, &[0], 2, 256, 16, num_pages, policy).expect("4K needle pool");
        let layout = pool.layers[0].layout;

        let mut k_host = vec![0u16; ctx_len * 2 * 256];
        let mut v_host = vec![0u16; ctx_len * 2 * 256];
        let mut needle_key = [0.0f32; 256];
        let mut needle_val = [0.0f32; 256];
        for c in 0..256 {
            needle_key[c] = if (c * 7) % 2 == 0 { 1.25 } else { -1.25 };
            needle_val[c] = ((c as f32) - 128.0) * 0.03;
        }

        for t in 0..ctx_len {
            for h in 0..2 {
                for c in 0..256 {
                    if t == needle_pos {
                        k_host[(t * 2 + h) * 256 + c] = f32_to_f16(needle_key[c]);
                        v_host[(t * 2 + h) * 256 + c] = f32_to_f16(needle_val[c]);
                    } else {
                        // Orthogonal pseudo-random distractors
                        let phase = ((t * 131 + h * 17 + c * 31) % 251) as f32;
                        let dk = (phase * 0.1).sin() * 0.25;
                        let dv = (phase * 0.1).cos() * 0.5;
                        k_host[(t * 2 + h) * 256 + c] = f32_to_f16(dk);
                        v_host[(t * 2 + h) * 256 + c] = f32_to_f16(dv);
                    }
                }
            }
        }

        let mut q_host = vec![0u16; 8 * 256];
        for qh in 0..8 {
            for c in 0..256 {
                q_host[qh * 256 + c] = f32_to_f16(needle_key[c] * 2.0);
            }
        }

        // Permuted physical pages across all 256 pages
        let pages: Vec<i32> = (0..num_pages)
            .map(|i| ((i * 73 + 19) % num_pages) as i32)
            .collect();
        let slots: Vec<i32> = (0..ctx_len)
            .map(|pos| pages[pos / 16] * 16 + (pos % 16) as i32)
            .collect();

        let d_k = DeviceBuffer::alloc(0, k_host.len() * 2).unwrap();
        let d_v = DeviceBuffer::alloc(0, v_host.len() * 2).unwrap();
        let d_q = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        let d_out = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        let d_slots = DeviceBuffer::alloc(0, slots.len() * 4).unwrap();
        let d_bt = DeviceBuffer::alloc(0, pages.len() * 4).unwrap();
        let d_qs = DeviceBuffer::alloc(0, 8).unwrap();
        let d_cb = DeviceBuffer::alloc(0, 4).unwrap();
        let d_sa = DeviceBuffer::alloc(0, 4).unwrap();

        d_k.copy_from_host(&k_host, &stream).unwrap();
        d_v.copy_from_host(&v_host, &stream).unwrap();
        d_q.copy_from_host(&q_host, &stream).unwrap();
        d_slots.copy_from_host(&slots, &stream).unwrap();
        d_bt.copy_from_host(&pages, &stream).unwrap();
        d_qs.copy_from_host(&[0i32, 1i32], &stream).unwrap();
        d_cb.copy_from_host(&[(ctx_len - 1) as i32], &stream)
            .unwrap();
        d_sa.copy_from_host(&[ctx_len as i32], &stream).unwrap();

        let mut batch = W34AttentionBatch {
            block_tables: d_bt.addr() as *const i32,
            write_slot_mapping: d_slots.addr() as *const i32,
            query_start_loc: d_qs.addr() as *const i32,
            context_lens_before: d_cb.addr() as *const i32,
            seq_lens_after: d_sa.addr() as *const i32,
            valid_req_mask: core::ptr::null(),
            valid_token_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: ctx_len as i32,
            max_blocks_per_req: num_pages as i32,
            max_q_len: 1,
            max_seq_len: ctx_len as i32,
            sliding_window: 0,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sm_scale: 1.0 / (256.0f32).sqrt(),
            reserved: 0,
        };
        assert_eq!(
            unsafe {
                ffi::w34_kv_cache_write(
                    &ex,
                    &layout,
                    &batch,
                    d_k.addr() as *const c_void,
                    d_v.addr() as *const c_void,
                )
            },
            0
        );
        batch.total_tokens = 1;
        assert_eq!(
            unsafe {
                ffi::w34_paged_attention_f16(
                    &ex,
                    &layout,
                    &batch,
                    d_q.addr() as *const c_void,
                    d_out.addr() as *mut c_void,
                    8,
                )
            },
            0
        );
        let mut out_host = vec![0u16; q_host.len()];
        d_out.copy_to_host(&mut out_host, &stream).unwrap();

        let mut dot = 0.0f32;
        let mut norm_out = 0.0f32;
        let mut norm_ref = 0.0f32;
        for c in 0..256 {
            let o = f16_to_f32(out_host[c]);
            let r = needle_val[c];
            dot += o * r;
            norm_out += o * o;
            norm_ref += r * r;
        }
        let cos_sim = dot / (norm_out.sqrt() * norm_ref.sqrt());
        assert!(
            cos_sim > 0.994,
            "4K needle-in-a-haystack cosine similarity too low for {policy:?}: {cos_sim}"
        );
    }
}

#[test]
fn phase5_2_vision_smart_resize_mrope_and_multimodal_radix_key() {
    use w34_host::server::parse_openai_messages_multimodal;
    use w34_host::vision::{
        compute_mrope_positions, compute_multimodal_radix_key, preprocess_rgb_image, smart_resize,
    };

    // 1. smart_resize aligns to factor=32 (`patch_size=16 * merge_size=2`) and respects pixel bounds.
    let (rh, rw) = smart_resize(100, 150, 32, 32 * 32, 1024 * 1024);
    assert_eq!(rh % 32, 0);
    assert_eq!(rw % 32, 0);
    assert!(rh * rw >= 32 * 32 && rh * rw <= 1024 * 1024);

    // 2. preprocess_rgb_image produces `[1, H/16, W/16]` grid and deterministic content hash.
    let rgb_a = vec![128u8; 64 * 64 * 3];
    let mut rgb_b = vec![128u8; 64 * 64 * 3];
    rgb_b[0] = 255;
    let img_a1 = preprocess_rgb_image(&rgb_a, 64, 64, 16, 2, 2, 32 * 32, 1024 * 1024);
    let img_a2 = preprocess_rgb_image(&rgb_a, 64, 64, 16, 2, 2, 32 * 32, 1024 * 1024);
    let img_b = preprocess_rgb_image(&rgb_b, 64, 64, 16, 2, 2, 32 * 32, 1024 * 1024);
    assert_eq!(img_a1.grid_thw, [1, 4, 4]);
    assert_eq!(img_a1.num_merged_tokens(2), 4);
    assert_eq!(img_a1.content_hash, img_a2.content_hash);
    assert_ne!(img_a1.content_hash, img_b.content_hash);

    // 3. 3D mRoPE positions for `[BOS, vision_start, pad, pad, pad, pad, vision_end, tok7, tok8]`.
    let img_tok = 248056;
    let prompt = vec![
        1, 248053, img_tok, img_tok, img_tok, img_tok, 248054, 100, 101,
    ];
    let (planes, delta) = compute_mrope_positions(&prompt, &[img_a1.grid_thw], img_tok, 2);
    // BOS -> (0,0,0); vision_start -> (1,1,1); 4 merged patches (2x2) -> T=2, H in {2,3}, W in {2,3}
    assert_eq!(planes[0], vec![0, 1, 2, 2, 2, 2, 4, 5, 6]);
    assert_eq!(planes[1], vec![0, 1, 2, 2, 3, 3, 4, 5, 6]);
    assert_eq!(planes[2], vec![0, 1, 2, 3, 2, 3, 4, 5, 6]);
    // 9 tokens in prompt, max_pos = 6 => next_pos = 7 => delta = 7 - 9 = -2.
    assert_eq!(delta, -2);

    // 4. Multimodal radix key distinguishes different images with identical `<|image_pad|>` runs.
    let key_a1 = compute_multimodal_radix_key(&prompt, &[img_a1], img_tok, 2);
    let key_a2 = compute_multimodal_radix_key(&prompt, &[img_a2], img_tok, 2);
    let key_b = compute_multimodal_radix_key(&prompt, &[img_b], img_tok, 2);
    assert_eq!(key_a1, key_a2);
    assert_ne!(key_a1, key_b);
    assert_eq!(key_a1[0], 1);
    assert_eq!(key_a1[6], 248054);

    // 5. OpenAI `image_url` message parser expands multimodal content parts into `<|vision_start|><|image_pad|>...<|vision_end|>`.
    let msgs_json = serde_json::json!([
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/cat.ppm"}},
                {"type": "text", "text": "What is in this image?"}
            ]
        }
    ]);
    let (parsed_msgs, parsed_imgs) = parse_openai_messages_multimodal(&msgs_json).unwrap();
    assert_eq!(parsed_msgs.len(), 1);
    assert_eq!(parsed_imgs.len(), 1);
    let content_str = parsed_msgs[0].content.as_deref().unwrap();
    assert!(content_str.starts_with("<|vision_start|><|image_pad|>"));
    assert!(content_str.ends_with("<|vision_end|>What is in this image?"));

    // 6. Security: reject local `file://` URLs and empty base64 payloads.
    assert!(w34_host::vision::parse_image_url_to_rgb("file:///etc/passwd").is_err());
    assert!(w34_host::vision::parse_image_url_to_rgb("data:image/png;base64,").is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn phase5_2_vision_cuda_kernels_and_embedding_splice() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);

    // Test LayerNorm + 2D RoPE full attention + GELU + embedding splice
    let total_tokens = 4usize;
    let num_heads = 2usize;
    let head_dim = 64usize;
    let hidden = num_heads * head_dim;

    let mut qkv_host = vec![0u16; total_tokens * 3 * hidden];
    for (i, v) in qkv_host.iter_mut().enumerate() {
        *v = f32_to_f16(((i % 19) as f32 - 9.0) * 0.05);
    }
    let bias_host = vec![f32_to_f16(0.01); 3 * hidden];
    let pos_hw: Vec<i32> = vec![0, 0, 0, 1, 1, 0, 1, 1];
    let cu_seqlens: Vec<i32> = vec![0, 4];

    let d_qkv = DeviceBuffer::alloc(0, qkv_host.len() * 2).unwrap();
    let d_bias = DeviceBuffer::alloc(0, bias_host.len() * 2).unwrap();
    let d_pos = DeviceBuffer::alloc(0, pos_hw.len() * 4).unwrap();
    let d_cu = DeviceBuffer::alloc(0, cu_seqlens.len() * 4).unwrap();
    let d_out = DeviceBuffer::alloc(0, total_tokens * hidden * 2).unwrap();

    d_qkv.copy_from_host(&qkv_host, &stream).unwrap();
    d_bias.copy_from_host(&bias_host, &stream).unwrap();
    d_pos.copy_from_host(&pos_hw, &stream).unwrap();
    d_cu.copy_from_host(&cu_seqlens, &stream).unwrap();

    assert_eq!(
        unsafe {
            ffi::w34_vision_qkv_rope_attn_f16(
                &ex,
                d_qkv.addr() as *mut c_void,
                d_bias.addr() as *const c_void,
                d_pos.addr() as *const i32,
                d_cu.addr() as *const i32,
                d_out.addr() as *mut c_void,
                total_tokens as i32,
                1,
                num_heads as i32,
                head_dim as i32,
                10000.0,
            )
        },
        0
    );
    let mut attn_out = vec![0u16; total_tokens * hidden];
    d_out.copy_to_host(&mut attn_out, &stream).unwrap();
    assert!(attn_out.iter().all(|&x| f16_to_f32(x).is_finite()));

    // Test embedding splice: overwrite token 1 and token 3 in a 4-token sequence
    let mut tok_emb = vec![f32_to_f16(1.0); 4 * 16];
    let vis_emb = vec![f32_to_f16(42.0); 2 * 16];
    let splice_idx = vec![1i32, 3i32];
    let d_tok = DeviceBuffer::alloc(0, tok_emb.len() * 2).unwrap();
    let d_vis = DeviceBuffer::alloc(0, vis_emb.len() * 2).unwrap();
    let d_idx = DeviceBuffer::alloc(0, splice_idx.len() * 4).unwrap();
    d_tok.copy_from_host(&tok_emb, &stream).unwrap();
    d_vis.copy_from_host(&vis_emb, &stream).unwrap();
    d_idx.copy_from_host(&splice_idx, &stream).unwrap();
    assert_eq!(
        unsafe {
            ffi::w34_embedding_splice_f16(
                &ex,
                d_tok.addr() as *mut c_void,
                d_vis.addr() as *const c_void,
                d_idx.addr() as *const i32,
                2,
                16,
            )
        },
        0
    );
    d_tok.copy_to_host(&mut tok_emb, &stream).unwrap();
    assert_eq!(f16_to_f32(tok_emb[0]), 1.0);
    assert_eq!(f16_to_f32(tok_emb[16]), 42.0);
    assert_eq!(f16_to_f32(tok_emb[32]), 1.0);
    assert_eq!(f16_to_f32(tok_emb[48]), 42.0);
}

#[cfg(feature = "cuda")]
#[test]
fn phase5_3_moe_router_and_fused_w4a16_f16_kernels() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);

    // 1. Top-k router softmax with renormalization (num_experts = 8, top_k = 2)
    let logits_f32 = [0.1f32, 2.5, -1.0, 0.5, 3.2, -0.4, 0.0, 1.1];
    let logits_f16: Vec<u16> = logits_f32.iter().copied().map(f32_to_f16).collect();
    let d_logits = DeviceBuffer::alloc(0, logits_f16.len() * 2).unwrap();
    let d_weights = DeviceBuffer::alloc(0, 2 * 4).unwrap();
    let d_ids = DeviceBuffer::alloc(0, 2 * 4).unwrap();
    d_logits.copy_from_host(&logits_f16, &stream).unwrap();
    assert_eq!(
        unsafe {
            ffi::w34_moe_topk_router_f16(
                &ex,
                d_logits.addr() as *const c_void,
                d_weights.addr() as *mut f32,
                d_ids.addr() as *mut i32,
                1,
                8,
                2,
                1,
                0, // softmax
            )
        },
        0
    );
    let mut top_w = [0.0f32; 2];
    let mut top_i = [0i32; 2];
    d_weights.copy_to_host(&mut top_w, &stream).unwrap();
    d_ids.copy_to_host(&mut top_i, &stream).unwrap();
    // Highest logits are index 4 (3.2) and index 1 (2.5)
    assert_eq!(top_i, [4, 1]);
    assert!((top_w[0] + top_w[1] - 1.0).abs() < 1e-5);
    assert!(top_w[0] > top_w[1]);

    // 2. Fused W4A16 MoE gate_up_silu + down_accum + shared_expert + shared_expert_gate
    // Compare W4A16 (with neutral nibble 8 + one expert having known weights) against analytical reference
    let num_tokens = 1usize;
    let top_k = 2usize;
    let num_experts = 4usize;
    let hidden = 128usize;
    let interm = 64usize;
    let group_size = 128usize;

    // Pack all nibbles = 9 -> (9 - 8) = +1.0 * scale
    let packed_plus1 = 0x9999_9999u32;
    let gu_words = num_experts * (2 * interm) * (hidden / 8);
    let gu_scales_len = num_experts * (2 * interm) * hidden.div_ceil(group_size);
    let dn_words = num_experts * hidden * (interm / 8);
    let dn_scales_len = num_experts * hidden * interm.div_ceil(group_size);

    let gu_packed = vec![packed_plus1; gu_words];
    let gu_scales = vec![f32_to_f16(0.01); gu_scales_len];
    let dn_packed = vec![packed_plus1; dn_words];
    let dn_scales = vec![f32_to_f16(0.02); dn_scales_len];

    let x_host = vec![f32_to_f16(1.0); num_tokens * hidden];
    let sh_out_host = vec![f32_to_f16(0.5); num_tokens * hidden];
    let sh_gate_host = vec![f32_to_f16(0.0); num_tokens]; // sigmoid(0.0) = 0.5 -> +0.25

    let d_x = DeviceBuffer::alloc(0, x_host.len() * 2).unwrap();
    let d_gu_p = DeviceBuffer::alloc(0, gu_packed.len() * 4).unwrap();
    let d_gu_s = DeviceBuffer::alloc(0, gu_scales.len() * 2).unwrap();
    let d_dn_p = DeviceBuffer::alloc(0, dn_packed.len() * 4).unwrap();
    let d_dn_s = DeviceBuffer::alloc(0, dn_scales.len() * 2).unwrap();
    let d_interm = DeviceBuffer::alloc(0, num_tokens * top_k * interm * 2).unwrap();
    let d_sh_out = DeviceBuffer::alloc(0, sh_out_host.len() * 2).unwrap();
    let d_sh_gate = DeviceBuffer::alloc(0, sh_gate_host.len() * 2).unwrap();
    let d_out = DeviceBuffer::alloc(0, num_tokens * hidden * 2).unwrap();

    d_x.copy_from_host(&x_host, &stream).unwrap();
    d_gu_p.copy_from_host(&gu_packed, &stream).unwrap();
    d_gu_s.copy_from_host(&gu_scales, &stream).unwrap();
    d_dn_p.copy_from_host(&dn_packed, &stream).unwrap();
    d_dn_s.copy_from_host(&dn_scales, &stream).unwrap();
    d_sh_out.copy_from_host(&sh_out_host, &stream).unwrap();
    d_sh_gate.copy_from_host(&sh_gate_host, &stream).unwrap();
    d_ids.copy_from_host(&[1i32, 2i32], &stream).unwrap();
    d_weights
        .copy_from_host(&[0.6f32, 0.4f32], &stream)
        .unwrap();

    assert_eq!(
        unsafe {
            ffi::w34_moe_w4a16_gate_up_silu_f16(
                &ex,
                d_x.addr() as *const c_void,
                d_gu_p.addr() as *const u32,
                d_gu_s.addr() as *const c_void,
                d_ids.addr() as *const i32,
                d_interm.addr() as *mut c_void,
                num_tokens as i32,
                top_k as i32,
                num_experts as i32,
                interm as i32,
                hidden as i32,
                group_size as i32,
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            ffi::w34_moe_w4a16_down_accum_f16(
                &ex,
                d_interm.addr() as *const c_void,
                d_dn_p.addr() as *const u32,
                d_dn_s.addr() as *const c_void,
                d_ids.addr() as *const i32,
                d_weights.addr() as *const f32,
                d_sh_out.addr() as *const c_void,
                d_sh_gate.addr() as *const c_void,
                d_out.addr() as *mut c_void,
                num_tokens as i32,
                top_k as i32,
                num_experts as i32,
                hidden as i32,
                interm as i32,
                group_size as i32,
            )
        },
        0
    );

    let mut out_host = vec![0u16; hidden];
    d_out.copy_to_host(&mut out_host, &stream).unwrap();
    // Analytical reference:
    // gate = 128 * 0.01 = 1.28, up = 1.28
    // act = silu(1.28) * 1.28 = (1.28 / (1 + exp(-1.28))) * 1.28 = 1.00143 * 1.28 = 1.28183
    // down per expert = 64 * 0.02 * 1.28183 = 1.64074
    // weighted sum (0.6 + 0.4 = 1.0) + sigmoid(0.0)*0.5 = 1.64074 + 0.25 = 1.89074
    let got = f16_to_f32(out_host[0]);
    assert!(
        (got - 1.8907).abs() < 0.02,
        "MoE W4A16 fused output mismatch: got {got}, expected ~1.8907"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn phase5_2_live_qwen3_vision_encoder_and_multimodal_radix_cache() {
    use w34_host::loader::resolve_hf_model_dir;
    use w34_host::model::Qwen3_5Engine;
    use w34_host::vision::preprocess_rgb_image;

    let Ok(dir) = resolve_hf_model_dir("nchapman/Qwen3.5-4B-W4A16-GPTQ") else {
        return;
    };
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut eng =
        Qwen3_5Engine::load_from_dir(&dir, 1, KvFormatPolicy::FirstLast2, Some(128), false)
            .expect("load Qwen3.5-4B-W4A16-GPTQ for vision test");

    // Create two distinct 64x64 RGB images (each produces a 1x4x4 patch grid -> 4 merged visual tokens).
    let mut rgb_a = vec![0u8; 64 * 64 * 3];
    for (i, b) in rgb_a.iter_mut().enumerate() {
        *b = ((i * 7 + 13) & 0xFF) as u8;
    }
    let mut rgb_b = vec![200u8; 64 * 64 * 3];
    for (i, b) in rgb_b.iter_mut().enumerate() {
        *b = (255usize.wrapping_sub(i * 3) & 0xFF) as u8;
    }
    let img_a = preprocess_rgb_image(&rgb_a, 64, 64, 16, 2, 2, 32 * 32, 1024 * 1024);
    let img_b = preprocess_rgb_image(&rgb_b, 64, 64, 16, 2, 2, 32 * 32, 1024 * 1024);
    assert_eq!(img_a.num_merged_tokens(2), 4);

    let img_tok = eng
        .cfg
        .vision_config
        .as_ref()
        .map(|v| v.image_token_id)
        .unwrap_or(248056);
    // Construct a 68-token multimodal prompt (> 64 tokens so 1 full radix checkpoint is donated at C=64)
    // containing 4 `<|image_pad|>` tokens at positions 4..8.
    let mut prompt_ids = vec![
        151644, 872, 198, 248053, img_tok, img_tok, img_tok, img_tok, 248054,
    ];
    while prompt_ids.len() < 68 {
        prompt_ids.push(198 + ((prompt_ids.len() % 17) as i32));
    }

    // 1. Cold multimodal generation with img_a
    let m_cold = eng
        .generate_multimodal_with_radix_cache(
            &prompt_ids,
            std::slice::from_ref(&img_a),
            6,
            64,
            false,
        )
        .expect("cold multimodal generate");
    assert_eq!(m_cold.cached_prefix_tokens, 0);
    assert_eq!(m_cold.donated_seq_len, 64);
    assert_eq!(m_cold.generated_token_ids.len(), 6);

    // 2. Warm multimodal generation with identical img_a -> hits 64-token radix checkpoint & matches greedy IDs
    let m_warm = eng
        .generate_multimodal_with_radix_cache(&prompt_ids, &[img_a], 6, 64, false)
        .expect("warm multimodal generate");
    assert_eq!(m_warm.cached_prefix_tokens, 64);
    assert_eq!(m_warm.generated_token_ids, m_cold.generated_token_ids);

    // 3. Multimodal generation with distinct img_b at the same `<|image_pad|>` positions ->
    // content hash diverges at token 4 (before C=64 boundary), so cached_prefix_tokens == 0!
    let m_other = eng
        .generate_multimodal_with_radix_cache(&prompt_ids, &[img_b], 6, 64, false)
        .expect("distinct image multimodal generate");
    assert_eq!(m_other.cached_prefix_tokens, 0);
}

#[cfg(feature = "cuda")]
#[test]
fn phase5_3_moe_checkpoint_config_and_dense_f16_tp_parity() {
    use w34_host::loader::{ModelConfig, resolve_hf_model_dir};

    // 1. Verify Qwen3.6-35B-A3B MoE checkpoint config parsing if cached on disk
    if let Ok(dir) = resolve_hf_model_dir(
        "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4",
    ) {
        let cfg = ModelConfig::from_dir(&dir).expect("parse 35B-A3B MoE config");
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 512);
        assert_eq!(cfg.shared_expert_intermediate_size, 512);
        assert!(cfg.norm_topk_prob);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        assert!(cfg.vision_config.is_some());
    }

    // 2. Verify Dense FP16 MoE kernels (`w34_moe_f16_gate_up_silu` + `w34_moe_f16_down_accum`)
    // match the W4A16 MoE kernels for identical effective weights.
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);

    let num_tokens = 1usize;
    let top_k = 2usize;
    let num_experts = 4usize;
    let hidden = 128usize;
    let interm = 64usize;

    let gu_w = vec![f32_to_f16(0.01); num_experts * (2 * interm) * hidden];
    let dn_w = vec![f32_to_f16(0.02); num_experts * hidden * interm];
    let x_host = vec![f32_to_f16(1.0); num_tokens * hidden];
    let sh_out_host = vec![f32_to_f16(0.5); num_tokens * hidden];
    let sh_gate_host = vec![f32_to_f16(0.0); num_tokens];

    let d_x = DeviceBuffer::alloc(0, x_host.len() * 2).unwrap();
    let d_gu = DeviceBuffer::alloc(0, gu_w.len() * 2).unwrap();
    let d_dn = DeviceBuffer::alloc(0, dn_w.len() * 2).unwrap();
    let d_ids = DeviceBuffer::alloc(0, top_k * 4).unwrap();
    let d_weights = DeviceBuffer::alloc(0, top_k * 4).unwrap();
    let d_interm = DeviceBuffer::alloc(0, num_tokens * top_k * interm * 2).unwrap();
    let d_sh_out = DeviceBuffer::alloc(0, sh_out_host.len() * 2).unwrap();
    let d_sh_gate = DeviceBuffer::alloc(0, sh_gate_host.len() * 2).unwrap();
    let d_out = DeviceBuffer::alloc(0, num_tokens * hidden * 2).unwrap();

    d_x.copy_from_host(&x_host, &stream).unwrap();
    d_gu.copy_from_host(&gu_w, &stream).unwrap();
    d_dn.copy_from_host(&dn_w, &stream).unwrap();
    d_ids.copy_from_host(&[1i32, 2i32], &stream).unwrap();
    d_weights
        .copy_from_host(&[0.6f32, 0.4f32], &stream)
        .unwrap();
    d_sh_out.copy_from_host(&sh_out_host, &stream).unwrap();
    d_sh_gate.copy_from_host(&sh_gate_host, &stream).unwrap();

    assert_eq!(
        unsafe {
            ffi::w34_moe_f16_gate_up_silu(
                &ex,
                d_x.addr() as *const c_void,
                d_gu.addr() as *const c_void,
                d_ids.addr() as *const i32,
                d_interm.addr() as *mut c_void,
                num_tokens as i32,
                top_k as i32,
                num_experts as i32,
                interm as i32,
                hidden as i32,
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            ffi::w34_moe_f16_down_accum(
                &ex,
                d_interm.addr() as *const c_void,
                d_dn.addr() as *const c_void,
                d_ids.addr() as *const i32,
                d_weights.addr() as *const f32,
                d_sh_out.addr() as *const c_void,
                d_sh_gate.addr() as *const c_void,
                d_out.addr() as *mut c_void,
                num_tokens as i32,
                top_k as i32,
                num_experts as i32,
                hidden as i32,
                interm as i32,
            )
        },
        0
    );
    let mut out_host = vec![0u16; hidden];
    d_out.copy_to_host(&mut out_host, &stream).unwrap();
    let got = f16_to_f32(out_host[0]);
    assert!(
        (got - 1.8907).abs() < 0.02,
        "MoE FP16 fused output mismatch: got {got}, expected ~1.8907"
    );
}

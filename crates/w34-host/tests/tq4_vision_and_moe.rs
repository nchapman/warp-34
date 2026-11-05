//! TQ4 KV Cache, Multimodal Vision, and Hybrid MoE Integration Tests:
//! - Paged `TQ4` (`132 B`/row) & `Q8_0` (`272 B`/row) KV cache with Seed-42
//!   Rademacher + WHT-256 rotation, boundary-anchored `Tq4` (`21,376 B/token`, `3.07x`:
//!   rotated `Q8_0` on first 2 & last 2 full-attention layers + least-squares `TQ4` interior),
//!   uniform `Tq4All` (`16,896 B/token`, `3.88x`), 0-ULP page-table permutation invariance,
//!   and long-context needle-in-a-haystack retrieval.
//! - Multimodal Vision (`smart_resize`, bicubic + normalize, 3D `mRoPE`,
//!   content-hashed `MambaRadixCache` keys, OpenAI `image_url`, `op_vision.cu` kernels,
//!   and `Qwen3VisionEncoder` + multimodal generation).
//! - Hybrid MoE (`op_moe.cu` top-k router, fused `W4A16` & `FP16` MoE
//!   gate+up+SiLU + down+shared_expert+gate kernels, `Qwen3.6-35B-A3B` config & engine).

mod common;

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
fn test_tq4_and_q8_0_byte_budgets() {
    // Qwen 27B: 16 full-attention layers, 4 KV heads total, head_dim = 256.
    let num_full = 16usize;
    let kv_heads = 4usize;
    let head_dim = 256usize;

    assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_FP16, head_dim), 512);
    assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_TQ4, head_dim), 132);
    assert_eq!(kv_format_row_bytes(W34_KV_FORMAT_Q8_0, head_dim), 272);

    let fp16_bpt = KvFormatPolicy::Fp16.bytes_per_token(num_full, kv_heads, head_dim);
    let q8_bpt = KvFormatPolicy::Q8_0.bytes_per_token(num_full, kv_heads, head_dim);
    let tq4_bpt = KvFormatPolicy::Tq4.bytes_per_token(num_full, kv_heads, head_dim);
    let tq4_all_bpt = KvFormatPolicy::Tq4All.bytes_per_token(num_full, kv_heads, head_dim);

    assert_eq!(fp16_bpt, 65_536);
    assert_eq!(q8_bpt, 34_816); // 1.88x raw compression (plain 8-bit K & V)
    assert_eq!(tq4_bpt, 21_376); // 3.07x effective compression (4 rotated Q8_0 + 12 rotated TQ4)
    assert_eq!(tq4_all_bpt, 16_896); // 3.88x raw compression (16 rotated TQ4)

    assert_eq!(
        KvFormatPolicy::Q8_0.resolve_layer(0, num_full),
        (
            W34_KV_FORMAT_Q8_0,
            W34_KV_FORMAT_Q8_0,
            W34_KV_TRANSFORM_IDENTITY,
            W34_KV_TRANSFORM_IDENTITY
        )
    );

    for ord in 0..num_full {
        let (kf, vf, kt, vt) = KvFormatPolicy::Tq4.resolve_layer(ord, num_full);
        if ord < 2 || ord >= num_full - 2 {
            assert_eq!(kf, W34_KV_FORMAT_Q8_0);
            assert_eq!(vf, W34_KV_FORMAT_Q8_0);
            assert_eq!(kt, W34_KV_TRANSFORM_RADEMACHER_WHT256);
            assert_eq!(vt, W34_KV_TRANSFORM_RADEMACHER_WHT256);
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
fn test_wht256_outlier_protection_and_long_context_needle() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");
    let ex = stream.exec(0, 0);

    // 1. Verify WHT-256 rotation protects channel-outlier KV vectors on Tq4/Tq4All and
    //    preserves 0-ULP page-table permutation invariance across Q8_0, Tq4, and Tq4All.
    for policy in [
        KvFormatPolicy::Q8_0,
        KvFormatPolicy::Tq4,
        KvFormatPolicy::Tq4All,
    ] {
        let layer_ids = [3, 7, 11, 15, 19];
        let pool = KvPool::new(0, 0, &layer_ids, 2, 256, 16, 16, policy).expect("KvPool");
        // Test interior layer (ordinal 2 -> Q8_0 or rotated TQ4)
        let layout = pool.layers[2].layout;
        let expected_transform = if policy == KvFormatPolicy::Q8_0 {
            W34_KV_TRANSFORM_IDENTITY
        } else {
            W34_KV_TRANSFORM_RADEMACHER_WHT256
        };
        assert_eq!(
            (layout.k_transform, layout.v_transform),
            (expected_transform, expected_transform)
        );

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
                // SAFETY: `layout`, `batch`, `d_k`, and `d_v` are live on device 0 for `seq_len` tokens.
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
                // SAFETY: `layout`, `batch`, `d_q`, `d_gate`, and `d_out` are live on device 0.
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
    //    for both TQ4 and Q8_0.
    let ctx_len = 4096usize;
    let num_pages = ctx_len / 16;
    let needle_pos = 2713usize; // deep inside fragmented page table

    for policy in [
        KvFormatPolicy::Tq4All,
        KvFormatPolicy::Tq4,
        KvFormatPolicy::Q8_0,
    ] {
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
            // SAFETY: `layout`, `batch`, `d_k`, and `d_v` are live on device 0 for needle KV write.
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
            // SAFETY: `layout`, `batch`, `d_q`, `d_gate`, and `d_out` are live on device 0 for needle query.
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

fn encode_base64_for_test(data: &[u8]) -> String {
    const LUT: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let v = (b0 << 16) | (b1 << 8) | b2;
        out.push(LUT[((v >> 18) & 63) as usize] as char);
        out.push(LUT[((v >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            LUT[((v >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            LUT[(v & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[test]
fn vision_smart_resize_mrope_and_multimodal_radix_key() {
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
                {"type": "image_url", "image_url": {"url": "rgb://https://example.test/cat.ppm"}},
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

    check_image_decoding_and_url_policy();
}

fn check_image_decoding_and_url_policy() {
    // Reject local `file://` URLs, bare paths over HTTP, empty base64, and corrupt PNG/JPEG.
    for bad_url in [
        "file:///etc/passwd",
        "/etc/passwd",
        "data:image/png;base64,",
        "data:image/png;base64,bm90LWEtcG5n",
        "data:image/jpeg;base64,bm90LWEtanBlZw==",
        "data:image/webp;base64,UklGRg==",
    ] {
        assert!(w34_host::vision::parse_image_url_to_rgb(bad_url).is_err());
    }

    // 7. PNG (RGBA8 -> RGB8) and JPEG decoding via both `data:image/...;base64,...` and local CLI files.
    let encode_b64 = encode_base64_for_test;

    let (w, h) = (36u32, 24u32);
    let mut rgba_img = image::RgbaImage::new(w, h);
    let mut expected_rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 7 + y * 3) & 0xFF) as u8;
            let g = ((x * 5 + y * 11 + 40) & 0xFF) as u8;
            let b = ((x * 13 + y * 2 + 90) & 0xFF) as u8;
            rgba_img.put_pixel(x, y, image::Rgba([r, g, b, 128]));
            expected_rgb.extend_from_slice(&[r, g, b]);
        }
    }
    let mut png_bytes = Vec::new();
    image::DynamicImage::ImageRgba8(rgba_img)
        .write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    let png_uri = format!("data:image/png;base64,{}", encode_b64(&png_bytes));
    let (dec_png_rgb, dec_png_h, dec_png_w) =
        w34_host::vision::parse_image_url_to_rgb(&png_uri).unwrap();
    assert_eq!((dec_png_h, dec_png_w), (h as usize, w as usize));
    assert_eq!(dec_png_rgb, expected_rgb);

    let solid_rgb = image::RgbImage::from_pixel(32, 48, image::Rgb([200, 100, 50]));
    let mut jpg_bytes = Vec::new();
    image::DynamicImage::ImageRgb8(solid_rgb)
        .write_to(
            &mut std::io::Cursor::new(&mut jpg_bytes),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    let jpg_uri = format!("data:image/jpeg;base64,{}", encode_b64(&jpg_bytes));
    let (dec_jpg_rgb, dec_jpg_h, dec_jpg_w) =
        w34_host::vision::parse_image_url_to_rgb(&jpg_uri).unwrap();
    assert_eq!((dec_jpg_h, dec_jpg_w), (48, 32));
    assert!((dec_jpg_rgb[0] as i32 - 200).abs() <= 4);
    assert!((dec_jpg_rgb[1] as i32 - 100).abs() <= 4);
    assert!((dec_jpg_rgb[2] as i32 - 50).abs() <= 4);

    let pid = std::process::id();
    let tmp_png = std::env::temp_dir().join(format!("w34_vision_test_{pid}.png"));
    let tmp_bad = std::env::temp_dir().join(format!("w34_vision_test_{pid}.txt"));
    std::fs::write(&tmp_png, &png_bytes).unwrap();
    std::fs::write(&tmp_bad, b"not an image").unwrap();
    let cli_png = w34_host::vision::load_cli_image_to_rgb(tmp_png.to_str().unwrap());
    let http_png = w34_host::vision::parse_image_url_to_rgb(tmp_png.to_str().unwrap());
    let cli_bad = w34_host::vision::load_cli_image_to_rgb(tmp_bad.to_str().unwrap());
    let _ = std::fs::remove_file(&tmp_png);
    let _ = std::fs::remove_file(&tmp_bad);
    let (cli_rgb, cli_h, cli_w) = cli_png.unwrap();
    assert_eq!((cli_h, cli_w), (h as usize, w as usize));
    assert_eq!(cli_rgb, expected_rgb);
    assert!(http_png.is_err());
    assert!(cli_bad.is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn vision_cuda_kernels_and_embedding_splice() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let attention_scratch = DeviceBuffer::alloc(0, 4 * 2 * 4).unwrap();
    let ex = stream.exec(attention_scratch.addr(), 4 * 2 * 4);

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
        // SAFETY: `d_qkv`, `d_bias`, `d_pos`, `d_cu`, and `d_out` are live buffers matching vision RoPE+attn geometry.
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
        // SAFETY: `d_tok`, `d_vis`, and `d_idx` are live buffers on device 0 with valid splice indices.
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
fn moe_router_and_fused_w4a16_f16_kernels() {
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
        // SAFETY: `d_logits`, `d_weights`, and `d_ids` are live on device 0 for `1 * 4` logits and `top_k = 2`.
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
        // SAFETY: All `W4A16` MoE gate+up+SiLU buffers are live on device 0 and match the synthetic geometry.
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
        // SAFETY: All `W4A16` MoE down+shared accumulation buffers are live on device 0.
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
fn live_qwen3_vision_encoder_and_multimodal_radix_cache() {
    use w34_host::model::Qwen3_5Engine;
    use w34_host::vision::preprocess_rgb_image;

    let test_name = "live_qwen3_vision_encoder_and_multimodal_radix_cache";
    if !common::require_or_skip_gpus(1, test_name) {
        return;
    }
    let Some(dir) =
        common::require_or_skip_cached_model("nchapman/Qwen3.5-4B-W4A16-GPTQ", test_name)
    else {
        return;
    };
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut eng = Qwen3_5Engine::load_from_dir(&dir, 1, KvFormatPolicy::Tq4, Some(128), false)
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
fn moe_checkpoint_config_and_dense_f16_tp_parity() {
    use w34_host::loader::ModelConfig;

    // 1. Verify Qwen3.6-35B-A3B MoE checkpoint config parsing if cached on disk
    if let Some(dir) = common::require_or_skip_cached_model(
        "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4",
        "moe_checkpoint_config_and_dense_f16_tp_parity",
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
        // SAFETY: All `FP16` MoE gate+up+SiLU buffers are live on device 0.
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
        // SAFETY: All `FP16` MoE down+shared accumulation buffers are live on device 0.
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

#[cfg(feature = "cuda")]
#[test]
fn gqa_mma_and_split_kv_long_context_parity() {
    check_gqa_split_kv_parity(4096);
}

#[cfg(feature = "cuda")]
#[test]
fn short_split_kv_parity() {
    for seq_len in [128usize, 192, 256, 512, 1024] {
        check_gqa_split_kv_parity(seq_len);
    }
    check_batched_decode_short_split_tp1_tp2_0ulp();
}

#[cfg(feature = "cuda")]
fn check_batched_decode_short_split_tp1_tp2_0ulp() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let d_ws = DeviceBuffer::alloc(0, 8 * 1024 * 1024).unwrap();
    let ex_split = stream.exec(d_ws.addr(), d_ws.len_bytes);

    let num_reqs = 8usize;
    let seq_len = 256usize;
    let pages_per_req = seq_len / 16;
    let total_pages = num_reqs * pages_per_req;
    let kv_heads_tp1 = 4usize;
    let q_heads_tp1 = 16usize;
    let kv_heads_tp2 = 2usize;
    let q_heads_tp2 = 8usize;
    let head_dim = 256usize;
    let layer_ids = [0usize, 1, 2];

    let total_kv_tokens = num_reqs * seq_len;
    let mut k_tp1 = vec![0u16; total_kv_tokens * kv_heads_tp1 * head_dim];
    let mut v_tp1 = vec![0u16; total_kv_tokens * kv_heads_tp1 * head_dim];
    for t in 0..total_kv_tokens {
        for h in 0..kv_heads_tp1 {
            for c in 0..head_dim {
                let kv = (((t * 17 + h * 11 + c * 5) % 97) as f32 - 48.0) * 0.02;
                let vv = (((t * 23 + h * 13 + c * 7) % 89) as f32 - 44.0) * 0.02;
                let idx = (t * kv_heads_tp1 + h) * head_dim + c;
                k_tp1[idx] = f32_to_f16(kv);
                v_tp1[idx] = f32_to_f16(vv);
            }
        }
    }
    let split_kv_shard = |src: &[u16], rank: usize| -> Vec<u16> {
        let mut out = vec![0u16; total_kv_tokens * kv_heads_tp2 * head_dim];
        for t in 0..total_kv_tokens {
            for h in 0..kv_heads_tp2 {
                let src_off = (t * kv_heads_tp1 + rank * kv_heads_tp2 + h) * head_dim;
                let dst_off = (t * kv_heads_tp2 + h) * head_dim;
                out[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
            }
        }
        out
    };
    let mut q_tp1 = vec![0u16; num_reqs * q_heads_tp1 * head_dim];
    for r in 0..num_reqs {
        for h in 0..q_heads_tp1 {
            for c in 0..head_dim {
                let qv = (((r * 29 + h * 7 + c * 3) % 83) as f32 - 41.0) * 0.025;
                q_tp1[(r * q_heads_tp1 + h) * head_dim + c] = f32_to_f16(qv);
            }
        }
    }
    let split_q_shard = |rank: usize| -> Vec<u16> {
        let mut out = vec![0u16; num_reqs * q_heads_tp2 * head_dim];
        for r in 0..num_reqs {
            for h in 0..q_heads_tp2 {
                let src_off = (r * q_heads_tp1 + rank * q_heads_tp2 + h) * head_dim;
                let dst_off = (r * q_heads_tp2 + h) * head_dim;
                out[dst_off..dst_off + head_dim]
                    .copy_from_slice(&q_tp1[src_off..src_off + head_dim]);
            }
        }
        out
    };

    let pool_tp1 = KvPool::new(
        0,
        0,
        &layer_ids,
        kv_heads_tp1,
        head_dim,
        16,
        total_pages,
        KvFormatPolicy::Tq4,
    )
    .unwrap();
    let pool_r0 = KvPool::new(
        0,
        0,
        &layer_ids,
        kv_heads_tp2,
        head_dim,
        16,
        total_pages,
        KvFormatPolicy::Tq4,
    )
    .unwrap();
    let pool_r1 = KvPool::new(
        0,
        0,
        &layer_ids,
        kv_heads_tp2,
        head_dim,
        16,
        total_pages,
        KvFormatPolicy::Tq4,
    )
    .unwrap();

    let slots_w: Vec<i32> = (0..total_kv_tokens as i32).collect();
    let d_qs_w = DeviceBuffer::alloc(0, 8).unwrap();
    let d_ctx_w = DeviceBuffer::alloc(0, 4).unwrap();
    let d_seq_w = DeviceBuffer::alloc(0, 4).unwrap();
    let d_bt = DeviceBuffer::alloc(0, total_pages * 4).unwrap();
    let d_slots_w = DeviceBuffer::alloc(0, total_kv_tokens * 4).unwrap();
    d_qs_w
        .copy_from_host(&[0i32, total_kv_tokens as i32], &stream)
        .unwrap();
    d_ctx_w.copy_from_host(&[0i32], &stream).unwrap();
    d_seq_w
        .copy_from_host(&[total_kv_tokens as i32], &stream)
        .unwrap();
    let bt_host: Vec<i32> = (0..total_pages as i32).collect();
    d_bt.copy_from_host(&bt_host, &stream).unwrap();
    d_slots_w.copy_from_host(&slots_w, &stream).unwrap();

    let wb = W34AttentionBatch {
        query_start_loc: d_qs_w.addr() as *const i32,
        context_lens_before: d_ctx_w.addr() as *const i32,
        seq_lens_after: d_seq_w.addr() as *const i32,
        block_tables: d_bt.addr() as *const i32,
        write_slot_mapping: d_slots_w.addr() as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: total_kv_tokens as i32,
        max_blocks_per_req: total_pages as i32,
        max_q_len: total_kv_tokens as i32,
        max_seq_len: total_kv_tokens as i32,
        mask_mode: W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: 1.0 / (head_dim as f32).sqrt(),
        reserved: 0,
    };
    let populate = |layout: &ffi::W34KvLayerLayout, kh: &[u16], vh: &[u16]| {
        let dk = DeviceBuffer::alloc(0, kh.len() * 2).unwrap();
        let dv = DeviceBuffer::alloc(0, vh.len() * 2).unwrap();
        dk.copy_from_host(kh, &stream).unwrap();
        dv.copy_from_host(vh, &stream).unwrap();
        assert_eq!(
            // SAFETY: `dk`, `dv`, `wb`, and `layout` live on device 0 with matching dimensions.
            unsafe {
                ffi::w34_kv_cache_write(
                    &ex_split,
                    layout,
                    &wb,
                    dk.addr() as *const c_void,
                    dv.addr() as *const c_void,
                )
            },
            0
        );
    };
    populate(&pool_tp1.layers[1].layout, &k_tp1, &v_tp1);
    populate(
        &pool_r0.layers[1].layout,
        &split_kv_shard(&k_tp1, 0),
        &split_kv_shard(&v_tp1, 0),
    );
    populate(
        &pool_r1.layers[1].layout,
        &split_kv_shard(&k_tp1, 1),
        &split_kv_shard(&v_tp1, 1),
    );

    let qs_host: Vec<i32> = (0..=num_reqs as i32).collect();
    let ctx_host = vec![(seq_len - 1) as i32; num_reqs];
    let seq_host = vec![seq_len as i32; num_reqs];
    let d_qs = DeviceBuffer::alloc(0, qs_host.len() * 4).unwrap();
    let d_ctx = DeviceBuffer::alloc(0, ctx_host.len() * 4).unwrap();
    let d_seq = DeviceBuffer::alloc(0, seq_host.len() * 4).unwrap();
    d_qs.copy_from_host(&qs_host, &stream).unwrap();
    d_ctx.copy_from_host(&ctx_host, &stream).unwrap();
    d_seq.copy_from_host(&seq_host, &stream).unwrap();

    let d_q_tp1 = DeviceBuffer::alloc(0, q_tp1.len() * 2).unwrap();
    let d_q_r0 = DeviceBuffer::alloc(0, q_tp1.len()).unwrap();
    let d_q_r1 = DeviceBuffer::alloc(0, q_tp1.len()).unwrap();
    d_q_tp1.copy_from_host(&q_tp1, &stream).unwrap();
    d_q_r0.copy_from_host(&split_q_shard(0), &stream).unwrap();
    d_q_r1.copy_from_host(&split_q_shard(1), &stream).unwrap();

    let d_out_tp1 = DeviceBuffer::alloc(0, q_tp1.len() * 2).unwrap();
    let d_out_r0 = DeviceBuffer::alloc(0, q_tp1.len()).unwrap();
    let d_out_r1 = DeviceBuffer::alloc(0, q_tp1.len()).unwrap();

    for max_seq_bound in [seq_len as i32, 1024i32] {
        let ab = W34AttentionBatch {
            query_start_loc: d_qs.addr() as *const i32,
            context_lens_before: d_ctx.addr() as *const i32,
            seq_lens_after: d_seq.addr() as *const i32,
            block_tables: d_bt.addr() as *const i32,
            write_slot_mapping: core::ptr::null(),
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: num_reqs as i32,
            total_tokens: num_reqs as i32,
            max_blocks_per_req: pages_per_req as i32,
            max_q_len: 1,
            max_seq_len: max_seq_bound,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            reserved: 0,
        };
        // SAFETY: all device buffers and layouts match `ab` on device 0.
        unsafe {
            assert_eq!(
                ffi::w34_paged_attention_f16(
                    &ex_split,
                    &pool_tp1.layers[1].layout,
                    &ab,
                    d_q_tp1.addr() as *const c_void,
                    d_out_tp1.addr() as *mut c_void,
                    q_heads_tp1 as i32,
                ),
                0
            );
            assert_eq!(
                ffi::w34_paged_attention_f16(
                    &ex_split,
                    &pool_r0.layers[1].layout,
                    &ab,
                    d_q_r0.addr() as *const c_void,
                    d_out_r0.addr() as *mut c_void,
                    q_heads_tp2 as i32,
                ),
                0
            );
            assert_eq!(
                ffi::w34_paged_attention_f16(
                    &ex_split,
                    &pool_r1.layers[1].layout,
                    &ab,
                    d_q_r1.addr() as *const c_void,
                    d_out_r1.addr() as *mut c_void,
                    q_heads_tp2 as i32,
                ),
                0
            );
        }
        let mut h_tp1 = vec![0u16; q_tp1.len()];
        let mut h_r0 = vec![0u16; q_tp1.len() / 2];
        let mut h_r1 = vec![0u16; q_tp1.len() / 2];
        d_out_tp1.copy_to_host(&mut h_tp1, &stream).unwrap();
        d_out_r0.copy_to_host(&mut h_r0, &stream).unwrap();
        d_out_r1.copy_to_host(&mut h_r1, &stream).unwrap();
        for r in 0..num_reqs {
            let tp1_r0 =
                &h_tp1[(r * q_heads_tp1) * head_dim..(r * q_heads_tp1 + q_heads_tp2) * head_dim];
            let tp1_r1 = &h_tp1
                [(r * q_heads_tp1 + q_heads_tp2) * head_dim..(r + 1) * q_heads_tp1 * head_dim];
            let got_r0 = &h_r0[(r * q_heads_tp2) * head_dim..(r + 1) * q_heads_tp2 * head_dim];
            let got_r1 = &h_r1[(r * q_heads_tp2) * head_dim..(r + 1) * q_heads_tp2 * head_dim];
            assert_eq!(
                tp1_r0, got_r0,
                "0-ULP TP=1 == TP=2 rank 0 violated at req {r} max_seq_bound={max_seq_bound}"
            );
            assert_eq!(
                tp1_r1, got_r1,
                "0-ULP TP=1 == TP=2 rank 1 violated at req {r} max_seq_bound={max_seq_bound}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn check_gqa_split_kv_parity(seq_len: usize) {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let d_ws = DeviceBuffer::alloc(0, 16 * 1024 * 1024).unwrap();
    let ex_no_split = stream.exec(0, 0);
    let ex_split = stream.exec(d_ws.addr(), d_ws.len_bytes);

    let num_pages = seq_len / 16;
    let local_kv_heads = 2usize;
    let num_q_heads = 8usize; // gqa_ratio = 4
    let head_dim = 256usize;
    let layer_ids = [0usize, 1, 2, 3, 4];

    let mut k_host = vec![0u16; seq_len * local_kv_heads * head_dim];
    let mut v_host = vec![0u16; seq_len * local_kv_heads * head_dim];
    for t in 0..seq_len {
        for h in 0..local_kv_heads {
            for c in 0..head_dim {
                let kv = (((t * 17 + h * 11 + c * 5) % 97) as f32 - 48.0) * 0.02;
                let vv = (((t * 23 + h * 13 + c * 7) % 89) as f32 - 44.0) * 0.02;
                let idx = (t * local_kv_heads + h) * head_dim + c;
                k_host[idx] = f32_to_f16(kv);
                v_host[idx] = f32_to_f16(vv);
            }
        }
    }
    let d_k = DeviceBuffer::alloc(0, k_host.len() * 2).unwrap();
    let d_v = DeviceBuffer::alloc(0, v_host.len() * 2).unwrap();
    d_k.copy_from_host(&k_host, &stream).unwrap();
    d_v.copy_from_host(&v_host, &stream).unwrap();

    for (policy, layer_ord) in [
        (KvFormatPolicy::Fp16, 0usize),
        (KvFormatPolicy::Q8_0, 0usize),
        (KvFormatPolicy::Tq4, 0usize), // rotated Q8_0 boundary
        (KvFormatPolicy::Tq4, 2usize), // rotated TQ4 interior
    ] {
        let pool_contig = KvPool::new(
            0,
            0,
            &layer_ids,
            local_kv_heads,
            head_dim,
            16,
            num_pages,
            policy,
        )
        .unwrap();
        let pool_perm = KvPool::new(
            0,
            0,
            &layer_ids,
            local_kv_heads,
            head_dim,
            16,
            num_pages,
            policy,
        )
        .unwrap();
        let layout_c = pool_contig.layers[layer_ord].layout;
        let layout_p = pool_perm.layers[layer_ord].layout;

        let bt_contig: Vec<i32> = (0..num_pages as i32).collect();
        let bt_perm: Vec<i32> = (0..num_pages)
            .map(|p| ((p * 73 + 19) % num_pages) as i32)
            .collect();
        let slots_c: Vec<i32> = (0..seq_len as i32).collect();
        let slots_p: Vec<i32> = (0..seq_len)
            .map(|pos| bt_perm[pos / 16] * 16 + (pos % 16) as i32)
            .collect();

        let d_q_start_w = DeviceBuffer::alloc(0, 8).unwrap();
        let d_ctx_w = DeviceBuffer::alloc(0, 4).unwrap();
        let d_seq_w = DeviceBuffer::alloc(0, 4).unwrap();
        let d_bt_c = DeviceBuffer::alloc(0, num_pages * 4).unwrap();
        let d_bt_p = DeviceBuffer::alloc(0, num_pages * 4).unwrap();
        let d_slots_c = DeviceBuffer::alloc(0, seq_len * 4).unwrap();
        let d_slots_p = DeviceBuffer::alloc(0, seq_len * 4).unwrap();
        d_q_start_w
            .copy_from_host(&[0i32, seq_len as i32], &stream)
            .unwrap();
        d_ctx_w.copy_from_host(&[0i32], &stream).unwrap();
        d_seq_w.copy_from_host(&[seq_len as i32], &stream).unwrap();
        d_bt_c.copy_from_host(&bt_contig, &stream).unwrap();
        d_bt_p.copy_from_host(&bt_perm, &stream).unwrap();
        d_slots_c.copy_from_host(&slots_c, &stream).unwrap();
        d_slots_p.copy_from_host(&slots_p, &stream).unwrap();

        let mut wb = W34AttentionBatch {
            query_start_loc: d_q_start_w.addr() as *const i32,
            context_lens_before: d_ctx_w.addr() as *const i32,
            seq_lens_after: d_seq_w.addr() as *const i32,
            block_tables: d_bt_c.addr() as *const i32,
            write_slot_mapping: d_slots_c.addr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: seq_len as i32,
            max_blocks_per_req: num_pages as i32,
            max_q_len: seq_len as i32,
            max_seq_len: seq_len as i32,
            mask_mode: W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: 0.0625,
            reserved: 0,
        };
        assert_eq!(
            // SAFETY: `layout_c`, `wb`, `d_k`, and `d_v` are live on device 0 for contiguous KV write.
            unsafe {
                ffi::w34_kv_cache_write(
                    &ex_no_split,
                    &layout_c,
                    &wb,
                    d_k.addr() as *const c_void,
                    d_v.addr() as *const c_void,
                )
            },
            0
        );
        wb.block_tables = d_bt_p.addr() as *const i32;
        wb.write_slot_mapping = d_slots_p.addr() as *const i32;
        assert_eq!(
            // SAFETY: `layout_p`, `wb`, `d_k`, and `d_v` are live on device 0 for permuted KV write.
            unsafe {
                ffi::w34_kv_cache_write(
                    &ex_no_split,
                    &layout_p,
                    &wb,
                    d_k.addr() as *const c_void,
                    d_v.addr() as *const c_void,
                )
            },
            0
        );

        for q_len in [1usize, 16usize] {
            let q_elems = q_len * num_q_heads * head_dim;
            let q_host: Vec<u16> = (0..q_elems)
                .map(|i| f32_to_f16((((i * 29 + 7) % 101) as f32 - 50.0) * 0.02))
                .collect();
            let d_q = DeviceBuffer::alloc(0, q_elems * 2).unwrap();
            let d_out_single = DeviceBuffer::alloc(0, q_elems * 2).unwrap();
            let d_out_split_c = DeviceBuffer::alloc(0, q_elems * 2).unwrap();
            let d_out_split_p = DeviceBuffer::alloc(0, q_elems * 2).unwrap();
            d_q.copy_from_host(&q_host, &stream).unwrap();

            let d_qs = DeviceBuffer::alloc(0, 8).unwrap();
            let d_ctx = DeviceBuffer::alloc(0, 4).unwrap();
            d_qs.copy_from_host(&[0i32, q_len as i32], &stream).unwrap();
            d_ctx
                .copy_from_host(&[(seq_len - q_len) as i32], &stream)
                .unwrap();

            let mut ab = W34AttentionBatch {
                query_start_loc: d_qs.addr() as *const i32,
                context_lens_before: d_ctx.addr() as *const i32,
                seq_lens_after: d_seq_w.addr() as *const i32,
                block_tables: d_bt_c.addr() as *const i32,
                write_slot_mapping: core::ptr::null(),
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: q_len as i32,
                max_blocks_per_req: num_pages as i32,
                max_q_len: q_len as i32,
                max_seq_len: seq_len as i32,
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: 0.0625,
                reserved: 0,
            };

            assert_eq!(
                // SAFETY: `layout_c`, `ab`, `d_q`, and `d_out_single` are live on device 0.
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex_no_split,
                        &layout_c,
                        &ab,
                        d_q.addr() as *const c_void,
                        d_out_single.addr() as *mut c_void,
                        num_q_heads as i32,
                    )
                },
                0
            );
            assert_eq!(
                // SAFETY: `layout_c`, `ab`, `d_q`, and `d_out_split_c` are live on device 0 with split-KV workspace.
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex_split,
                        &layout_c,
                        &ab,
                        d_q.addr() as *const c_void,
                        d_out_split_c.addr() as *mut c_void,
                        num_q_heads as i32,
                    )
                },
                0
            );
            ab.block_tables = d_bt_p.addr() as *const i32;
            assert_eq!(
                // SAFETY: `layout_p`, `ab`, `d_q`, and `d_out_split_p` are live on device 0 with permuted block tables.
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex_split,
                        &layout_p,
                        &ab,
                        d_q.addr() as *const c_void,
                        d_out_split_p.addr() as *mut c_void,
                        num_q_heads as i32,
                    )
                },
                0
            );

            let mut h_single = vec![0u16; q_elems];
            let mut h_split_c = vec![0u16; q_elems];
            let mut h_split_p = vec![0u16; q_elems];
            d_out_single.copy_to_host(&mut h_single, &stream).unwrap();
            d_out_split_c.copy_to_host(&mut h_split_c, &stream).unwrap();
            d_out_split_p.copy_to_host(&mut h_split_p, &stream).unwrap();

            if seq_len <= 256 {
                // Captured short graphs declare 1024 keys even when live lengths
                // are smaller. The bound must not change the active partitions.
                let mut graph_batch = ab;
                graph_batch.max_seq_len = 1024;
                assert_eq!(
                    // SAFETY: same live buffers and layout as the preceding launch;
                    // device lengths still bound all KV accesses.
                    unsafe {
                        ffi::w34_paged_attention_f16(
                            &ex_split,
                            &layout_p,
                            &graph_batch,
                            d_q.addr() as *const c_void,
                            d_out_split_p.addr() as *mut c_void,
                            num_q_heads as i32,
                        )
                    },
                    0
                );
                let mut graph_out = vec![0u16; q_elems];
                d_out_split_p.copy_to_host(&mut graph_out, &stream).unwrap();
                assert_eq!(h_split_p, graph_out, "graph-bound partition parity");
            }

            // M=16 uses one token per CTA; padding with one invalid token
            // selects the packed prefill dispatch. Valid rows must remain bitwise
            // identical: token packing must not alter QK/PV reduction order.
            if q_len == 16 {
                let padded_elems = 17 * num_q_heads * head_dim;
                let mut padded_q = q_host.clone();
                padded_q.resize(padded_elems, 0);
                let d_padded_q = DeviceBuffer::alloc(0, padded_elems * 2).unwrap();
                let d_padded_out = DeviceBuffer::alloc(0, padded_elems * 2).unwrap();
                let d_mask = DeviceBuffer::alloc(0, 17).unwrap();
                let mut mask = vec![1u8; 17];
                mask[16] = 0;
                d_padded_q.copy_from_host(&padded_q, &stream).unwrap();
                d_mask.copy_from_host(&mask, &stream).unwrap();
                d_qs.copy_from_host(&[0i32, 17], &stream).unwrap();
                let mut packed = ab;
                packed.block_tables = d_bt_c.addr() as *const i32;
                packed.total_tokens = 17;
                packed.max_q_len = 17;
                packed.valid_token_mask = d_mask.addr() as *const u8;
                assert_eq!(
                    // SAFETY: `layout_c`, `packed`, `d_q17`, `d_g17`, and `d_out17` are live on device 0 for 17 tokens.
                    unsafe {
                        ffi::w34_paged_attention_f16(
                            &ex_no_split,
                            &layout_c,
                            &packed,
                            d_padded_q.addr() as *const c_void,
                            d_padded_out.addr() as *mut c_void,
                            num_q_heads as i32,
                        )
                    },
                    0
                );
                let mut packed_out = vec![0u16; padded_elems];
                d_padded_out.copy_to_host(&mut packed_out, &stream).unwrap();
                assert_eq!(
                    h_single,
                    packed_out[..q_elems],
                    "0-ULP token-packing parity violated for policy={policy:?} ord={layer_ord}"
                );
            }

            assert_eq!(
                h_split_c, h_split_p,
                "0-ULP Split-KV page-permutation invariance violated for policy={policy:?} ord={layer_ord} q_len={q_len}"
            );
            let max_diff = h_single
                .iter()
                .zip(h_split_c.iter())
                .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 2e-3,
                "Split-KV vs single-pass drift {max_diff} >= 2e-3 for policy={policy:?} ord={layer_ord} q_len={q_len}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn moe_grouped_mma_exact_parity_and_tp2() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();

    // 48 tokens * top_k=8 = 384 pairs across 16 experts -> average 24 pairs/expert (multi-tile
    // BLOCK_M=16 + partial tail tiles + intentionally skipped inactive experts 7 & 13).
    let num_tokens = 48usize;
    let top_k = 8usize;
    let num_experts = 16usize;
    let hidden = 128usize;
    let interm = 128usize;
    let group_size = 64usize;

    let gu_words = num_experts * (2 * interm) * (hidden / 8);
    let gu_scales_len = num_experts * (2 * interm) * (hidden / group_size);
    let dn_words = num_experts * hidden * (interm / 8);
    let dn_scales_len = num_experts * hidden * (interm / group_size);

    let mut gu_packed = vec![0u32; gu_words];
    for (i, w) in gu_packed.iter_mut().enumerate() {
        let mut word = 0u32;
        for b in 0..8 {
            let nib = (((i * 11 + b * 7 + 3) % 15) as u32) & 0x0F;
            word |= nib << (4 * b);
        }
        *w = word;
    }
    let gu_scales: Vec<u16> = (0..gu_scales_len)
        .map(|i| f32_to_f16(0.015 + 0.005 * ((i % 7) as f32)))
        .collect();

    let mut dn_packed = vec![0u32; dn_words];
    for (i, w) in dn_packed.iter_mut().enumerate() {
        let mut word = 0u32;
        for b in 0..8 {
            let nib = (((i * 13 + b * 5 + 1) % 15) as u32) & 0x0F;
            word |= nib << (4 * b);
        }
        *w = word;
    }
    let dn_scales: Vec<u16> = (0..dn_scales_len)
        .map(|i| f32_to_f16(0.012 + 0.004 * ((i % 5) as f32)))
        .collect();

    let x_host: Vec<u16> = (0..(num_tokens * hidden))
        .map(|i| f32_to_f16((((i * 17 + 5) % 29) as f32 - 14.0) * 0.05))
        .collect();
    let sh_out_host: Vec<u16> = (0..(num_tokens * hidden))
        .map(|i| f32_to_f16((((i * 9 + 2) % 19) as f32 - 9.0) * 0.03))
        .collect();
    let sh_gate_host: Vec<u16> = (0..num_tokens)
        .map(|t| f32_to_f16((t as f32 - 20.0) * 0.1))
        .collect();

    let mut topk_ids = vec![0i32; num_tokens * top_k];
    let mut topk_weights = vec![0.0f32; num_tokens * top_k];
    for t in 0..num_tokens {
        let mut w_sum = 0.0f32;
        for s in 0..top_k {
            let mut e = ((t * 5 + s * 3) % num_experts) as i32;
            if e == 7 || e == 13 {
                e = (e + 1) % (num_experts as i32);
            }
            topk_ids[t * top_k + s] = e;
            let w = 1.0f32 / ((s + 1) as f32);
            topk_weights[t * top_k + s] = w;
            w_sum += w;
        }
        for s in 0..top_k {
            topk_weights[t * top_k + s] /= w_sum;
        }
    }

    let d_x = DeviceBuffer::alloc(0, x_host.len() * 2).unwrap();
    let d_gu_w = DeviceBuffer::alloc(0, gu_packed.len() * 4).unwrap();
    let d_gu_s = DeviceBuffer::alloc(0, gu_scales.len() * 2).unwrap();
    let d_dn_w = DeviceBuffer::alloc(0, dn_packed.len() * 4).unwrap();
    let d_dn_s = DeviceBuffer::alloc(0, dn_scales.len() * 2).unwrap();
    let d_ids = DeviceBuffer::alloc(0, topk_ids.len() * 4).unwrap();
    let d_tw = DeviceBuffer::alloc(0, topk_weights.len() * 4).unwrap();
    let d_sh_out = DeviceBuffer::alloc(0, sh_out_host.len() * 2).unwrap();
    let d_sh_gate = DeviceBuffer::alloc(0, sh_gate_host.len() * 2).unwrap();
    let d_interm_ref = DeviceBuffer::alloc(0, num_tokens * top_k * interm * 2).unwrap();
    let d_interm_mma = DeviceBuffer::alloc(0, num_tokens * top_k * interm * 2).unwrap();
    let d_out_ref = DeviceBuffer::alloc(0, num_tokens * hidden * 2).unwrap();
    let d_out_mma = DeviceBuffer::alloc(0, num_tokens * hidden * 2).unwrap();
    let ws_cap = 4 * 1024 * 1024;
    let d_ws = DeviceBuffer::alloc(0, ws_cap).unwrap();

    d_x.copy_from_host(&x_host, &stream).unwrap();
    d_gu_w.copy_from_host(&gu_packed, &stream).unwrap();
    d_gu_s.copy_from_host(&gu_scales, &stream).unwrap();
    d_dn_w.copy_from_host(&dn_packed, &stream).unwrap();
    d_dn_s.copy_from_host(&dn_scales, &stream).unwrap();
    d_ids.copy_from_host(&topk_ids, &stream).unwrap();
    d_tw.copy_from_host(&topk_weights, &stream).unwrap();
    d_sh_out.copy_from_host(&sh_out_host, &stream).unwrap();
    d_sh_gate.copy_from_host(&sh_gate_host, &stream).unwrap();

    // Reference (`scratch_ptr = 0` -> warp-per-row FP32 kernel) vs Grouped `mma.sync` (`d_ws`)
    let ex_ref = stream.exec(0, 0);
    let ex_mma = stream.exec(d_ws.addr(), ws_cap);
    for (ex, d_interm, d_out) in [
        (&ex_ref, &d_interm_ref, &d_out_ref),
        (&ex_mma, &d_interm_mma, &d_out_mma),
    ] {
        assert_eq!(
            // SAFETY: All `W4A16` MoE input, weight, routing, intermediate, and output buffers are live on device 0.
            unsafe {
                ffi::w34_moe_w4a16_mlp_f16(
                    ex,
                    d_x.addr() as *const c_void,
                    d_gu_w.addr() as *const u32,
                    d_gu_s.addr() as *const c_void,
                    d_dn_w.addr() as *const u32,
                    d_dn_s.addr() as *const c_void,
                    d_ids.addr() as *const i32,
                    d_tw.addr() as *const f32,
                    d_sh_out.addr() as *const c_void,
                    d_sh_gate.addr() as *const c_void,
                    d_interm.addr() as *mut c_void,
                    d_out.addr() as *mut c_void,
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
    }

    let mut h_ref = vec![0u16; num_tokens * hidden];
    let mut h_mma = vec![0u16; num_tokens * hidden];
    d_out_ref.copy_to_host(&mut h_ref, &stream).unwrap();
    d_out_mma.copy_to_host(&mut h_mma, &stream).unwrap();

    let mut max_diff = 0.0f32;
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&r, &m) in h_ref.iter().zip(h_mma.iter()) {
        let rf = f16_to_f32(r);
        let mf = f16_to_f32(m);
        let d = (rf - mf).abs();
        max_diff = max_diff.max(d);
        diff_sq += (d as f64) * (d as f64);
        ref_sq += (rf as f64) * (rf as f64);
    }
    let rel_l2 = (diff_sq.sqrt()) / (ref_sq.sqrt().max(1e-12));
    assert!(
        max_diff < 2e-3 && rel_l2 < 1e-3,
        "Grouped mma.sync MoE vs FP32 reference drift too high: max_diff={max_diff} rel_l2={rel_l2}"
    );

    // Also verify TP=2 sharded intermediate split (`interm = 2 * 64`) summed across 2 shards
    // matches TP=1 (`interm = 128`) without shared_expert double-counting!
    let half_interm = interm / 2;
    let mut gu_w_r = [vec![0u32; gu_words / 2], vec![0u32; gu_words / 2]];
    let mut gu_s_r = [vec![0u16; gu_scales_len / 2], vec![0u16; gu_scales_len / 2]];
    let mut dn_w_r = [vec![0u32; dn_words / 2], vec![0u32; dn_words / 2]];
    let mut dn_s_r = [vec![0u16; dn_scales_len / 2], vec![0u16; dn_scales_len / 2]];
    let w_row = hidden / 8;
    let g_row = hidden / group_size;
    let dw_half = half_interm / 8;
    let dg_half = half_interm / group_size;
    for e in 0..num_experts {
        for r in 0..2 {
            // Gate rows [r*64 .. (r+1)*64] and Up rows [128 + r*64 .. 128 + (r+1)*64]
            for c in 0..half_interm {
                let dst_gate = (e * (2 * half_interm) + c) * w_row;
                let src_gate = (e * (2 * interm) + r * half_interm + c) * w_row;
                gu_w_r[r][dst_gate..dst_gate + w_row]
                    .copy_from_slice(&gu_packed[src_gate..src_gate + w_row]);
                let dst_gs = (e * (2 * half_interm) + c) * g_row;
                let src_gs = (e * (2 * interm) + r * half_interm + c) * g_row;
                gu_s_r[r][dst_gs..dst_gs + g_row]
                    .copy_from_slice(&gu_scales[src_gs..src_gs + g_row]);

                let dst_up = (e * (2 * half_interm) + half_interm + c) * w_row;
                let src_up = (e * (2 * interm) + interm + r * half_interm + c) * w_row;
                gu_w_r[r][dst_up..dst_up + w_row]
                    .copy_from_slice(&gu_packed[src_up..src_up + w_row]);
                let dst_us = (e * (2 * half_interm) + half_interm + c) * g_row;
                let src_us = (e * (2 * interm) + interm + r * half_interm + c) * g_row;
                gu_s_r[r][dst_us..dst_us + g_row]
                    .copy_from_slice(&gu_scales[src_us..src_us + g_row]);
            }
            for h in 0..hidden {
                let dst_dw = (e * hidden + h) * dw_half;
                let src_dw = (e * hidden + h) * (interm / 8) + r * dw_half;
                dn_w_r[r][dst_dw..dst_dw + dw_half]
                    .copy_from_slice(&dn_packed[src_dw..src_dw + dw_half]);
                let dst_ds = (e * hidden + h) * dg_half;
                let src_ds = (e * hidden + h) * (interm / group_size) + r * dg_half;
                dn_s_r[r][dst_ds..dst_ds + dg_half]
                    .copy_from_slice(&dn_scales[src_ds..src_ds + dg_half]);
            }
        }
    }

    assert_eq!(
        // SAFETY: All full-expert `W4A16` MoE buffers are live on device 0.
        unsafe {
            ffi::w34_moe_w4a16_mlp_f16(
                &ex_mma,
                d_x.addr() as *const c_void,
                d_gu_w.addr() as *const u32,
                d_gu_s.addr() as *const c_void,
                d_dn_w.addr() as *const u32,
                d_dn_s.addr() as *const c_void,
                d_ids.addr() as *const i32,
                d_tw.addr() as *const f32,
                core::ptr::null(),
                core::ptr::null(),
                d_interm_mma.addr() as *mut c_void,
                d_out_ref.addr() as *mut c_void,
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
    let mut h_tp1 = vec![0u16; num_tokens * hidden];
    d_out_ref.copy_to_host(&mut h_tp1, &stream).unwrap();

    let mut tp2_sum = vec![0.0f32; num_tokens * hidden];
    for r in 0..2 {
        let d_gu_wr = DeviceBuffer::alloc(0, gu_w_r[r].len() * 4).unwrap();
        let d_gu_sr = DeviceBuffer::alloc(0, gu_s_r[r].len() * 2).unwrap();
        let d_dn_wr = DeviceBuffer::alloc(0, dn_w_r[r].len() * 4).unwrap();
        let d_dn_sr = DeviceBuffer::alloc(0, dn_s_r[r].len() * 2).unwrap();
        d_gu_wr.copy_from_host(&gu_w_r[r], &stream).unwrap();
        d_gu_sr.copy_from_host(&gu_s_r[r], &stream).unwrap();
        d_dn_wr.copy_from_host(&dn_w_r[r], &stream).unwrap();
        d_dn_sr.copy_from_host(&dn_s_r[r], &stream).unwrap();
        assert_eq!(
            // SAFETY: Sharded `W4A16` MoE buffers for rank `r` are live on device 0.
            unsafe {
                ffi::w34_moe_w4a16_mlp_f16(
                    &ex_mma,
                    d_x.addr() as *const c_void,
                    d_gu_wr.addr() as *const u32,
                    d_gu_sr.addr() as *const c_void,
                    d_dn_wr.addr() as *const u32,
                    d_dn_sr.addr() as *const c_void,
                    d_ids.addr() as *const i32,
                    d_tw.addr() as *const f32,
                    core::ptr::null(),
                    core::ptr::null(),
                    d_interm_mma.addr() as *mut c_void,
                    d_out_mma.addr() as *mut c_void,
                    num_tokens as i32,
                    top_k as i32,
                    num_experts as i32,
                    half_interm as i32,
                    hidden as i32,
                    group_size as i32,
                )
            },
            0
        );
        let mut h_shard = vec![0u16; num_tokens * hidden];
        d_out_mma.copy_to_host(&mut h_shard, &stream).unwrap();
        for (acc, &v) in tp2_sum.iter_mut().zip(h_shard.iter()) {
            *acc += f16_to_f32(v);
        }
    }

    let mut tp_diff_sq = 0.0f64;
    let mut tp_ref_sq = 0.0f64;
    for (&t1, &t2) in h_tp1.iter().zip(tp2_sum.iter()) {
        let f1 = f16_to_f32(t1);
        let f2 = f16_to_f32(f32_to_f16(t2));
        let d = (f1 - f2) as f64;
        tp_diff_sq += d * d;
        tp_ref_sq += (f1 as f64) * (f1 as f64);
    }
    let tp_rel_l2 = tp_diff_sq.sqrt() / tp_ref_sq.sqrt().max(1e-12);
    assert!(
        tp_rel_l2 < 1e-3,
        "Grouped mma.sync MoE TP=1 vs TP=2 rel_l2 {tp_rel_l2} >= 1e-3"
    );
}

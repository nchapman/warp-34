use super::*;

#[test]
fn test_f16_f32_conversions_and_specials() {
    // Zero and negative zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x8000), -0.0);
    assert_eq!(f32_to_f16(0.0), 0x0000);
    assert_eq!(f32_to_f16(-0.0), 0x8000);

    // Exact powers of 2
    assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
    assert_eq!(f16_to_f32(f32_to_f16(-1.0)), -1.0);
    assert_eq!(f16_to_f32(f32_to_f16(0.5)), 0.5);
    assert_eq!(f16_to_f32(f32_to_f16(2.0)), 2.0);

    // Max representable FP16: 65504.0
    assert_eq!(f16_to_f32(0x7bff), 65504.0);
    assert_eq!(f32_to_f16(65504.0), 0x7bff);

    // Infinities
    assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
    assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
    assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);

    // NaN
    assert!(f16_to_f32(0x7e00).is_nan());
    let nan_f16 = f32_to_f16(f32::NAN);
    assert_eq!(nan_f16 & 0x7c00, 0x7c00);
    assert_ne!(nan_f16 & 0x03ff, 0);
}

#[test]
fn test_bf16_f32_conversions() {
    assert_eq!(bf16_to_f32(0x0000), 0.0);
    assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
    assert_eq!(bf16_to_f32(f32_to_bf16(-2.5)), -2.5);

    // Overflow of FP16 (e.g. 70000.0) is preserved in BF16!
    let big = 70000.0f32;
    let big_bf16 = f32_to_bf16(big);
    let roundtrip = bf16_to_f32(big_bf16);
    assert!((roundtrip - big).abs() / big < 0.01);

    // Verify SIMD slice conversions (`f16_to_f32_slice`, `f32_to_f16_slice`,
    // `bf16_to_f16_slice_inplace`) match scalar conversions across all finite values
    // plus unaligned tail lengths.
    let finite_f16: Vec<u16> = (0..=u16::MAX)
        .filter(|&v| v & 0x7c00 != 0x7c00 || v & 0x03ff == 0)
        .collect();
    let mut f32_simd = vec![0.0f32; finite_f16.len()];
    f16_to_f32_slice(&finite_f16, &mut f32_simd);
    let f32_scalar: Vec<f32> = finite_f16.iter().copied().map(f16_to_f32).collect();
    assert_eq!(
        f32_simd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        f32_scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let mut f16_roundtrip = vec![0u16; f32_simd.len()];
    f32_to_f16_slice(&f32_simd, &mut f16_roundtrip);
    assert_eq!(f16_roundtrip, finite_f16);

    let finite_bf16: Vec<u16> = (0..=u16::MAX)
        .filter(|&v| v & 0x7f80 != 0x7f80 || v & 0x007f == 0)
        .collect();
    let mut bf16_simd = finite_bf16.clone();
    bf16_to_f16_slice_inplace(&mut bf16_simd);
    let bf16_scalar: Vec<u16> = finite_bf16.iter().copied().map(bf16_to_f16).collect();
    assert_eq!(bf16_simd, bf16_scalar);
}

#[test]
fn test_capacity_planner_budgeting_logic() {
    let total_vram = 24 * 1024 * 1024 * 1024; // 24 GiB
    let free_vram = 22 * 1024 * 1024 * 1024; // 22 GiB
    let weight_arena = 8 * 1024 * 1024 * 1024; // 8 GiB
    let scratch_arena = 1024 * 1024 * 1024; // 1 GiB
    let state_bundle = 512 * 1024 * 1024; // 512 MiB
    let bytes_per_page = 16 * 1024; // 16 KiB

    // Auto page calculation
    let plan = CapacityPlanner::plan(
        total_vram,
        free_vram,
        weight_arena,
        scratch_arena,
        state_bundle,
        bytes_per_page,
        None,
    )
    .unwrap();

    assert!(plan.max_kv_pages_per_layer > 0);
    assert!(plan.headroom_bytes >= MIN_HEADROOM_BYTES);

    // Explicit requested pages exceeding capacity should fail
    let huge_req = plan.max_kv_pages_per_layer + 1_000_000;
    let err = CapacityPlanner::plan(
        total_vram,
        free_vram,
        weight_arena,
        scratch_arena,
        state_bundle,
        bytes_per_page,
        Some(huge_req),
    );
    assert!(err.is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn test_prefill_64q_zero_ulp_parity_across_kv_formats() {
    prefill_variant_fixture(false);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU microbenchmark; run serially on an idle card with --nocapture"]
fn bench_prefill_query_tiles() {
    prefill_variant_fixture(true);
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_prefill_tile_benchmark(
    ex: &W34Exec,
    stream: &CudaStream,
    arena: &mut DeviceArena,
    layout: &W34KvLayerLayout,
    write_batch: &W34AttentionBatch,
    block_table_host: &[i32],
    num_pages: usize,
    q_dev: u64,
    out_128q_addr: u64,
    fmt: i32,
) {
    // Repeated physical pages isolate tile compute/dequant costs without a model.
    // End-to-end cold-prefill benchmarks separately measure real HBM working sets.
    let table: Vec<i32> = (0..4096).map(|i| block_table_host[i % num_pages]).collect();
    let pages = arena.upload_slice(&table, stream).unwrap();
    for heads in [16, 24] {
        for (queries, context) in [(256, 8192), (1024, 8192), (1024, 64512)] {
            let starts = arena.upload_slice(&[0i32, queries], stream).unwrap();
            let ctx = arena.upload_slice(&[context], stream).unwrap();
            let seq = arena.upload_slice(&[context + queries], stream).unwrap();
            let batch = W34AttentionBatch {
                query_start_loc: starts as *const i32,
                context_lens_before: ctx as *const i32,
                seq_lens_after: seq as *const i32,
                block_tables: pages as *const i32,
                write_slot_mapping: core::ptr::null(),
                total_tokens: queries,
                max_blocks_per_req: 4096,
                max_q_len: queries,
                max_seq_len: context + queries,
                ..*write_batch
            };
            for variant in [1, 2] {
                let invoke = || {
                    check_rc(
                        // SAFETY: `q_dev`, `out_128q_addr`, `layout`, and `batch` pointers
                        // live in `arena`/`k_pool`/`v_pool` for the duration of the test.
                        unsafe {
                            ffi::w34_paged_attention_f16_force_kernel_for_test(
                                ex,
                                layout,
                                &batch,
                                q_dev as *const c_void,
                                out_128q_addr as *mut c_void,
                                heads,
                                variant,
                            )
                        },
                        "prefill tile benchmark",
                    )
                    .unwrap()
                };
                invoke();
                stream.synchronize().unwrap();
                let start = CudaEvent::new(0, true).unwrap();
                let end = CudaEvent::new(0, true).unwrap();
                start.record(stream).unwrap();
                for _ in 0..5 {
                    invoke();
                }
                end.record(stream).unwrap();
                stream.synchronize().unwrap();
                eprintln!(
                    "tile_bench fmt={fmt} heads={heads} q={queries} ctx={context} rows={} ms={:.4}",
                    if variant == 1 { 64 } else { 128 },
                    start.elapsed_ms(&end).unwrap() / 5.0
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn prefill_variant_fixture(benchmark: bool) {
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut arena = DeviceArena::new(0, 64 * 1024 * 1024).unwrap();
    let ex = stream.exec(arena.buf.addr(), 4 * 1024 * 1024);
    let _ = arena.alloc_aligned(4 * 1024 * 1024).unwrap();

    let num_pages = 32usize;
    let kv_heads = 4usize;
    let head_dim = 256usize;
    let max_tokens = num_pages * 16; // 512

    // Deterministic pseudo-random FP16 inputs for Q, K, and V across 512 tokens.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_f16 = || -> u16 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let val = ((rng & 0xFFFF) as f32 / 65535.0 - 0.5) * 1.5;
        f32_to_f16(val)
    };

    let max_q_heads = 24usize;
    let query_capacity = if benchmark { 1024 } else { max_tokens };
    let q_host: Vec<u16> = (0..query_capacity * max_q_heads * head_dim)
        .map(|_| next_f16())
        .collect();
    let k_host: Vec<u16> = (0..max_tokens * kv_heads * head_dim)
        .map(|_| next_f16())
        .collect();
    let v_host: Vec<u16> = (0..max_tokens * kv_heads * head_dim)
        .map(|_| next_f16())
        .collect();

    let q_dev = arena.upload_slice(&q_host, &stream).unwrap();
    let k_in_dev = arena.upload_slice(&k_host, &stream).unwrap();
    let v_in_dev = arena.upload_slice(&v_host, &stream).unwrap();
    let out_16q = DeviceBuffer::alloc(0, query_capacity * max_q_heads * head_dim * 2).unwrap();
    let out_64q = DeviceBuffer::alloc(0, query_capacity * max_q_heads * head_dim * 2).unwrap();
    let out_128q = DeviceBuffer::alloc(0, query_capacity * max_q_heads * head_dim * 2).unwrap();

    // Permuted page table to exercise non-contiguous physical page lookups.
    let block_table_host: Vec<i32> = (0..num_pages as i32)
        .map(|i| (i * 7 + 3) % (num_pages as i32))
        .collect();
    let block_table_dev = arena.upload_slice(&block_table_host, &stream).unwrap();
    let write_slots_host: Vec<i32> = (0..max_tokens)
        .map(|t| block_table_host[t / 16] * 16 + (t % 16) as i32)
        .collect();
    let write_slots_dev = arena.upload_slice(&write_slots_host, &stream).unwrap();

    for (fmt, transform, row_bytes) in [
        (
            ffi::W34_KV_FORMAT_FP16,
            ffi::W34_KV_TRANSFORM_IDENTITY,
            512usize,
        ),
        (
            ffi::W34_KV_FORMAT_Q8_0,
            ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
            272usize,
        ),
        (
            ffi::W34_KV_FORMAT_TQ4,
            ffi::W34_KV_TRANSFORM_RADEMACHER_WHT256,
            132usize,
        ),
    ] {
        let pool_bytes = num_pages * kv_heads * 16 * row_bytes;
        let k_pool = DeviceBuffer::alloc(0, pool_bytes).unwrap();
        let v_pool = DeviceBuffer::alloc(0, pool_bytes).unwrap();
        let layout = W34KvLayerLayout {
            k_base: k_pool.addr(),
            v_base: v_pool.addr(),
            k_page_stride_bytes: (kv_heads * 16 * row_bytes) as u64,
            v_page_stride_bytes: (kv_heads * 16 * row_bytes) as u64,
            k_head_stride_bytes: (16 * row_bytes) as u64,
            v_head_stride_bytes: (16 * row_bytes) as u64,
            k_token_stride_bytes: row_bytes as u64,
            v_token_stride_bytes: row_bytes as u64,
            namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
            layer_id: 0,
            local_kv_heads: kv_heads as i32,
            head_dim: head_dim as i32,
            block_size: 16,
            k_format: fmt,
            v_format: fmt,
            k_transform: transform,
            v_transform: transform,
            reserved: 0,
        };

        // Populate all 512 KV positions via the real w34_kv_cache_write_strided kernel.
        let q_start_all = arena
            .upload_slice(&[0i32, max_tokens as i32], &stream)
            .unwrap();
        let ctx_zero = arena.upload_slice(&[0i32], &stream).unwrap();
        let seq_all = arena.upload_slice(&[max_tokens as i32], &stream).unwrap();
        let write_batch = W34AttentionBatch {
            query_start_loc: q_start_all as *const i32,
            context_lens_before: ctx_zero as *const i32,
            seq_lens_after: seq_all as *const i32,
            block_tables: block_table_dev as *const i32,
            write_slot_mapping: write_slots_dev as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: max_tokens as i32,
            max_blocks_per_req: num_pages as i32,
            max_q_len: max_tokens as i32,
            max_seq_len: max_tokens as i32,
            mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
            sliding_window: 0,
            sm_scale: 0.0625,
            reserved: 0,
        };
        check_rc(
            // SAFETY: `k_pool`/`v_pool`, `write_batch` arrays, and `k_in_dev`/`v_in_dev` are
            // live device allocations sized for `max_tokens * kv_heads * head_dim`.
            unsafe {
                ffi::w34_kv_cache_write(
                    &ex,
                    &layout,
                    &write_batch,
                    k_in_dev as *const c_void,
                    v_in_dev as *const c_void,
                )
            },
            "populate kv pool for 64q test",
        )
        .unwrap();

        if benchmark {
            run_prefill_tile_benchmark(
                &ex,
                &stream,
                &mut arena,
                &layout,
                &write_batch,
                &block_table_host,
                num_pages,
                q_dev,
                out_128q.addr(),
                fmt,
            );
            continue;
        }

        // Test both 9B/35B GQA=4 (16 Q heads) and 27B GQA=6 (24 Q heads),
        // including odd chunk sizes and non-zero prefix offsets.
        for num_q_heads in [16usize, 24usize] {
            for (q_len, ctx_before) in [(64usize, 0usize), (65, 19), (137, 128), (256, 200)] {
                let seq_after = ctx_before + q_len;
                let q_start = arena.upload_slice(&[0i32, q_len as i32], &stream).unwrap();
                let ctx_dev = arena.upload_slice(&[ctx_before as i32], &stream).unwrap();
                let seq_dev = arena.upload_slice(&[seq_after as i32], &stream).unwrap();
                let batch = W34AttentionBatch {
                    query_start_loc: q_start as *const i32,
                    context_lens_before: ctx_dev as *const i32,
                    seq_lens_after: seq_dev as *const i32,
                    block_tables: block_table_dev as *const i32,
                    write_slot_mapping: core::ptr::null(),
                    valid_token_mask: core::ptr::null(),
                    valid_req_mask: core::ptr::null(),
                    num_reqs: 1,
                    total_tokens: q_len as i32,
                    max_blocks_per_req: num_pages as i32,
                    max_q_len: q_len as i32,
                    max_seq_len: seq_after as i32,
                    mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
                    sliding_window: 0,
                    sm_scale: 0.0625,
                    reserved: 0,
                };
                for (use_64q, dst) in [
                    (0i32, out_16q.addr()),
                    (1i32, out_64q.addr()),
                    (2i32, out_128q.addr()),
                ] {
                    check_rc(
                        // SAFETY: `q_dev`, `dst`, `layout`, and `batch` pointers are live device
                        // buffers sized for `q_len * num_q_heads * head_dim`.
                        unsafe {
                            ffi::w34_paged_attention_f16_force_kernel_for_test(
                                &ex,
                                &layout,
                                &batch,
                                q_dev as *const c_void,
                                dst as *mut c_void,
                                num_q_heads as i32,
                                use_64q,
                            )
                        },
                        "run 16q vs 64q paged attention",
                    )
                    .unwrap();
                }

                let elems = q_len * num_q_heads * head_dim;
                let mut h16 = vec![0u16; elems];
                let mut h64 = vec![0u16; elems];
                let mut h128 = vec![0u16; elems];
                out_128q.copy_to_host(&mut h128, &stream).unwrap();
                out_16q.copy_to_host(&mut h16, &stream).unwrap();
                out_64q.copy_to_host(&mut h64, &stream).unwrap();
                stream.synchronize().unwrap();
                assert_eq!(
                    h16, h128,
                    "128q mismatch fmt={fmt} heads={num_q_heads} q={q_len} ctx={ctx_before}"
                );
                for idx in 0..elems {
                    assert_eq!(
                        h16[idx], h64[idx],
                        "non-zero ULP mismatch fmt={fmt} q_heads={num_q_heads} q_len={q_len} ctx={ctx_before} idx={idx}: 16q=0x{:04x} vs 64q=0x{:04x}",
                        h16[idx], h64[idx]
                    );
                }
            }

            // Multi-request mixed decode + concurrent prefill (num_reqs = 3: 1 decode + 99 prefill + 156 prefill)
            let q_starts_mr = [0i32, 1, 100, 256];
            let ctx_mr = [143i32, 64, 95];
            let seq_mr = [144i32, 163, 251];
            let max_blk_mr = 16usize;
            let mut bt_mr = Vec::with_capacity(3 * max_blk_mr);
            for r in 0..3usize {
                for p in 0..max_blk_mr {
                    bt_mr.push(((p + r * 5) % num_pages) as i32);
                }
            }
            let q_start_dev = arena.upload_slice(&q_starts_mr, &stream).unwrap();
            let ctx_mr_dev = arena.upload_slice(&ctx_mr, &stream).unwrap();
            let seq_mr_dev = arena.upload_slice(&seq_mr, &stream).unwrap();
            let bt_mr_dev = arena.upload_slice(&bt_mr, &stream).unwrap();
            let mr_batch = W34AttentionBatch {
                query_start_loc: q_start_dev as *const i32,
                context_lens_before: ctx_mr_dev as *const i32,
                seq_lens_after: seq_mr_dev as *const i32,
                block_tables: bt_mr_dev as *const i32,
                write_slot_mapping: core::ptr::null(),
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 3,
                total_tokens: 256,
                max_blocks_per_req: max_blk_mr as i32,
                max_q_len: 156,
                max_seq_len: 251,
                mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
                sliding_window: 0,
                sm_scale: 0.0625,
                reserved: 0,
            };
            for (use_64q, dst) in [
                (0i32, out_16q.addr()),
                (1i32, out_64q.addr()),
                (2i32, out_128q.addr()),
            ] {
                check_rc(
                    // SAFETY: `q_dev`, `dst`, `layout`, and `mr_batch` pointers are live device
                    // buffers sized for `256 * num_q_heads * head_dim`.
                    unsafe {
                        ffi::w34_paged_attention_f16_force_kernel_for_test(
                            &ex,
                            &layout,
                            &mr_batch,
                            q_dev as *const c_void,
                            dst as *mut c_void,
                            num_q_heads as i32,
                            use_64q,
                        )
                    },
                    "run multi-req 16q vs 64q paged attention",
                )
                .unwrap();
            }
            let elems = 256 * num_q_heads * head_dim;
            let mut h16 = vec![0u16; elems];
            let mut h64 = vec![0u16; elems];
            let mut h128 = vec![0u16; elems];
            out_128q.copy_to_host(&mut h128, &stream).unwrap();
            out_16q.copy_to_host(&mut h16, &stream).unwrap();
            out_64q.copy_to_host(&mut h64, &stream).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(
                h16, h128,
                "multi-req 128q mismatch fmt={fmt} heads={num_q_heads}"
            );
            for idx in 0..elems {
                assert_eq!(
                    h16[idx], h64[idx],
                    "multi-req non-zero ULP mismatch fmt={fmt} q_heads={num_q_heads} idx={idx}: 16q=0x{:04x} vs 64q=0x{:04x}",
                    h16[idx], h64[idx]
                );
            }

            // Skip the middle request entirely and mask rows around tile boundaries.
            // Both kernels must preserve inactive outputs, including all-masked tiles.
            let mut valid_tokens = vec![1u8; 256];
            valid_tokens[100..164].fill(0);
            valid_tokens[255] = 0;
            let valid_tokens_dev = arena.upload_slice(&valid_tokens, &stream).unwrap();
            let valid_reqs_dev = arena.upload_slice(&[1u8, 0, 1], &stream).unwrap();
            let masked_batch = W34AttentionBatch {
                valid_token_mask: valid_tokens_dev as *const u8,
                valid_req_mask: valid_reqs_dev as *const u8,
                sliding_window: 33,
                ..mr_batch
            };
            for (variant, dst) in [(0, &out_16q), (2, &out_128q)] {
                dst.copy_from_host(&vec![0x5500u16; elems], &stream)
                    .unwrap();
                check_rc(
                    // SAFETY: `q_dev`, `dst`, `layout`, and `masked_batch` pointers (including
                    // `valid_tokens_dev` and `valid_reqs_dev`) are live device buffers.
                    unsafe {
                        ffi::w34_paged_attention_f16_force_kernel_for_test(
                            &ex,
                            &layout,
                            &masked_batch,
                            q_dev as *const c_void,
                            dst.addr() as *mut c_void,
                            num_q_heads as i32,
                            variant,
                        )
                    },
                    "masked sliding-window 16q vs 128q",
                )
                .unwrap();
            }
            out_16q.copy_to_host(&mut h16, &stream).unwrap();
            out_128q.copy_to_host(&mut h128, &stream).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(
                h16, h128,
                "masked 128q mismatch fmt={fmt} heads={num_q_heads}"
            );
            for row in 0..256 {
                if (1..100).contains(&row) || valid_tokens[row] == 0 {
                    assert!(
                        h128[row * num_q_heads * head_dim..(row + 1) * num_q_heads * head_dim]
                            .iter()
                            .all(|&v| v == 0x5500)
                    );
                }
            }
        }
    }
}

#[test]
fn test_cuda_error_display_includes_runtime_message_only_for_cuda_codes() {
    let synthetic = CudaError::with_detail(-10, "model loader error", "missing tensor");
    assert_eq!(synthetic.runtime_error_string(), None);
    assert_eq!(
        synthetic.to_string(),
        "CUDA error -10 in model loader error: missing tensor"
    );

    let cublas = CudaError::new(3, "w34_cublas_create");
    assert_eq!(cublas.runtime_error_string(), None);
    assert_eq!(cublas.to_string(), "CUDA error 3 in w34_cublas_create");

    #[cfg(feature = "cuda")]
    {
        let oom = CudaError::new(2, "w34_malloc");
        assert_eq!(oom.runtime_error_string(), Some("out of memory"));
        assert_eq!(
            oom.to_string(),
            "CUDA error 2 (out of memory) in w34_malloc"
        );
    }
}

#[test]
fn test_device_props_validation_across_30_and_40_series_lineup() {
    // 30-series lineup: RTX 3060 (28 SMs, 3 MiB L2) -> RTX 3090 Ti (84 SMs, 6 MiB L2)
    for (name, sms, l2_mib) in [
        ("NVIDIA GeForce RTX 3060", 28, 3),
        ("NVIDIA GeForce RTX 3090", 82, 6),
        ("NVIDIA GeForce RTX 3090 Ti", 84, 6),
    ] {
        let props = CudaDeviceProps {
            device_id: 0,
            compute_major: 8,
            compute_minor: 6,
            multi_processor_count: sms,
            max_shared_mem_per_block_optin: 101_376,
            max_shared_mem_per_multiprocessor: 102_400,
            l2_cache_bytes: l2_mib * 1024 * 1024,
            persisting_l2_max_bytes: (l2_mib * 3 / 4) * 1024 * 1024,
            name: name.to_string(),
        };
        assert_eq!(props.sm_tag(), "sm_86");
        assert_eq!(
            props.validate_supported().unwrap(),
            CudaArchFamily::AmpereSm86
        );
        assert_eq!(props.marlin_workspace_ints(), (sms as usize) * 64);
        assert_eq!(props.marlin_workspace_bytes(), (sms as usize) * 64 * 4);
    }

    // 40-series lineup: RTX 4060 (24 SMs, 24 MiB L2) -> RTX 4090 (128 SMs, 72 MiB L2) -> L40S (142 SMs, 96 MiB L2)
    for (name, sms, l2_mib) in [
        ("NVIDIA GeForce RTX 4060", 24, 24),
        ("NVIDIA GeForce RTX 4080 SUPER", 80, 64),
        ("NVIDIA GeForce RTX 4090", 128, 72),
        ("NVIDIA L40S", 142, 96),
    ] {
        let props = CudaDeviceProps {
            device_id: 0,
            compute_major: 8,
            compute_minor: 9,
            multi_processor_count: sms,
            max_shared_mem_per_block_optin: 101_376,
            max_shared_mem_per_multiprocessor: 102_400,
            l2_cache_bytes: l2_mib * 1024 * 1024,
            persisting_l2_max_bytes: 50 * 1024 * 1024,
            name: name.to_string(),
        };
        assert_eq!(props.sm_tag(), "sm_89");
        assert_eq!(props.validate_supported().unwrap(), CudaArchFamily::AdaSm89);
        assert_eq!(props.marlin_workspace_ints(), (sms as usize) * 64);
        assert_eq!(props.marlin_workspace_bytes(), (sms as usize) * 64 * 4);
    }

    // Future / larger-SM GPU (> 144 SMs) scales marlin workspace automatically.
    let large_sm_props = CudaDeviceProps {
        device_id: 0,
        compute_major: 9,
        compute_minor: 0,
        multi_processor_count: 160,
        max_shared_mem_per_block_optin: 232_448,
        max_shared_mem_per_multiprocessor: 233_472,
        l2_cache_bytes: 50 * 1024 * 1024,
        persisting_l2_max_bytes: 36 * 1024 * 1024,
        name: "NVIDIA H200".to_string(),
    };
    assert_eq!(
        large_sm_props.validate_supported().unwrap(),
        CudaArchFamily::OtherAmpereOrNewer { major: 9, minor: 0 }
    );
    assert_eq!(large_sm_props.marlin_workspace_ints(), 160 * 64);
    assert_eq!(large_sm_props.marlin_workspace_bytes(), 160 * 64 * 4);

    // Pre-sm_86 (< 8.6, e.g. Turing sm_75 or sm_80 without sm_80 SASS in the default fatbinary) is rejected.
    for (major, minor, name) in [
        (7, 5, "NVIDIA GeForce RTX 2080 Ti"),
        (8, 0, "NVIDIA A100-SXM4-80GB"),
    ] {
        let unsupported = CudaDeviceProps {
            compute_major: major,
            compute_minor: minor,
            name: name.to_string(),
            ..large_sm_props.clone()
        };
        assert!(unsupported.validate_supported().is_err());
    }

    // Insufficient opt-in shared memory is rejected.
    let small_smem = CudaDeviceProps {
        max_shared_mem_per_block_optin: 49_152,
        ..large_sm_props
    };
    assert!(small_smem.validate_supported().is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn test_runtime_gpu_device_props_and_tp2_sm_cap() {
    let count = CudaDevice::count().unwrap();
    assert!(count >= 1);
    let dev = CudaDevice::set_current(0).unwrap();
    let props = dev.props().unwrap();
    assert_eq!(props.device_id, 0);
    assert!(props.compute_major >= 8);
    assert!(props.multi_processor_count >= 24);
    assert!(props.max_shared_mem_per_block_optin >= MIN_REQUIRED_OPTIN_SMEM_BYTES);
    assert!(props.l2_cache_bytes >= 2 * 1024 * 1024);
    assert!(!props.name.is_empty());
    let family = props.validate_supported().unwrap();
    assert!(matches!(
        family,
        CudaArchFamily::AmpereSm86
            | CudaArchFamily::AdaSm89
            | CudaArchFamily::OtherAmpereOrNewer { .. }
    ));

    let mut cur_raw = ffi::W34DeviceProps::default();
    check_rc(
        // SAFETY: `cur_raw` is a valid initialized stack `W34DeviceProps`.
        unsafe { ffi::w34_get_current_device_props(&raw mut cur_raw) },
        "w34_get_current_device_props",
    )
    .unwrap();
    assert_eq!(cur_raw.multi_processor_count, props.multi_processor_count);

    let mut min_sms = 0i32;
    check_rc(
        // SAFETY: `min_sms` is a valid initialized stack `i32`.
        unsafe { ffi::w34_get_tp2_min_sm_count(&raw mut min_sms) },
        "w34_get_tp2_min_sm_count",
    )
    .unwrap();
    assert!(min_sms >= 24 && min_sms <= props.multi_processor_count);
}

#[cfg(feature = "cuda")]
fn strict_test_env() -> bool {
    ["W34_STRICT_TESTS", "CI"].into_iter().any(|var| {
        std::env::var(var).is_ok_and(|val| {
            let trimmed = val.trim();
            trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
        })
    })
}

#[cfg(feature = "cuda")]
#[test]
fn test_tp2_collectives_entrypoints_and_validation() {
    assert!(Tp2Collectives::new_for_tp(1, 4096).is_err());
    assert!(Tp2Collectives::new_for_tp(3, 4096).is_err());
    if CudaDevice::count().unwrap_or(0) < 2 {
        assert!(
            !strict_test_env(),
            "test_tp2_collectives_entrypoints_and_validation requires 2 GPUs"
        );
        return;
    }
    let comm = Tp2Collectives::new_for_tp_with_p2p(2, 128 * 1024, false).unwrap();
    assert_eq!(comm.tp_size, 2);
    assert!(!comm.p2p_vram);

    let s0 = CudaStream::new(0).unwrap();
    let s1 = CudaStream::new(1).unwrap();

    // 1. all_reduce_f16, all_reduce_bf16, all_reduce_f32, broadcast_i32,
    //    all_gather_top1_f32_i32, and gather_vocab_f16 with distinct buffers and concurrent rank launches.
    let n = 128usize;
    let d0_ar_f16 = DeviceBuffer::alloc(0, n * 2).unwrap();
    let d1_ar_f16 = DeviceBuffer::alloc(1, n * 2).unwrap();
    d0_ar_f16
        .copy_from_host(&vec![f32_to_f16(1.0); n], &s0)
        .unwrap();
    d1_ar_f16
        .copy_from_host(&vec![f32_to_f16(2.0); n], &s1)
        .unwrap();

    let d0_ar_bf16 = DeviceBuffer::alloc(0, n * 2).unwrap();
    let d1_ar_bf16 = DeviceBuffer::alloc(1, n * 2).unwrap();
    d0_ar_bf16
        .copy_from_host(&vec![f32_to_bf16(1.25); n], &s0)
        .unwrap();
    d1_ar_bf16
        .copy_from_host(&vec![f32_to_bf16(2.75); n], &s1)
        .unwrap();

    let d0_ar_f32 = DeviceBuffer::alloc(0, 64 * 4).unwrap();
    let d1_ar_f32 = DeviceBuffer::alloc(1, 64 * 4).unwrap();
    d0_ar_f32.copy_from_host(&[1.5f32; 64], &s0).unwrap();
    d1_ar_f32.copy_from_host(&[2.5f32; 64], &s1).unwrap();

    let d0_bcast_i32 = DeviceBuffer::alloc(0, 16 * 4).unwrap();
    let d1_bcast_i32 = DeviceBuffer::alloc(1, 16 * 4).unwrap();
    let bcast_in: Vec<i32> = (0..16).map(|i| 42 + i).collect();
    d0_bcast_i32.copy_from_host(&bcast_in, &s0).unwrap();

    let d0_top1_val_in = DeviceBuffer::alloc(0, 4 * 4).unwrap();
    let d1_top1_val_in = DeviceBuffer::alloc(1, 4 * 4).unwrap();
    let d0_top1_idx_in = DeviceBuffer::alloc(0, 4 * 4).unwrap();
    let d1_top1_idx_in = DeviceBuffer::alloc(1, 4 * 4).unwrap();
    let d0_top1_val_out = DeviceBuffer::alloc(0, 4 * 4).unwrap();
    let d1_top1_val_out = DeviceBuffer::alloc(1, 4 * 4).unwrap();
    let d0_top1_idx_out = DeviceBuffer::alloc(0, 4 * 4).unwrap();
    let d1_top1_idx_out = DeviceBuffer::alloc(1, 4 * 4).unwrap();
    d0_top1_val_in
        .copy_from_host(&[9.0f32, 3.0, 8.0, 2.0], &s0)
        .unwrap();
    d1_top1_val_in
        .copy_from_host(&[4.0f32, 7.0, 5.0, 6.0], &s1)
        .unwrap();
    d0_top1_idx_in
        .copy_from_host(&[10i32, 11, 12, 13], &s0)
        .unwrap();
    d1_top1_idx_in
        .copy_from_host(&[20i32, 21, 22, 23], &s1)
        .unwrap();

    let d0_vocab_in = DeviceBuffer::alloc(0, n * 2).unwrap();
    let d1_vocab_in = DeviceBuffer::alloc(1, n * 2).unwrap();
    let d0_vocab_full = DeviceBuffer::alloc(0, n * 4).unwrap();
    d0_vocab_in
        .copy_from_host(&vec![f32_to_f16(1.0); n], &s0)
        .unwrap();
    d1_vocab_in
        .copy_from_host(&vec![f32_to_f16(2.0); n], &s1)
        .unwrap();
    s0.synchronize().unwrap();
    s1.synchronize().unwrap();

    std::thread::scope(|scope| {
        let comm_ref = &comm;
        let s0_ref = &s0;
        let s1_ref = &s1;
        let d0_ar_f16_addr = d0_ar_f16.addr();
        let d1_ar_f16_addr = d1_ar_f16.addr();
        let d0_ar_bf16_addr = d0_ar_bf16.addr();
        let d1_ar_bf16_addr = d1_ar_bf16.addr();
        let d0_ar_f32_addr = d0_ar_f32.addr();
        let d1_ar_f32_addr = d1_ar_f32.addr();
        let d0_bcast_addr = d0_bcast_i32.addr();
        let d1_bcast_addr = d1_bcast_i32.addr();
        let d0_tv_in = d0_top1_val_in.addr();
        let d1_tv_in = d1_top1_val_in.addr();
        let d0_ti_in = d0_top1_idx_in.addr();
        let d1_ti_in = d1_top1_idx_in.addr();
        let d0_tv_out = d0_top1_val_out.addr();
        let d1_tv_out = d1_top1_val_out.addr();
        let d0_ti_out = d0_top1_idx_out.addr();
        let d1_ti_out = d1_top1_idx_out.addr();
        let d0_vin = d0_vocab_in.addr();
        let d1_vin = d1_vocab_in.addr();
        let d0_vfull = d0_vocab_full.addr();

        let t0 = scope.spawn(move || {
            let _dev = CudaDevice::set_current(0).unwrap();
            let ex0 = s0_ref.exec(0, 0);
            comm_ref
                .all_reduce_f16(&ex0, d0_ar_f16_addr, d0_ar_f16_addr, 0, n as i32)
                .unwrap();
            comm_ref
                .all_reduce_bf16(&ex0, d0_ar_bf16_addr, d0_ar_bf16_addr, 0, n as i32)
                .unwrap();
            comm_ref
                .all_reduce_f32(&ex0, d0_ar_f32_addr, d0_ar_f32_addr, 0, 64)
                .unwrap();
            comm_ref
                .broadcast_i32(&ex0, d0_bcast_addr, d0_bcast_addr, 0, 16)
                .unwrap();
            comm_ref
                .all_gather_top1_f32_i32(&ex0, d0_tv_in, d0_ti_in, d0_tv_out, d0_ti_out, 0, 4)
                .unwrap();
            comm_ref
                .gather_vocab_f16(&ex0, d0_vin, d0_vfull, 0, 1, n as i32)
                .unwrap();
            s0_ref.synchronize().unwrap();
        });
        let t1 = scope.spawn(move || {
            let _dev = CudaDevice::set_current(1).unwrap();
            let ex1 = s1_ref.exec(0, 0);
            comm_ref
                .all_reduce_f16(&ex1, d1_ar_f16_addr, d1_ar_f16_addr, 1, n as i32)
                .unwrap();
            comm_ref
                .all_reduce_bf16(&ex1, d1_ar_bf16_addr, d1_ar_bf16_addr, 1, n as i32)
                .unwrap();
            comm_ref
                .all_reduce_f32(&ex1, d1_ar_f32_addr, d1_ar_f32_addr, 1, 64)
                .unwrap();
            comm_ref
                .broadcast_i32(&ex1, d1_bcast_addr, d1_bcast_addr, 1, 16)
                .unwrap();
            comm_ref
                .all_gather_top1_f32_i32(&ex1, d1_tv_in, d1_ti_in, d1_tv_out, d1_ti_out, 1, 4)
                .unwrap();
            comm_ref
                .gather_vocab_f16(&ex1, d1_vin, 0, 1, 1, n as i32)
                .unwrap();
            s1_ref.synchronize().unwrap();
        });
        t0.join().unwrap();
        t1.join().unwrap();
    });

    let mut h0_f16 = vec![0u16; n];
    let mut h1_f16 = vec![0u16; n];
    d0_ar_f16.copy_to_host(&mut h0_f16, &s0).unwrap();
    d1_ar_f16.copy_to_host(&mut h1_f16, &s1).unwrap();
    let mut h0_bf16 = vec![0u16; n];
    let mut h1_bf16 = vec![0u16; n];
    d0_ar_bf16.copy_to_host(&mut h0_bf16, &s0).unwrap();
    d1_ar_bf16.copy_to_host(&mut h1_bf16, &s1).unwrap();
    let mut h0_f32 = vec![0.0f32; 64];
    let mut h1_f32 = vec![0.0f32; 64];
    d0_ar_f32.copy_to_host(&mut h0_f32, &s0).unwrap();
    d1_ar_f32.copy_to_host(&mut h1_f32, &s1).unwrap();
    let mut bcast_out = vec![0i32; 16];
    d1_bcast_i32.copy_to_host(&mut bcast_out, &s1).unwrap();
    let mut top1_v0 = vec![0.0f32; 4];
    let mut top1_i0 = vec![0i32; 4];
    let mut top1_v1 = vec![0.0f32; 4];
    let mut top1_i1 = vec![0i32; 4];
    d0_top1_val_out.copy_to_host(&mut top1_v0, &s0).unwrap();
    d0_top1_idx_out.copy_to_host(&mut top1_i0, &s0).unwrap();
    d1_top1_val_out.copy_to_host(&mut top1_v1, &s1).unwrap();
    d1_top1_idx_out.copy_to_host(&mut top1_i1, &s1).unwrap();
    let mut vocab_out = vec![0u16; n * 2];
    d0_vocab_full.copy_to_host(&mut vocab_out, &s0).unwrap();
    s0.synchronize().unwrap();
    s1.synchronize().unwrap();

    assert_eq!(h0_f16, vec![f32_to_f16(3.0); n]);
    assert_eq!(h1_f16, vec![f32_to_f16(3.0); n]);
    assert_eq!(h0_bf16, vec![f32_to_bf16(4.0); n]);
    assert_eq!(h1_bf16, vec![f32_to_bf16(4.0); n]);
    assert_eq!(h0_f32, vec![4.0f32; 64]);
    assert_eq!(h1_f32, vec![4.0f32; 64]);
    assert_eq!(bcast_out, bcast_in);
    assert_eq!(top1_v0, vec![9.0, 7.0, 8.0, 6.0]);
    assert_eq!(top1_i0, vec![10, 21, 12, 23]);
    assert_eq!(top1_v1, top1_v0);
    assert_eq!(top1_i1, top1_i0);
    assert_eq!(&vocab_out[..n], &vec![f32_to_f16(1.0); n][..]);
    assert_eq!(&vocab_out[n..], &vec![f32_to_f16(2.0); n][..]);

    verify_tp2_fused_residual_rms_norm_regimes(&comm, &s0, &s1);
    verify_tp2_dma_crossover_and_graph_replay(&s0, &s1);
}

#[cfg(feature = "cuda")]
fn verify_tp2_fused_residual_rms_norm_regimes(
    comm: &Tp2Collectives,
    s0: &CudaStream,
    s1: &CudaStream,
) {
    // 2. Verify all_reduce_residual_gemma_rms_norm_f16 across all 4 hidden_size template regimes
    //    (<=2560, <=4096, <=5120, >5120) including ar_out, residual, out, and tap_dst_bf16.
    for hidden in [128usize, 2560, 4096, 5120, 6144] {
        let d0_in = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d1_in = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d0_ar = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d1_ar = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d0_res = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d1_res = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d0_w = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d1_w = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d0_out = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d1_out = DeviceBuffer::alloc(1, hidden * 2).unwrap();
        let d0_tap = DeviceBuffer::alloc(0, hidden * 2).unwrap();
        let d1_tap = DeviceBuffer::alloc(1, hidden * 2).unwrap();

        d0_in
            .copy_from_host(&vec![f32_to_f16(1.0); hidden], s0)
            .unwrap();
        d1_in
            .copy_from_host(&vec![f32_to_f16(2.0); hidden], s1)
            .unwrap();
        d0_res
            .copy_from_host(&vec![f32_to_f16(0.5); hidden], s0)
            .unwrap();
        d1_res
            .copy_from_host(&vec![f32_to_f16(0.5); hidden], s1)
            .unwrap();
        d0_w.copy_from_host(&vec![f32_to_f16(0.25); hidden], s0)
            .unwrap();
        d1_w.copy_from_host(&vec![f32_to_f16(0.25); hidden], s1)
            .unwrap();
        s0.synchronize().unwrap();
        s1.synchronize().unwrap();

        std::thread::scope(|scope| {
            let (in0, ar0, res0, w0, out0, tap0) = (
                d0_in.addr(),
                d0_ar.addr(),
                d0_res.addr(),
                d0_w.addr(),
                d0_out.addr(),
                d0_tap.addr(),
            );
            let (in1, ar1, res1, w1, out1, tap1) = (
                d1_in.addr(),
                d1_ar.addr(),
                d1_res.addr(),
                d1_w.addr(),
                d1_out.addr(),
                d1_tap.addr(),
            );
            let t0 = scope.spawn(move || {
                let _dev = CudaDevice::set_current(0).unwrap();
                let ex0 = s0.exec(0, 0);
                comm.all_reduce_residual_gemma_rms_norm_f16(
                    &ex0,
                    in0,
                    ar0,
                    res0,
                    w0,
                    out0,
                    tap0,
                    0,
                    1,
                    hidden as i32,
                    1e-6,
                    hidden as i32,
                    0,
                )
                .unwrap();
                s0.synchronize().unwrap();
            });
            let t1 = scope.spawn(move || {
                let _dev = CudaDevice::set_current(1).unwrap();
                let ex1 = s1.exec(0, 0);
                comm.all_reduce_residual_gemma_rms_norm_f16(
                    &ex1,
                    in1,
                    ar1,
                    res1,
                    w1,
                    out1,
                    tap1,
                    1,
                    1,
                    hidden as i32,
                    1e-6,
                    hidden as i32,
                    0,
                )
                .unwrap();
                s1.synchronize().unwrap();
            });
            t0.join().unwrap();
            t1.join().unwrap();
        });

        let mut h_ar = vec![0u16; hidden];
        let mut h_res = vec![0u16; hidden];
        let mut h_out = vec![0u16; hidden];
        let mut h_tap = vec![0u16; hidden];
        d0_ar.copy_to_host(&mut h_ar, s0).unwrap();
        d0_res.copy_to_host(&mut h_res, s0).unwrap();
        d0_out.copy_to_host(&mut h_out, s0).unwrap();
        d0_tap.copy_to_host(&mut h_tap, s0).unwrap();
        s0.synchronize().unwrap();

        assert_eq!(h_ar, vec![f32_to_f16(3.0); hidden], "ar hidden={hidden}");
        assert_eq!(h_res, vec![f32_to_f16(3.5); hidden], "res hidden={hidden}");
        assert_eq!(
            h_tap,
            vec![f32_to_bf16(3.5); hidden],
            "tap_bf16 hidden={hidden}"
        );
        for &v in &h_out {
            assert!(
                (f16_to_f32(v) - 1.25).abs() < 2e-3,
                "norm out mismatch at hidden={hidden}: got {}",
                f16_to_f32(v)
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn verify_tp2_dma_crossover_and_graph_replay(s0: &CudaStream, s1: &CudaStream) {
    // Exercise both forced mapped-host staging (`allow_p2p = false`) and available P2P (`allow_p2p = true`)
    // across `< 64 KiB`, `== 64 KiB`, and `> 64 KiB` transfers, including back-to-back even-epoch
    // large `gather_vocab_f16` followed immediately by `>= 64 KiB` DMA `all_reduce_f16` and
    // repeated CUDA Graph replays with mutating payloads.
    for allow_p2p in [false, true] {
        let comm = Tp2Collectives::new_for_tp_with_p2p(2, 256 * 1024, allow_p2p).unwrap();

        // 1. Threshold sweep: just below (`65,520 B`), at (`65,536 B`), and above (`98,304 B`) 64 KiB.
        for n_f16 in [32_760usize, 32_768, 49_152] {
            let n_f32 = n_f16 / 2;
            let d0_f16 = DeviceBuffer::alloc(0, n_f16 * 2).unwrap();
            let d1_f16 = DeviceBuffer::alloc(1, n_f16 * 2).unwrap();
            let d0_bf16 = DeviceBuffer::alloc(0, n_f16 * 2).unwrap();
            let d1_bf16 = DeviceBuffer::alloc(1, n_f16 * 2).unwrap();
            let d0_f32 = DeviceBuffer::alloc(0, n_f32 * 4).unwrap();
            let d1_f32 = DeviceBuffer::alloc(1, n_f32 * 4).unwrap();

            d0_f16
                .copy_from_host(&vec![f32_to_f16(1.25); n_f16], s0)
                .unwrap();
            d1_f16
                .copy_from_host(&vec![f32_to_f16(2.75); n_f16], s1)
                .unwrap();
            d0_bf16
                .copy_from_host(&vec![f32_to_bf16(0.5); n_f16], s0)
                .unwrap();
            d1_bf16
                .copy_from_host(&vec![f32_to_bf16(1.5); n_f16], s1)
                .unwrap();
            d0_f32.copy_from_host(&vec![3.25f32; n_f32], s0).unwrap();
            d1_f32.copy_from_host(&vec![4.75f32; n_f32], s1).unwrap();
            s0.synchronize().unwrap();
            s1.synchronize().unwrap();

            std::thread::scope(|scope| {
                let (a0_f16, a0_bf16, a0_f32) = (d0_f16.addr(), d0_bf16.addr(), d0_f32.addr());
                let (a1_f16, a1_bf16, a1_f32) = (d1_f16.addr(), d1_bf16.addr(), d1_f32.addr());
                let c = &comm;
                let t0 = scope.spawn(move || {
                    let _dev = CudaDevice::set_current(0).unwrap();
                    let ex0 = s0.exec(0, 0);
                    c.all_reduce_f16(&ex0, a0_f16, a0_f16, 0, n_f16 as i32)
                        .unwrap();
                    c.all_reduce_bf16(&ex0, a0_bf16, a0_bf16, 0, n_f16 as i32)
                        .unwrap();
                    c.all_reduce_f32(&ex0, a0_f32, a0_f32, 0, n_f32 as i32)
                        .unwrap();
                    s0.synchronize().unwrap();
                });
                let t1 = scope.spawn(move || {
                    let _dev = CudaDevice::set_current(1).unwrap();
                    let ex1 = s1.exec(0, 0);
                    c.all_reduce_f16(&ex1, a1_f16, a1_f16, 1, n_f16 as i32)
                        .unwrap();
                    c.all_reduce_bf16(&ex1, a1_bf16, a1_bf16, 1, n_f16 as i32)
                        .unwrap();
                    c.all_reduce_f32(&ex1, a1_f32, a1_f32, 1, n_f32 as i32)
                        .unwrap();
                    s1.synchronize().unwrap();
                });
                t0.join().unwrap();
                t1.join().unwrap();
            });

            let mut h0_f16 = vec![0u16; n_f16];
            let mut h1_f16 = vec![0u16; n_f16];
            let mut h0_bf16 = vec![0u16; n_f16];
            let mut h0_f32 = vec![0.0f32; n_f32];
            d0_f16.copy_to_host(&mut h0_f16, s0).unwrap();
            d1_f16.copy_to_host(&mut h1_f16, s1).unwrap();
            d0_bf16.copy_to_host(&mut h0_bf16, s0).unwrap();
            d0_f32.copy_to_host(&mut h0_f32, s0).unwrap();
            s0.synchronize().unwrap();
            s1.synchronize().unwrap();
            assert_eq!(h0_f16, vec![f32_to_f16(4.0); n_f16]);
            assert_eq!(h1_f16, vec![f32_to_f16(4.0); n_f16]);
            assert_eq!(h0_bf16, vec![f32_to_bf16(2.0); n_f16]);
            assert_eq!(h0_f32, vec![8.0f32; n_f32]);
        }

        // 2. Fused residual+GemmaRMSNorm and HC-combine across `< 64 KiB` (`M=7`), `== 64 KiB` (`M=8`),
        //    and `> 64 KiB` (`M=12`) at `hidden = 4096`.
        let hidden = 4096usize;
        for num_tokens in [7usize, 8, 12] {
            let n = num_tokens * hidden;
            let d0_in = DeviceBuffer::alloc(0, n * 2).unwrap();
            let d1_in = DeviceBuffer::alloc(1, n * 2).unwrap();
            let d0_res = DeviceBuffer::alloc(0, n * 2).unwrap();
            let d1_res = DeviceBuffer::alloc(1, n * 2).unwrap();
            let d0_w = DeviceBuffer::alloc(0, hidden * 2).unwrap();
            let d1_w = DeviceBuffer::alloc(1, hidden * 2).unwrap();
            let d0_out = DeviceBuffer::alloc(0, n * 2).unwrap();
            let d1_out = DeviceBuffer::alloc(1, n * 2).unwrap();
            let d0_hc = DeviceBuffer::alloc(0, n * 2).unwrap();
            let d1_hc = DeviceBuffer::alloc(1, n * 2).unwrap();
            let d0_inj = DeviceBuffer::alloc(0, num_tokens * 4).unwrap();
            let d1_inj = DeviceBuffer::alloc(1, num_tokens * 4).unwrap();
            d0_in.copy_from_host(&vec![f32_to_f16(1.0); n], s0).unwrap();
            d1_in.copy_from_host(&vec![f32_to_f16(2.0); n], s1).unwrap();
            d0_res
                .copy_from_host(&vec![f32_to_f16(0.5); n], s0)
                .unwrap();
            d1_res
                .copy_from_host(&vec![f32_to_f16(0.5); n], s1)
                .unwrap();
            d0_w.copy_from_host(&vec![f32_to_f16(0.25); hidden], s0)
                .unwrap();
            d1_w.copy_from_host(&vec![f32_to_f16(0.25); hidden], s1)
                .unwrap();
            d0_inj
                .copy_from_host(&vec![1.0f32; num_tokens], s0)
                .unwrap();
            d1_inj
                .copy_from_host(&vec![1.0f32; num_tokens], s1)
                .unwrap();
            s0.synchronize().unwrap();
            s1.synchronize().unwrap();

            std::thread::scope(|scope| {
                let (in0, res0, w0, out0, hc0, inj0) = (
                    d0_in.addr(),
                    d0_res.addr(),
                    d0_w.addr(),
                    d0_out.addr(),
                    d0_hc.addr(),
                    d0_inj.addr(),
                );
                let (in1, res1, w1, out1, hc1, inj1) = (
                    d1_in.addr(),
                    d1_res.addr(),
                    d1_w.addr(),
                    d1_out.addr(),
                    d1_hc.addr(),
                    d1_inj.addr(),
                );
                let c = &comm;
                let t0 = scope.spawn(move || {
                    let _dev = CudaDevice::set_current(0).unwrap();
                    let ex0 = s0.exec(0, 0);
                    c.all_reduce_residual_gemma_rms_norm_f16(
                        &ex0,
                        in0,
                        0,
                        res0,
                        w0,
                        out0,
                        0,
                        0,
                        num_tokens as i32,
                        hidden as i32,
                        1e-6,
                        0,
                        0,
                    )
                    .unwrap();
                    c.all_reduce_hc_combine_tap_f16(
                        &ex0,
                        res0,
                        in0,
                        0,
                        inj0,
                        0,
                        hc0,
                        0,
                        0,
                        0,
                        num_tokens as i32,
                        1,
                        hidden as i32,
                        0,
                        0,
                    )
                    .unwrap();
                    s0.synchronize().unwrap();
                });
                let t1 = scope.spawn(move || {
                    let _dev = CudaDevice::set_current(1).unwrap();
                    let ex1 = s1.exec(0, 0);
                    c.all_reduce_residual_gemma_rms_norm_f16(
                        &ex1,
                        in1,
                        0,
                        res1,
                        w1,
                        out1,
                        0,
                        1,
                        num_tokens as i32,
                        hidden as i32,
                        1e-6,
                        0,
                        0,
                    )
                    .unwrap();
                    c.all_reduce_hc_combine_tap_f16(
                        &ex1,
                        res1,
                        in1,
                        0,
                        inj1,
                        0,
                        hc1,
                        0,
                        0,
                        1,
                        num_tokens as i32,
                        1,
                        hidden as i32,
                        0,
                        0,
                    )
                    .unwrap();
                    s1.synchronize().unwrap();
                });
                t0.join().unwrap();
                t1.join().unwrap();
            });

            let mut h_res = vec![0u16; n];
            let mut h_hc = vec![0u16; n];
            d0_res.copy_to_host(&mut h_res, s0).unwrap();
            d0_hc.copy_to_host(&mut h_hc, s0).unwrap();
            s0.synchronize().unwrap();
            assert_eq!(h_res, vec![f32_to_f16(3.5); n]);
            assert_eq!(h_hc, vec![f32_to_f16(6.5); n]);
        }

        // 3. Back-to-back odd-epoch + even-epoch large `gather_vocab_f16` (`81,920 B >= 64 KiB`)
        //    followed immediately by `>= 64 KiB` DMA `all_reduce_f16` and `< 64 KiB` `all_reduce_f16`,
        //    both eager and across repeated CUDA Graph replays with changing payloads.
        let n_vocab_rows = 2usize;
        let local_vocab = 20_480usize; // 2 * 20,480 * 2B = 81,920 B (>= 64 KiB)
        let n_vocab_elems = n_vocab_rows * local_vocab;
        let n_large = 40_960usize; // 81,920 B (>= 64 KiB DMA)
        let n_small = 32_760usize; // 65,520 B (< 64 KiB SM WB)

        let d0_vin = DeviceBuffer::alloc(0, n_vocab_elems * 2).unwrap();
        let d1_vin = DeviceBuffer::alloc(1, n_vocab_elems * 2).unwrap();
        let d0_vout_odd = DeviceBuffer::alloc(0, n_vocab_elems * 4).unwrap();
        let d0_vout_even = DeviceBuffer::alloc(0, n_vocab_elems * 4).unwrap();
        let d0_large = DeviceBuffer::alloc(0, n_large * 2).unwrap();
        let d1_large = DeviceBuffer::alloc(1, n_large * 2).unwrap();
        let d0_small = DeviceBuffer::alloc(0, n_small * 2).unwrap();
        let d1_small = DeviceBuffer::alloc(1, n_small * 2).unwrap();

        let (vin0, vin1, vodd0, veven0) = (
            d0_vin.addr(),
            d1_vin.addr(),
            d0_vout_odd.addr(),
            d0_vout_even.addr(),
        );
        let (lg0, lg1, sm0, sm1) = (
            d0_large.addr(),
            d1_large.addr(),
            d0_small.addr(),
            d1_small.addr(),
        );

        let (g0, g1) = std::thread::scope(|scope| {
            let c = &comm;
            let t0 = scope.spawn(move || {
                let _dev = CudaDevice::set_current(0).unwrap();
                let ex0 = s0.exec(0, 0);
                super::CudaGraph::capture(s0, || {
                    c.gather_vocab_f16(
                        &ex0,
                        vin0,
                        vodd0,
                        0,
                        n_vocab_rows as i32,
                        local_vocab as i32,
                    )?;
                    c.gather_vocab_f16(
                        &ex0,
                        vin0,
                        veven0,
                        0,
                        n_vocab_rows as i32,
                        local_vocab as i32,
                    )?;
                    c.all_reduce_f16(&ex0, lg0, lg0, 0, n_large as i32)?;
                    c.all_reduce_f16(&ex0, sm0, sm0, 0, n_small as i32)
                })
                .unwrap()
            });
            let t1 = scope.spawn(move || {
                let _dev = CudaDevice::set_current(1).unwrap();
                let ex1 = s1.exec(0, 0);
                super::CudaGraph::capture(s1, || {
                    c.gather_vocab_f16(&ex1, vin1, 0, 1, n_vocab_rows as i32, local_vocab as i32)?;
                    c.gather_vocab_f16(&ex1, vin1, 0, 1, n_vocab_rows as i32, local_vocab as i32)?;
                    c.all_reduce_f16(&ex1, lg1, lg1, 1, n_large as i32)?;
                    c.all_reduce_f16(&ex1, sm1, sm1, 1, n_small as i32)
                })
                .unwrap()
            });
            (t0.join().unwrap(), t1.join().unwrap())
        });

        for replay in 1..=3usize {
            let v0_val = (replay as f32) * 1.5;
            let v1_val = (replay as f32) * 2.5;
            let lg0_val = (replay as f32) * 0.75;
            let lg1_val = (replay as f32) * 1.25;
            let sm0_val = (replay as f32) * 2.0;
            let sm1_val = (replay as f32) * 3.0;

            d0_vin
                .copy_from_host(&vec![f32_to_f16(v0_val); n_vocab_elems], s0)
                .unwrap();
            d1_vin
                .copy_from_host(&vec![f32_to_f16(v1_val); n_vocab_elems], s1)
                .unwrap();
            d0_large
                .copy_from_host(&vec![f32_to_f16(lg0_val); n_large], s0)
                .unwrap();
            d1_large
                .copy_from_host(&vec![f32_to_f16(lg1_val); n_large], s1)
                .unwrap();
            d0_small
                .copy_from_host(&vec![f32_to_f16(sm0_val); n_small], s0)
                .unwrap();
            d1_small
                .copy_from_host(&vec![f32_to_f16(sm1_val); n_small], s1)
                .unwrap();
            s0.synchronize().unwrap();
            s1.synchronize().unwrap();

            std::thread::scope(|scope| {
                let g0_ref = &g0;
                let g1_ref = &g1;
                let t0 = scope.spawn(move || {
                    let _dev = CudaDevice::set_current(0).unwrap();
                    g0_ref.launch(s0).unwrap();
                    s0.synchronize().unwrap();
                });
                let t1 = scope.spawn(move || {
                    let _dev = CudaDevice::set_current(1).unwrap();
                    g1_ref.launch(s1).unwrap();
                    s1.synchronize().unwrap();
                });
                t0.join().unwrap();
                t1.join().unwrap();
            });

            let mut h_vodd = vec![0u16; n_vocab_elems * 2];
            let mut h_veven = vec![0u16; n_vocab_elems * 2];
            let mut h_lg0 = vec![0u16; n_large];
            let mut h_lg1 = vec![0u16; n_large];
            let mut h_sm0 = vec![0u16; n_small];
            let mut h_sm1 = vec![0u16; n_small];
            d0_vout_odd.copy_to_host(&mut h_vodd, s0).unwrap();
            d0_vout_even.copy_to_host(&mut h_veven, s0).unwrap();
            d0_large.copy_to_host(&mut h_lg0, s0).unwrap();
            d1_large.copy_to_host(&mut h_lg1, s1).unwrap();
            d0_small.copy_to_host(&mut h_sm0, s0).unwrap();
            d1_small.copy_to_host(&mut h_sm1, s1).unwrap();
            s0.synchronize().unwrap();
            s1.synchronize().unwrap();

            for row in 0..n_vocab_rows {
                let base = row * local_vocab * 2;
                assert_eq!(
                    &h_vodd[base..base + local_vocab],
                    &vec![f32_to_f16(v0_val); local_vocab][..]
                );
                assert_eq!(
                    &h_vodd[base + local_vocab..base + 2 * local_vocab],
                    &vec![f32_to_f16(v1_val); local_vocab][..]
                );
                assert_eq!(
                    &h_veven[base..base + local_vocab],
                    &vec![f32_to_f16(v0_val); local_vocab][..]
                );
                assert_eq!(
                    &h_veven[base + local_vocab..base + 2 * local_vocab],
                    &vec![f32_to_f16(v1_val); local_vocab][..]
                );
            }
            assert_eq!(h_lg0, vec![f32_to_f16(lg0_val + lg1_val); n_large]);
            assert_eq!(h_lg1, vec![f32_to_f16(lg0_val + lg1_val); n_large]);
            assert_eq!(h_sm0, vec![f32_to_f16(sm0_val + sm1_val); n_small]);
            assert_eq!(h_sm1, vec![f32_to_f16(sm0_val + sm1_val); n_small]);
        }
    }
}

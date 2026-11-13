//! Bounded page16 D256 target prefill, GPU0. No model or external fixture required.
#![cfg(feature = "cuda")]
use std::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaGraph, CudaStream, DeviceBuffer, check_rc};
use w34_kernels::ffi;

fn upload<T: Copy>(s: &CudaStream, x: &[T]) -> DeviceBuffer {
    let b = DeviceBuffer::alloc(0, std::mem::size_of_val(x)).unwrap();
    b.copy_from_host(x, s).unwrap();
    b
}
fn half(x: u16) -> f32 {
    let sign = if x & 0x8000 == 0 { 1.0 } else { -1.0 };
    let e = ((x >> 10) & 31) as i32;
    sign * if e == 0 {
        (x & 1023) as f32 * 2f32.powi(-24)
    } else {
        (1.0 + (x & 1023) as f32 / 1024.0) * 2f32.powi(e - 15)
    }
}
#[test]
fn selected_device_execution_plan_warmup() {
    w34_host::cuda::prepare_execution_plan(0).unwrap();
}

#[test]
fn prefill_reference_graph_and_fallback_canaries() {
    let _dev = CudaDevice::set_current(0).unwrap();
    let s = CudaStream::new(0).unwrap();
    // Uniform scores give an independent CPU mean reference; V varies by physical
    // page, row and column. Unused pages are distinct, not copies of valid pages.
    let q = upload(&s, &vec![0u16; 306 * 2 * 256]);
    let k = upload(&s, &vec![0u16; 24 * 16 * 256]);
    let values: Vec<u16> = (0..24 * 16 * 256)
        .map(|i| {
            if (i / 256 + i % 256) % 3 == 0 {
                0x3c00
            } else {
                0xbc00
            }
        })
        .collect();
    let v = upload(&s, &values);
    let out = upload(&s, &vec![0x5500u16; 306 * 2 * 256]);
    let starts = upload(&s, &[0i32, 306]);
    let ctx = upload(&s, &[0i32]);
    let seq = upload(&s, &[306i32]);
    let mut permutation: Vec<i32> = (0..24).map(|i| (i * 7 + 5) % 24).collect();
    let pages = upload(&s, &permutation);
    let ws = upload(&s, &vec![0xa5u8; 306 * 2 * 4 + 32]);
    let masks = upload(&s, &vec![1u8; 306]);
    let inactive = upload(&s, &[0u8]);
    let layout = ffi::W34KvLayerLayout {
        k_base: k.addr(),
        v_base: v.addr(),
        k_page_stride_bytes: 8192,
        v_page_stride_bytes: 8192,
        k_head_stride_bytes: 512,
        v_head_stride_bytes: 512,
        k_token_stride_bytes: 512,
        v_token_stride_bytes: 512,
        namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 0,
        local_kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        k_format: ffi::W34_KV_FORMAT_FP16,
        v_format: ffi::W34_KV_FORMAT_FP16,
        k_transform: 0,
        v_transform: 0,
        reserved: 0,
    };
    let mut b = ffi::W34AttentionBatch {
        query_start_loc: starts.addr() as *const i32,
        context_lens_before: ctx.addr() as *const i32,
        seq_lens_after: seq.addr() as *const i32,
        block_tables: pages.addr() as *const i32,
        write_slot_mapping: std::ptr::null(),
        valid_token_mask: std::ptr::null(),
        valid_req_mask: std::ptr::null(),
        num_reqs: 1,
        total_tokens: 306,
        max_blocks_per_req: 24,
        max_q_len: 306,
        max_seq_len: 384,
        mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: 0.0625,
        reserved: 0,
    };
    let invoke = |b: &ffi::W34AttentionBatch, size| {
        check_rc(
            // SAFETY: buffers and metadata remain live on this stream; workspace
            // includes the advertised size plus external canaries.
            unsafe {
                ffi::w34_paged_attention_f16(
                    &s.exec(ws.addr() + 16, size),
                    &layout,
                    b,
                    q.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    2,
                )
            },
            "paged FA2",
        )
    };
    let verify = |n: usize, context: usize, perm: &[i32]| {
        let mut got = vec![0u16; 306 * 2 * 256];
        out.copy_to_host(&mut got, &s).unwrap();
        s.synchronize().unwrap();
        for row in 0..n {
            for col in 0..256 {
                let end = context + row + 1;
                let expected: f32 = (0..end)
                    .map(|t| half(values[(perm[t / 16] as usize * 16 + t % 16) * 256 + col]))
                    .sum::<f32>()
                    / end as f32;
                for h in 0..2 {
                    assert!(
                        (half(got[(row * 2 + h) * 256 + col]) - expected).abs() < 0.001,
                        "row={row} col={col}"
                    );
                }
            }
        }
    };
    for (n, context) in [
        (1, 0),
        (16, 15),
        (17, 17),
        (50, 63),
        (64, 65),
        (65, 16),
        (128, 17),
        (306, 0),
    ] {
        starts.copy_from_host(&[0, n], &s).unwrap();
        ctx.copy_from_host(&[context], &s).unwrap();
        seq.copy_from_host(&[context + n], &s).unwrap();
        b.total_tokens = n;
        b.max_q_len = n;
        invoke(&b, 306 * 2 * 4).unwrap();
        verify(n as usize, context as usize, &permutation);
    }
    let mut guard = vec![0u8; 306 * 2 * 4 + 32];
    ws.copy_to_host(&mut guard, &s).unwrap();
    s.synchronize().unwrap();
    assert!(
        guard[..16]
            .iter()
            .chain(guard[guard.len() - 16..].iter())
            .all(|&x| x == 0xa5)
    );
    assert!(
        guard[16..guard.len() - 16].iter().any(|&x| x != 0xa5),
        "eligible path must write LSE"
    );
    // Capture once with loose bounds, then change device lengths, query offsets
    // and the page table without rebuilding the graph.
    let graph = CudaGraph::capture(&s, || invoke(&b, 306 * 2 * 4)).unwrap();
    permutation.reverse();
    pages.copy_from_host(&permutation, &s).unwrap();
    starts.copy_from_host(&[0, 65], &s).unwrap();
    ctx.copy_from_host(&[31], &s).unwrap();
    seq.copy_from_host(&[96], &s).unwrap();
    graph.launch(&s).unwrap();
    verify(65, 31, &permutation);
    b.total_tokens = 65;
    b.max_q_len = 65;
    // Insufficient LSE workspace must fall back without touching any scratch.
    ws.copy_from_host(&vec![0xa5u8; 306 * 2 * 4 + 32], &s)
        .unwrap();
    invoke(&b, 65 * 2 * 4 - 1).unwrap();
    verify(65, 31, &permutation);
    let mut canary = vec![0u8; 306 * 2 * 4 + 32];
    ws.copy_to_host(&mut canary, &s).unwrap();
    s.synchronize().unwrap();
    assert!(canary.iter().all(|&x| x == 0xa5));
    for ptr in [0, ws.addr() + 17] {
        assert_eq!(
            // SAFETY: input/output buffers cover the descriptor; intentionally
            // unusable scratch must be rejected before any workspace access.
            unsafe {
                ffi::w34_paged_attention_f16(
                    &s.exec(ptr, 306 * 2 * 4),
                    &layout,
                    &b,
                    q.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    2,
                )
            },
            0
        );
        verify(65, 31, &permutation);
    }
    ws.copy_to_host(&mut canary, &s).unwrap();
    s.synchronize().unwrap();
    assert!(canary.iter().all(|&x| x == 0xa5));
    b.valid_token_mask = masks.addr() as *const u8;
    invoke(&b, 306 * 2 * 4).unwrap();
    verify(65, 31, &permutation);
    let mut token_mask = vec![1u8; 306];
    token_mask[3] = 0;
    masks.copy_from_host(&token_mask, &s).unwrap();
    out.copy_from_host(&vec![0x5500u16; 306 * 2 * 256], &s)
        .unwrap();
    invoke(&b, 306 * 2 * 4).unwrap();
    let mut masked = vec![0u16; 306 * 2 * 256];
    out.copy_to_host(&mut masked, &s).unwrap();
    s.synchronize().unwrap();
    assert!(
        masked[3 * 2 * 256..4 * 2 * 256]
            .iter()
            .all(|&x| x == 0x5500)
    );
    // Inactive request has an unusable page table and must preserve output.
    b.valid_req_mask = inactive.addr() as *const u8;
    pages.copy_from_host(&[-1i32; 24], &s).unwrap();
    out.copy_from_host(&vec![0x5500u16; 306 * 2 * 256], &s)
        .unwrap();
    invoke(&b, 306 * 2 * 4).unwrap();
    let mut got = vec![0u16; 306 * 2 * 256];
    out.copy_to_host(&mut got, &s).unwrap();
    s.synchronize().unwrap();
    assert!(got.iter().all(|&x| x == 0x5500));
}

#[test]
fn multi_request_mixed_decode_and_prefill_fa2() {
    let _dev = CudaDevice::set_current(0).unwrap();
    let s = CudaStream::new(0).unwrap();
    w34_host::cuda::prepare_execution_plan(0).unwrap();

    // 48 physical pages of 16 tokens x 256 head_dim: Req 0 uses 13 pages (200 tok: 199 ctx + 1 decode),
    // Req 1 uses 22 pages (337 tok: 47 ctx + 290 prefill).
    let total_q = 1 + 290;
    let max_q = 290;
    let max_blocks = 24;
    let q = upload(&s, &vec![0u16; total_q * 4 * 256]);
    let k = upload(&s, &vec![0u16; 48 * 16 * 256]);
    let values: Vec<u16> = (0..48 * 16 * 256)
        .map(|i| {
            if (i / 256 + (i % 256) * 2) % 5 == 0 {
                0x3c00
            } else {
                0xbc00
            }
        })
        .collect();
    let v = upload(&s, &values);
    let out = upload(&s, &vec![0x5500u16; total_q * 4 * 256]);
    let starts = upload(&s, &[0i32, 1, 291]);
    let ctx = upload(&s, &[199i32, 47]);
    let seq = upload(&s, &[200i32, 337]);

    let perm0: Vec<i32> = (0..max_blocks).map(|i| (i * 5 + 3) % 24).collect();
    let perm1: Vec<i32> = (0..max_blocks).map(|i| 24 + ((i * 7 + 1) % 24)).collect();
    let mut table = Vec::with_capacity((2 * max_blocks) as usize);
    table.extend_from_slice(&perm0);
    table.extend_from_slice(&perm1);
    let pages = upload(&s, &table);

    let lse_bytes = 2 * 4 * max_q * 4;
    let ws = upload(&s, &vec![0xa5u8; lse_bytes + 32]);
    let layout = ffi::W34KvLayerLayout {
        k_base: k.addr(),
        v_base: v.addr(),
        k_page_stride_bytes: 8192,
        v_page_stride_bytes: 8192,
        k_head_stride_bytes: 512,
        v_head_stride_bytes: 512,
        k_token_stride_bytes: 512,
        v_token_stride_bytes: 512,
        namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 0,
        local_kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        k_format: ffi::W34_KV_FORMAT_FP16,
        v_format: ffi::W34_KV_FORMAT_FP16,
        k_transform: 0,
        v_transform: 0,
        reserved: 0,
    };
    let b = ffi::W34AttentionBatch {
        query_start_loc: starts.addr() as *const i32,
        context_lens_before: ctx.addr() as *const i32,
        seq_lens_after: seq.addr() as *const i32,
        block_tables: pages.addr() as *const i32,
        write_slot_mapping: std::ptr::null(),
        valid_token_mask: std::ptr::null(),
        valid_req_mask: std::ptr::null(),
        num_reqs: 2,
        total_tokens: total_q as i32,
        max_blocks_per_req: max_blocks,
        max_q_len: max_q as i32,
        max_seq_len: 337,
        mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: 0.0625,
        reserved: 0,
    };
    check_rc(
        // SAFETY: allocations cover the descriptors and remain live through stream synchronization.
        unsafe {
            ffi::w34_paged_attention_f16(
                &s.exec(ws.addr() + 16, lse_bytes),
                &layout,
                &b,
                q.addr() as *const c_void,
                out.addr() as *mut c_void,
                4,
            )
        },
        "multi-req paged FA2",
    )
    .unwrap();

    let mut guard = vec![0u8; lse_bytes + 32];
    let mut got = vec![0u16; total_q * 4 * 256];
    ws.copy_to_host(&mut guard, &s).unwrap();
    out.copy_to_host(&mut got, &s).unwrap();
    s.synchronize().unwrap();

    assert!(
        guard[..16]
            .iter()
            .chain(guard[guard.len() - 16..].iter())
            .all(|&x| x == 0xa5),
        "canaries around multi-req LSE workspace must stay intact"
    );
    assert!(
        guard[16..guard.len() - 16].iter().any(|&x| x != 0xa5),
        "multi-req mixed decode+prefill must take FA2 path and write LSE"
    );

    for (q_base, q_len, ctx_before, perm) in [
        (0usize, 1usize, 199usize, &perm0[..]),
        (1, 290, 47, &perm1[..]),
    ] {
        for row in 0..q_len {
            let end = ctx_before + row + 1;
            for col in 0..256 {
                let expected: f32 = (0..end)
                    .map(|t| half(values[(perm[t / 16] as usize * 16 + t % 16) * 256 + col]))
                    .sum::<f32>()
                    / end as f32;
                for h in 0..4 {
                    let actual = half(got[((q_base + row) * 4 + h) * 256 + col]);
                    assert!(
                        (actual - expected).abs() < 0.002,
                        "q_base={q_base} row={row} h={h} col={col}: got={actual} expected={expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn single_request_long_context_graph_matches_eager() {
    let _dev = CudaDevice::set_current(0).unwrap();
    let s = CudaStream::new(0).unwrap();
    w34_host::cuda::prepare_execution_plan(0).unwrap();

    // Warm coding-agent turn shape: q_len = 97 (<= 256), context = 2400 -> seq_after = 2497 (>= 2048).
    let n_pages = 160usize;
    let q_len = 97usize;
    let ctx_before = 2400usize;
    let seq_after = ctx_before + q_len;
    let q = upload(&s, &vec![0u16; q_len * 4 * 256]);
    let k = upload(&s, &vec![0u16; n_pages * 16 * 256]);
    let values: Vec<u16> = (0..n_pages * 16 * 256)
        .map(|i| {
            if (i / 256 + i % 17) % 4 == 0 {
                0x3c00
            } else {
                0xbc00
            }
        })
        .collect();
    let v = upload(&s, &values);
    let out = upload(&s, &vec![0x5500u16; q_len * 4 * 256]);
    let starts = upload(&s, &[0i32, q_len as i32]);
    let ctx = upload(&s, &[ctx_before as i32]);
    let seq = upload(&s, &[seq_after as i32]);
    let perm: Vec<i32> = (0..n_pages as i32)
        .map(|i| (i * 13 + 7) % (n_pages as i32))
        .collect();
    let pages = upload(&s, &perm);

    // Large scratch must not change the arithmetic between eager and graph execution.
    let ws_size = 2 * 1024 * 1024;
    let ws = upload(&s, &vec![0xa5u8; ws_size]);
    let layout = ffi::W34KvLayerLayout {
        k_base: k.addr(),
        v_base: v.addr(),
        k_page_stride_bytes: 8192,
        v_page_stride_bytes: 8192,
        k_head_stride_bytes: 512,
        v_head_stride_bytes: 512,
        k_token_stride_bytes: 512,
        v_token_stride_bytes: 512,
        namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 0,
        local_kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        k_format: ffi::W34_KV_FORMAT_FP16,
        v_format: ffi::W34_KV_FORMAT_FP16,
        k_transform: 0,
        v_transform: 0,
        reserved: 0,
    };
    let b = ffi::W34AttentionBatch {
        query_start_loc: starts.addr() as *const i32,
        context_lens_before: ctx.addr() as *const i32,
        seq_lens_after: seq.addr() as *const i32,
        block_tables: pages.addr() as *const i32,
        write_slot_mapping: std::ptr::null(),
        valid_token_mask: std::ptr::null(),
        valid_req_mask: std::ptr::null(),
        num_reqs: 1,
        total_tokens: q_len as i32,
        max_blocks_per_req: n_pages as i32,
        max_q_len: q_len as i32,
        max_seq_len: seq_after as i32,
        mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: 0.0625,
        reserved: 0,
    };
    check_rc(
        // SAFETY: allocations cover the descriptors and remain live through stream synchronization.
        unsafe {
            ffi::w34_paged_attention_f16(
                &s.exec(ws.addr(), ws_size),
                &layout,
                &b,
                q.addr() as *const c_void,
                out.addr() as *mut c_void,
                4,
            )
        },
        "long-context paged FA2",
    )
    .unwrap();

    let mut got = vec![0u16; q_len * 4 * 256];
    out.copy_to_host(&mut got, &s).unwrap();
    s.synchronize().unwrap();

    for row in [0usize, 15, 63, 64, 96] {
        let end = ctx_before + row + 1;
        for col in 0..256 {
            let expected: f32 = (0..end)
                .map(|t| half(values[(perm[t / 16] as usize * 16 + t % 16) * 256 + col]))
                .sum::<f32>()
                / end as f32;
            for h in 0..4 {
                let actual = half(got[(row * 4 + h) * 256 + col]);
                assert!(
                    (actual - expected).abs() < 0.003,
                    "row={row} h={h} col={col}: got={actual} expected={expected}"
                );
            }
        }
    }

    // Nonuniform scores expose reduction/rounding differences hidden by Q=K=0.
    let pattern = |i: usize| [0x3800u16, 0xb400, 0x3000, 0x3a00, 0xb800][(i * 7 + i / 257) % 5];
    q.copy_from_host(&(0..q_len * 4 * 256).map(pattern).collect::<Vec<_>>(), &s)
        .unwrap();
    k.copy_from_host(
        &(0..n_pages * 16 * 256).map(pattern).collect::<Vec<_>>(),
        &s,
    )
    .unwrap();
    let invoke = || {
        check_rc(
            // SAFETY: captured allocations remain live; replay lengths stay within their bounds.
            unsafe {
                ffi::w34_paged_attention_f16(
                    &s.exec(ws.addr(), ws_size),
                    &layout,
                    &b,
                    q.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    4,
                )
            },
            "long-context graph/eager FA2",
        )
    };
    invoke().unwrap();
    out.copy_to_host(&mut got, &s).unwrap();
    s.synchronize().unwrap();
    let graph = CudaGraph::capture(&s, invoke).unwrap();
    graph.launch(&s).unwrap();
    let mut replay = vec![0u16; got.len()];
    out.copy_to_host(&mut replay, &s).unwrap();
    s.synchronize().unwrap();
    assert_eq!(got, replay, "graph and eager must be bit-exact");

    // Replay shorter device lengths without recapture; inactive output stays untouched.
    starts.copy_from_host(&[0i32, 65], &s).unwrap();
    seq.copy_from_host(&[(ctx_before + 65) as i32], &s).unwrap();
    out.copy_from_host(&vec![0x5500u16; got.len()], &s).unwrap();
    graph.launch(&s).unwrap();
    out.copy_to_host(&mut replay, &s).unwrap();
    s.synchronize().unwrap();
    assert!(replay[65 * 4 * 256..].iter().all(|&x| x == 0x5500));
    invoke().unwrap();
    out.copy_to_host(&mut got, &s).unwrap();
    s.synchronize().unwrap();
    assert_eq!(got, replay, "shortened replay must match eager");
}

#[test]
fn tight_two_page_partial_tile_does_not_read_past_block_table() {
    let _dev = CudaDevice::set_current(0).unwrap();
    let s = CudaStream::new(0).unwrap();
    // Exact 32-token (2-page) sequence inside a 64-token KV tile: `partial_block_size = 32`,
    // `max_blocks_per_req = 2`, and `pages` is allocated to exact `2 * sizeof(i32) = 8` bytes.
    let q = upload(&s, &vec![0u16; 32 * 2 * 256]);
    let k = upload(&s, &vec![0u16; 2 * 16 * 256]);
    let values: Vec<u16> = (0..2 * 16 * 256)
        .map(|i| {
            if (i / 256 + i % 256) % 2 == 0 {
                0x3c00
            } else {
                0xbc00
            }
        })
        .collect();
    let v = upload(&s, &values);
    let out = upload(&s, &vec![0u16; 32 * 2 * 256]);
    let starts = upload(&s, &[0i32, 32]);
    let ctx = upload(&s, &[0i32]);
    let seq = upload(&s, &[32i32]);
    // Place the 2-entry `block_tables` (`8 B`) at the end of a 16-byte allocation (`+8 B` offset,
    // 4B-aligned) so `block_table[2]` is immediately out of bounds at byte `+16`.
    let pages = upload(&s, &[0i32, 0i32, 1i32, 0i32]);
    let ws = upload(&s, &vec![0u8; 32 * 2 * 4]);
    let layout = ffi::W34KvLayerLayout {
        k_base: k.addr(),
        v_base: v.addr(),
        k_page_stride_bytes: 8192,
        v_page_stride_bytes: 8192,
        k_head_stride_bytes: 512,
        v_head_stride_bytes: 512,
        k_token_stride_bytes: 512,
        v_token_stride_bytes: 512,
        namespace_id: ffi::W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 0,
        local_kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        k_format: ffi::W34_KV_FORMAT_FP16,
        v_format: ffi::W34_KV_FORMAT_FP16,
        k_transform: 0,
        v_transform: 0,
        reserved: 0,
    };
    let batch = ffi::W34AttentionBatch {
        query_start_loc: starts.addr() as *const i32,
        context_lens_before: ctx.addr() as *const i32,
        seq_lens_after: seq.addr() as *const i32,
        block_tables: (pages.addr() + 8) as *const i32,
        write_slot_mapping: std::ptr::null(),
        valid_token_mask: std::ptr::null(),
        valid_req_mask: std::ptr::null(),
        num_reqs: 1,
        total_tokens: 32,
        max_blocks_per_req: 2,
        max_q_len: 32,
        max_seq_len: 32,
        mask_mode: ffi::W34_ATTN_MASK_CAUSAL,
        sliding_window: 0,
        sm_scale: 0.0625,
        reserved: 0,
    };
    check_rc(
        // SAFETY: all buffers are live on device 0; `pages` has exact length `2` matching `max_blocks_per_req`.
        unsafe {
            ffi::w34_paged_attention_f16(
                &s.exec(ws.addr(), 32 * 2 * 4),
                &layout,
                &batch,
                q.addr() as *const c_void,
                out.addr() as *mut c_void,
                2,
            )
        },
        "tight 2-page paged FA2",
    )
    .unwrap();
    let mut got = vec![0u16; 32 * 2 * 256];
    out.copy_to_host(&mut got, &s).unwrap();
    s.synchronize().unwrap();
    let perm = [1usize, 0usize];
    for row in 0..32usize {
        for col in 0..256usize {
            let end = row + 1;
            let expected: f32 = (0..end)
                .map(|t| half(values[(perm[t / 16] * 16 + t % 16) * 256 + col]))
                .sum::<f32>()
                / end as f32;
            for h in 0..2usize {
                assert!((half(got[(row * 2 + h) * 256 + col]) - expected).abs() < 0.001);
            }
        }
    }
}

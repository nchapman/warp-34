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

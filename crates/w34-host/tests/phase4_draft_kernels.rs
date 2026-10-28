#![cfg(feature = "cuda")]
//! Focused kernel regressions; these are not end-to-end Phase 4 gates.
use core::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, check_rc};
use w34_kernels::ffi;

#[test]
fn draft_attention_matches_pinned_sglang_flashinfer_operator() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/oracle/goldens/sglang_dflash_attention.safetensors");
    let archive = w34_host::loader::SafetensorsArchive::open(&path)
        .expect("run tools/oracle/sglang_dflash_attention_dump.py in pinned SGLang image");
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    for (name, rows, seq_len, window) in [
        ("short", 8, 24, 16),
        ("full", 16, 24, 0),
        ("long", 8, 2064, 2048),
        ("narrow", 16, 24, 4),
    ] {
        replay_paged_bf16(&archive, &stream, name, rows, seq_len, window);
        let q = archive.read_f32(&format!("{name}.q")).unwrap();
        let k = archive.read_bf16_bits(&format!("{name}.k")).unwrap();
        let v = archive.read_bf16_bits(&format!("{name}.v")).unwrap();
        let expected = archive.read_f32(&format!("{name}.out")).unwrap();
        let dq = DeviceBuffer::alloc(0, q.len() * 4).unwrap();
        let dk = DeviceBuffer::alloc(0, k.len() * 2).unwrap();
        let dv = DeviceBuffer::alloc(0, v.len() * 2).unwrap();
        let out = DeviceBuffer::alloc(0, expected.len() * 4).unwrap();
        dq.copy_from_host(&q, &stream).unwrap();
        dk.copy_from_host(&k, &stream).unwrap();
        dv.copy_from_host(&v, &stream).unwrap();
        // SAFETY: fixture arrays have Q=8/16,Hq=8,Hkv=2,D=128 and S cache rows.
        unsafe {
            check_rc(
                ffi::w34_dflash_attention_f32(
                    &ex,
                    dq.addr() as *const f32,
                    dk.addr() as *const c_void,
                    dv.addr() as *const c_void,
                    out.addr() as *mut f32,
                    rows,
                    seq_len - rows,
                    seq_len,
                    8,
                    2,
                    128,
                    window,
                    1.0 / 128.0f32.sqrt(),
                ),
                "donor attention replay",
            )
            .unwrap();
        }
        let mut actual = vec![0.0f32; expected.len()];
        out.copy_to_host(&mut actual, &stream).unwrap();
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            actual.iter().all(|x| x.is_finite()) && max_abs < 0.02,
            "{name}: max abs {max_abs} vs actual SGLang/FlashInfer BF16 output"
        );
    }
}

fn replay_paged_bf16(
    archive: &w34_host::loader::SafetensorsArchive,
    stream: &CudaStream,
    name: &str,
    rows: i32,
    seq_len: i32,
    window: i32,
) {
    use w34_host::cache::{KvFormatPolicy, KvPool};
    let ex = stream.exec(0, 0);
    let upload = |data: &[u8]| {
        let buf = DeviceBuffer::alloc(0, data.len()).unwrap();
        buf.copy_from_host(data, stream).unwrap();
        buf
    };
    let ints = |data: &[i32]| {
        upload(
            &data
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let q = upload(&archive.read_bytes(&format!("{name}.q")).unwrap());
    let k = upload(&archive.read_bytes(&format!("{name}.k")).unwrap());
    let v = upload(&archive.read_bytes(&format!("{name}.v")).unwrap());
    let expected = archive.read_f32(&format!("{name}.out")).unwrap();
    let pages = (seq_len as usize).div_ceil(16);
    let pool = KvPool::new(
        0,
        ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
        &[0],
        2,
        128,
        16,
        pages,
        KvFormatPolicy::Bf16,
    )
    .unwrap();
    let layout = &pool.layers[0].layout;
    let table: Vec<i32> = (0..pages as i32).rev().collect();
    let slots: Vec<i32> = (0..seq_len)
        .map(|i| table[i as usize / 16] * 16 + i % 16)
        .collect();
    let blocks = ints(&table);
    let mapping = ints(&slots);
    let kv_start = ints(&[0, seq_len]);
    let q_start = ints(&[0, rows]);
    let ctx = ints(&[seq_len - rows]);
    let seq = ints(&[seq_len]);
    let masked_req = upload(&[0u8]);
    let out = upload(&vec![0u8; expected.len() * 2]);
    let mut batch = ffi::W34AttentionBatch {
        query_start_loc: kv_start.addr() as *const i32,
        context_lens_before: ctx.addr() as *const i32,
        seq_lens_after: seq.addr() as *const i32,
        block_tables: blocks.addr() as *const i32,
        write_slot_mapping: mapping.addr() as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: seq_len,
        max_blocks_per_req: pages as i32,
        max_q_len: seq_len,
        max_seq_len: seq_len,
        sliding_window: window,
        mask_mode: ffi::W34_ATTN_MASK_NONCAUSAL_WINDOW,
        sm_scale: 1.0 / 128.0f32.sqrt(),
        reserved: 0,
    };
    // SAFETY: cache layout covers all reversed physical pages, metadata buffers
    // live through synchronization, and source tensors have 2 KV / 8 query heads.
    unsafe {
        check_rc(
            ffi::w34_kv_cache_write_bf16(
                &ex,
                layout,
                &batch,
                k.addr() as *const c_void,
                v.addr() as *const c_void,
            ),
            "paged BF16 write",
        )
        .unwrap();
        // A padded request must not overwrite live cache, even with valid slots.
        batch.valid_req_mask = masked_req.addr() as *const u8;
        check_rc(
            ffi::w34_kv_cache_write_bf16(
                &ex,
                layout,
                &batch,
                v.addr() as *const c_void,
                k.addr() as *const c_void,
            ),
            "masked BF16 write",
        )
        .unwrap();
        batch.valid_req_mask = core::ptr::null();
        batch.query_start_loc = q_start.addr() as *const i32;
        batch.total_tokens = rows;
        batch.max_q_len = rows;
        check_rc(
            ffi::w34_paged_attention_bf16(
                &ex,
                layout,
                &batch,
                q.addr() as *const c_void,
                out.addr() as *mut c_void,
                8,
            ),
            "paged BF16 attention",
        )
        .unwrap();
    }
    let mut bits = vec![0u16; expected.len()];
    out.copy_to_host(&mut bits, stream).unwrap();
    let actual: Vec<f32> = bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        actual.iter().all(|x| x.is_finite()) && max_abs < 0.02,
        "{name} paged native BF16: max abs {max_abs}"
    );
    // Masked requests leave output storage untouched (including captured replay).
    let sentinel = vec![0xC2F6u16; expected.len()];
    out.copy_from_host(&sentinel, stream).unwrap();
    batch.valid_req_mask = masked_req.addr() as *const u8;
    // SAFETY: unchanged validated buffers; only the live request mask is disabled.
    unsafe {
        check_rc(
            ffi::w34_paged_attention_bf16(
                &ex,
                layout,
                &batch,
                q.addr() as *const c_void,
                out.addr() as *mut c_void,
                8,
            ),
            "masked BF16 attention",
        )
        .unwrap();
    }
    out.copy_to_host(&mut bits, stream).unwrap();
    assert_eq!(bits, sentinel);
}

#[test]
fn draft_kv_preserves_bf16_range_through_attention() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let k = DeviceBuffer::alloc(0, 128 * 4).unwrap();
    let v = DeviceBuffer::alloc(0, 128 * 4).unwrap();
    let q = DeviceBuffer::alloc(0, 128 * 4).unwrap();
    let kc = DeviceBuffer::alloc(0, 128 * 2).unwrap();
    let vc = DeviceBuffer::alloc(0, 128 * 2).unwrap();
    let out = DeviceBuffer::alloc(0, 128 * 4).unwrap();
    k.copy_from_host(&vec![0.0f32; 128], &stream).unwrap();
    q.copy_from_host(&vec![1.0f32; 128], &stream).unwrap();
    v.copy_from_host(&vec![131072.0f32; 128], &stream).unwrap();
    // SAFETY: all device arrays cover a single 128-element head on this stream.
    unsafe {
        check_rc(
            ffi::w34_dflash_kv_write_f32_to_bf16(
                &ex,
                k.addr() as *const f32,
                v.addr() as *const f32,
                kc.addr() as *mut c_void,
                vc.addr() as *mut c_void,
                0,
                1,
                128,
            ),
            "draft kv write",
        )
        .unwrap();
        check_rc(
            ffi::w34_dflash_attention_f32(
                &ex,
                q.addr() as *const f32,
                kc.addr() as *const c_void,
                vc.addr() as *const c_void,
                out.addr() as *mut f32,
                1,
                0,
                1,
                1,
                1,
                128,
                0,
                1.0,
            ),
            "draft attention",
        )
        .unwrap();
    }
    let mut result = vec![0.0f32; 128];
    out.copy_to_host(&mut result, &stream).unwrap();
    assert!(
        result.iter().all(|&x| x == 131072.0),
        "BF16 range lost: {}",
        result[0]
    );
}

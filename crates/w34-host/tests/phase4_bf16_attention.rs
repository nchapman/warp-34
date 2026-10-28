#![cfg(feature = "cuda")]
//! Actual donor operands, never native upstream outputs. Run with CUDA_VISIBLE_DEVICES=1.
//! Pinned sm86 FA2 uses 16-key KV tiles (CTA_TILE_Q=64 is the QUERY tile).
//! The 21-key fixtures exercise two online probability-rounding updates.
//! Keep all five layers at the same strict 1e-4 bound.
use core::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, check_rc};
use w34_kernels::ffi;
fn replay(layers: std::ops::Range<usize>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/oracle/goldens/sglang_dflash_model_4b.safetensors");
    let archive = w34_host::loader::SafetensorsArchive::open(&path).unwrap();
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for layer in layers {
        replay_paged_bf16(&archive, &stream, &format!("layers.{layer}"), 16, 21, 0);
    }
}
#[test]
fn first_layer_actual_donor_attention() {
    replay(0..1);
}
#[test]
fn all_five_layers_actual_donor_attention() {
    let mut failed = Vec::new();
    for layer in 0..5 {
        if std::panic::catch_unwind(|| replay(layer..layer + 1)).is_err() {
            failed.push(layer);
        }
    }
    assert!(
        failed.is_empty(),
        "donor attention mismatch in layers {failed:?}"
    );
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
    let k = upload(&archive.read_bytes(&format!("{name}.k_all")).unwrap());
    let v = upload(&archive.read_bytes(&format!("{name}.v_all")).unwrap());
    let expected = archive.read_f32(&format!("{name}.attention")).unwrap();
    let pages = (seq_len as usize).div_ceil(16);
    let pool = KvPool::new(
        0,
        ffi::W34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
        &[0],
        4,
        256,
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
        sm_scale: 1.0 / 256.0f32.sqrt(),
        reserved: 0,
    };
    // SAFETY: cache layout covers all reversed physical pages, metadata buffers
    // live through synchronization, and source tensors have 4 KV / 16 query heads.
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
                16,
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
    let rel = (actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / expected.iter().map(|b| (*b as f64).powi(2)).sum::<f64>())
    .sqrt();
    eprintln!("{name}: relL2={rel:.9} maxabs={max_abs}");
    assert!(
        actual.iter().all(|x| x.is_finite()) && rel < 0.0001,
        "{name}: relL2={rel}, maxabs={max_abs}"
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
                16,
            ),
            "masked BF16 attention",
        )
        .unwrap();
    }
    out.copy_to_host(&mut bits, stream).unwrap();
    assert_eq!(bits, sentinel);
}

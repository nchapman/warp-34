#[cfg(feature = "cuda")]
use core::ffi::c_void;
mod common;
#[cfg(feature = "cuda")]
use std::sync::Mutex;
#[cfg(feature = "cuda")]
use w34_host::cache::{KvFormatPolicy, StateBundlePool, StateTx};
#[cfg(feature = "cuda")]
use w34_host::cuda::{
    CudaDevice, CudaStream, DeviceBuffer, Tp2Collectives, f16_to_f32, f32_to_f16,
    prepare_execution_plan,
};
#[cfg(feature = "cuda")]
use w34_host::model::{Qwen3_5Engine, TapPlan};
#[cfg(feature = "cuda")]
use w34_kernels::ffi::{self, W34GdnBatch};

#[cfg(feature = "cuda")]
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Runs one `w34_gdn_recurrent_f16` call over `num_reqs` requests described by host metadata
/// `[query_start_loc[R+1], state_in[R], state_out[R]]` (all requests have initial state).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_gdn_batch(
    ex: &w34_kernels::ffi::W34Exec,
    stream: &CudaStream,
    d_meta: &DeviceBuffer,
    qsl: &[i32],
    state_in: &[i32],
    state_out: &[i32],
    qkv: *const c_void,
    b: *const c_void,
    a: *const c_void,
    d_alog: &DeviceBuffer,
    d_dtb: &DeviceBuffer,
    ssm_state: u64,
    out: u64,
    dims: (usize, usize, usize, usize, usize),
) {
    let (hk, dk, hv, dv, num_slots) = dims;
    let r = state_in.len();
    let mut meta: Vec<i32> = Vec::new();
    meta.extend_from_slice(qsl);
    meta.extend_from_slice(state_in);
    meta.extend_from_slice(state_out);
    let has_init_off = meta.len() * 4;
    meta.push(0x0101_0101); // has_initial_state bytes (up to 4 requests)
    d_meta.copy_from_host(&meta, stream).unwrap();
    let batch = W34GdnBatch {
        query_start_loc: d_meta.addr() as *const i32,
        state_in_indices: (d_meta.addr() + (qsl.len() * 4) as u64) as *const i32,
        state_out_indices: (d_meta.addr() + ((qsl.len() + r) * 4) as u64) as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: (d_meta.addr() + has_init_off as u64) as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: r as i32,
        total_tokens: qsl[r] - qsl[0],
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    // SAFETY: `batch` and GDN recurrent buffers for range `qsl[0..r]` are live on device 0.
    unsafe {
        assert_eq!(
            ffi::w34_gdn_recurrent_f16(
                ex,
                &batch,
                qkv,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                b,
                a,
                d_alog.addr() as *const f32,
                d_dtb.addr() as *const f32,
                ssm_state as *mut f32,
                out as *mut c_void,
                hk as i32,
                dk as i32,
                hv as i32,
                dv as i32,
                num_slots as i32,
            ),
            0
        );
    }
    stream.synchronize().unwrap();
}

/// A mixed continuous-batching step (`q_len = [1, 256]`) takes the Tensor-Core WY path: the
/// prefill request must be 0-ULP identical to running it alone, and the 1-token decode request
/// must stay within WY-vs-recurrent tolerance of the dedicated decode kernel.
#[cfg(feature = "cuda")]
#[test]
fn test_gdn_wy_mixed_decode_prefill_batch_parity() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");

    let (hk, hv, dk, dv) = (2usize, 6usize, 128usize, 128usize);
    let conv_dim = 2 * hk * dk + hv * dv;
    let prefill_len = 256usize;
    let total_len = 1 + prefill_len;
    let num_slots = 6usize;
    let dims = (hk, dk, hv, dv, num_slots);

    let qkv_host: Vec<u16> = (0..(total_len * conv_dim))
        .map(|i| f32_to_f16(((i % 113) as f32 - 56.0) * 0.009))
        .collect();
    let b_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 17) as f32 - 8.0) * 0.1))
        .collect();
    let a_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 29) as f32 - 14.0) * 0.05))
        .collect();
    let a_log: Vec<f32> = (0..hv).map(|i| -1.5 - 0.08 * (i as f32)).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|i| -0.1 + 0.02 * (i as f32)).collect();
    let d_qkv = DeviceBuffer::alloc(0, qkv_host.len() * 2).unwrap();
    let d_b = DeviceBuffer::alloc(0, b_host.len() * 2).unwrap();
    let d_a = DeviceBuffer::alloc(0, a_host.len() * 2).unwrap();
    let d_alog = DeviceBuffer::alloc(0, a_log.len() * 4).unwrap();
    let d_dtb = DeviceBuffer::alloc(0, dt_bias.len() * 4).unwrap();
    d_qkv.copy_from_host(&qkv_host, &stream).unwrap();
    d_b.copy_from_host(&b_host, &stream).unwrap();
    d_a.copy_from_host(&a_host, &stream).unwrap();
    d_alog.copy_from_host(&a_log, &stream).unwrap();
    d_dtb.copy_from_host(&dt_bias, &stream).unwrap();

    // Slots: 0 = decode-request init, 1 = prefill-request init, 2/3 = solo outputs,
    // 4/5 = mixed-batch outputs.
    let pool = StateBundlePool::new(0, &[0], num_slots, conv_dim, 4, hv, dk, dv).unwrap();
    let st = &pool.layers[0];
    let state_elems = hv * dv * dk;
    let mut ssm_init = vec![0.0f32; num_slots * state_elems];
    for (i, v) in ssm_init[..2 * state_elems].iter_mut().enumerate() {
        *v = ((i % 97) as f32 - 48.0) * 0.002;
    }
    st.ssm_state.copy_from_host(&ssm_init, &stream).unwrap();

    let d_meta = DeviceBuffer::alloc(0, 256).unwrap();
    let d_ws = DeviceBuffer::alloc(0, 4 * 1024 * 1024).unwrap();
    let ex = stream.exec(d_ws.addr(), d_ws.len_bytes);
    let row_bytes = (hv * dv * 2) as u64;
    let qkv_at = |tok: usize| (d_qkv.addr() + (tok * conv_dim * 2) as u64) as *const c_void;
    let b_at = |tok: usize| (d_b.addr() + (tok * hv * 2) as u64) as *const c_void;
    let a_at = |tok: usize| (d_a.addr() + (tok * hv * 2) as u64) as *const c_void;
    let ssm = st.ssm_state.addr();

    let d_out_solo = DeviceBuffer::alloc(0, total_len * hv * dv * 2).unwrap();
    run_gdn_batch(
        &ex,
        &stream,
        &d_meta,
        &[0, 1],
        &[0],
        &[2],
        qkv_at(0),
        b_at(0),
        a_at(0),
        &d_alog,
        &d_dtb,
        ssm,
        d_out_solo.addr(),
        dims,
    );
    run_gdn_batch(
        &ex,
        &stream,
        &d_meta,
        &[0, prefill_len as i32],
        &[1],
        &[3],
        qkv_at(1),
        b_at(1),
        a_at(1),
        &d_alog,
        &d_dtb,
        ssm,
        d_out_solo.addr() + row_bytes,
        dims,
    );
    let d_out_mixed = DeviceBuffer::alloc(0, total_len * hv * dv * 2).unwrap();
    run_gdn_batch(
        &ex,
        &stream,
        &d_meta,
        &[0, 1, total_len as i32],
        &[0, 1],
        &[4, 5],
        qkv_at(0),
        b_at(0),
        a_at(0),
        &d_alog,
        &d_dtb,
        ssm,
        d_out_mixed.addr(),
        dims,
    );

    let mut out_solo = vec![0u16; total_len * hv * dv];
    let mut out_mixed = vec![0u16; total_len * hv * dv];
    let mut ssm_all = vec![0.0f32; num_slots * state_elems];
    d_out_solo.copy_to_host(&mut out_solo, &stream).unwrap();
    d_out_mixed.copy_to_host(&mut out_mixed, &stream).unwrap();
    st.ssm_state.copy_to_host(&mut ssm_all, &stream).unwrap();
    stream.synchronize().unwrap();

    let row = hv * dv;
    let slot = |s: usize| &ssm_all[s * state_elems..(s + 1) * state_elems];
    assert_eq!(
        out_mixed[row..],
        out_solo[row..],
        "mixed-batch prefill output must match solo WY prefill 0-ULP"
    );
    assert_eq!(
        slot(5),
        slot(3),
        "mixed-batch prefill ssm_state must match solo 0-ULP"
    );

    let max_out_err = out_solo[..row]
        .iter()
        .zip(out_mixed[..row].iter())
        .map(|(&r, &g)| (f16_to_f32(r) - f16_to_f32(g)).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_out_err < 3e-3,
        "mixed decode output max_abs_err={max_out_err}"
    );
    let max_s_err = slot(2)
        .iter()
        .zip(slot(4).iter())
        .map(|(&r, &g)| (r - g).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_s_err < 3e-3,
        "mixed decode ssm_state max_abs_err={max_s_err}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_graph_spec_rollback_and_tp2_collectives() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Cold-start TP=2 init & typed collectives stress (including token IDs > 65504)
    let dev0 = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let dev1 = CudaDevice::set_current(1).expect("CudaDevice(1)");
    let stream0 = CudaStream::new(dev0.device_id).expect("stream0");
    let stream1 = CudaStream::new(dev1.device_id).expect("stream1");
    prepare_execution_plan(dev0.device_id).unwrap();
    prepare_execution_plan(dev1.device_id).unwrap();

    let comm = Tp2Collectives::new(4096 * 2).expect("Tp2Collectives");
    let ex0 = stream0.exec(0, 0);
    let ex1 = stream1.exec(0, 0);

    let d_val0 = DeviceBuffer::alloc(0, 4).unwrap();
    let d_id0 = DeviceBuffer::alloc(0, 4).unwrap();
    let d_out_val0 = DeviceBuffer::alloc(0, 4).unwrap();
    let d_out_id0 = DeviceBuffer::alloc(0, 4).unwrap();
    d_val0.copy_from_host(&[12.5f32], &stream0).unwrap();
    d_id0.copy_from_host(&[100_001i32], &stream0).unwrap();

    let d_val1 = DeviceBuffer::alloc(1, 4).unwrap();
    let d_id1 = DeviceBuffer::alloc(1, 4).unwrap();
    let d_out_val1 = DeviceBuffer::alloc(1, 4).unwrap();
    let d_out_id1 = DeviceBuffer::alloc(1, 4).unwrap();
    d_val1.copy_from_host(&[19.75f32], &stream1).unwrap();
    d_id1.copy_from_host(&[240_123i32], &stream1).unwrap(); // > 65504 FP16 max!

    std::thread::scope(|s| {
        let h0 = s.spawn(|| {
            dev0.make_current().unwrap();
            comm.all_gather_top1_f32_i32(
                &ex0,
                d_val0.addr(),
                d_id0.addr(),
                d_out_val0.addr(),
                d_out_id0.addr(),
                0,
                1,
            )
            .unwrap();
            stream0.synchronize().unwrap();
        });
        let h1 = s.spawn(|| {
            dev1.make_current().unwrap();
            comm.all_gather_top1_f32_i32(
                &ex1,
                d_val1.addr(),
                d_id1.addr(),
                d_out_val1.addr(),
                d_out_id1.addr(),
                1,
                1,
            )
            .unwrap();
            stream1.synchronize().unwrap();
        });
        h0.join().unwrap();
        h1.join().unwrap();
    });

    let mut win0 = [0i32; 1];
    let mut win1 = [0i32; 1];
    d_out_id0.copy_to_host(&mut win0, &stream0).unwrap();
    d_out_id1.copy_to_host(&mut win1, &stream1).unwrap();
    assert_eq!(win0[0], 240_123, "INT32 token ID > 65504 preserved");
    assert_eq!(win1[0], 240_123, "both ranks agree on top-1");

    // CUDA Graph replay under metadata mutation + TapPlan + StateTx speculative rollback
    let taps = TapPlan {
        layer_ids: vec![5, 19, 31],
    };
    assert_eq!(taps.layer_ids.len(), 3);
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "test_cuda_graph_spec_rollback_and_tp2_collectives",
    )
    .is_none()
    {
        return;
    }

    let mut eng_graph =
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Fp16, Some(256), true)
            .expect("load 4B graph");
    let mut eng_eager =
        Qwen3_5Engine::load("Qwen/Qwen3.5-4B", 1, KvFormatPolicy::Fp16, Some(256), false)
            .expect("load 4B eager");

    let prompt_ids = eng_graph
        .encode("The capital of France is")
        .expect("encode");
    let m_graph = eng_graph.generate_tokens(&prompt_ids, 16, false).unwrap();
    let m_eager = eng_eager.generate_tokens(&prompt_ids, 16, false).unwrap();
    assert_eq!(
        m_graph.generated_token_ids, m_eager.generated_token_ids,
        "CUDA Graph replay matches eager execution bit-for-bit"
    );

    // Also verify StateTx speculative verify & rollback schedule (accepted_len = 3 out of 1+4)
    let mut tx = StateTx::new(0);
    tx.begin_prefill();
    tx.commit_prefill(prompt_ids.len());
    tx.begin_spec_verify(vec![1, 2, 3, 4]);
    assert_eq!(tx.verify_slot_schedule(), vec![0, 1, 2, 3, 4]);
    let freed = tx.resolve_spec_verify(3);
    assert_eq!(tx.committed_slot, 2);
    assert_eq!(freed, vec![0, 1, 3, 4]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_tp2_p2p_and_tp4_collectives() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let num_gpus = CudaDevice::count().unwrap_or(0);
    if num_gpus < 2 {
        let strict = ["W34_STRICT_TESTS", "CI"].into_iter().any(|var| {
            std::env::var(var).is_ok_and(|val| {
                let t = val.trim();
                t == "1" || t.eq_ignore_ascii_case("true")
            })
        });
        assert!(!strict, "test_tp2_p2p_and_tp4_collectives requires 2 GPUs");
        return;
    }
    for dev_id in 0..2 {
        prepare_execution_plan(dev_id).unwrap();
    }
    for allow_p2p in [false, true] {
        let comm = Tp2Collectives::new_for_tp_with_p2p(2, 64 * 1024, allow_p2p).unwrap();
        let dev0 = CudaDevice::set_current(0).unwrap();
        let dev1 = CudaDevice::set_current(1).unwrap();
        let s0 = CudaStream::new(0).unwrap();
        let s1 = CudaStream::new(1).unwrap();
        let n = 2560usize;
        let in0: Vec<u16> = (0..n).map(|i| f32_to_f16((i % 17) as f32 * 0.25)).collect();
        let in1: Vec<u16> = (0..n)
            .map(|i| f32_to_f16(((i + 5) % 19) as f32 * 0.5))
            .collect();
        let d_in0 = DeviceBuffer::alloc(0, n * 2).unwrap();
        let d_out0 = DeviceBuffer::alloc(0, n * 2).unwrap();
        let d_in1 = DeviceBuffer::alloc(1, n * 2).unwrap();
        let d_out1 = DeviceBuffer::alloc(1, n * 2).unwrap();
        d_in0.copy_from_host(&in0, &s0).unwrap();
        d_in1.copy_from_host(&in1, &s1).unwrap();
        std::thread::scope(|sc| {
            let h0 = sc.spawn(|| {
                dev0.make_current().unwrap();
                let ex = s0.exec(0, 0);
                comm.all_reduce_f16(&ex, d_in0.addr(), d_out0.addr(), 0, n as i32)
                    .unwrap();
                s0.synchronize().unwrap();
            });
            let h1 = sc.spawn(|| {
                dev1.make_current().unwrap();
                let ex = s1.exec(0, 0);
                comm.all_reduce_f16(&ex, d_in1.addr(), d_out1.addr(), 1, n as i32)
                    .unwrap();
                s1.synchronize().unwrap();
            });
            h0.join().unwrap();
            h1.join().unwrap();
        });
        let mut h_out0 = vec![0u16; n];
        let mut h_out1 = vec![0u16; n];
        d_out0.copy_to_host(&mut h_out0, &s0).unwrap();
        d_out1.copy_to_host(&mut h_out1, &s1).unwrap();
        s0.synchronize().unwrap();
        s1.synchronize().unwrap();
        assert_eq!(h_out0, h_out1);
        for i in 0..n {
            let expected = f16_to_f32(in0[i]) + f16_to_f32(in1[i]);
            assert!((f16_to_f32(h_out0[i]) - expected).abs() <= 1e-2);
        }
    }

    if num_gpus >= 4 {
        for dev_id in 0..4 {
            prepare_execution_plan(dev_id).unwrap();
        }
        let comm4 = Tp2Collectives::new_for_tp(4, 256 * 1024).unwrap();
        let n = 2560usize;
        let mut devs = Vec::new();
        let mut streams = Vec::new();
        let mut d_ins = Vec::new();
        let mut d_outs = Vec::new();
        let mut d_vals = Vec::new();
        let mut d_ids = Vec::new();
        let mut d_gvals = Vec::new();
        let mut d_gids = Vec::new();
        for r in 0..4i32 {
            let dev = CudaDevice::set_current(r).unwrap();
            let s = CudaStream::new(r).unwrap();
            let h_in: Vec<u16> = (0..n)
                .map(|i| f32_to_f16(((i + r as usize) % 13) as f32 * 0.25))
                .collect();
            let d_in = DeviceBuffer::alloc(r, n * 2).unwrap();
            let d_out = DeviceBuffer::alloc(r, n * 2).unwrap();
            d_in.copy_from_host(&h_in, &s).unwrap();
            let d_val = DeviceBuffer::alloc(r, 4).unwrap();
            let d_id = DeviceBuffer::alloc(r, 4).unwrap();
            let d_gval = DeviceBuffer::alloc(r, 4).unwrap();
            let d_gid = DeviceBuffer::alloc(r, 4).unwrap();
            d_val
                .copy_from_host(&[10.0f32 + (r as f32) * 3.5], &s)
                .unwrap();
            d_id.copy_from_host(&[100_000i32 + r * 50_000], &s).unwrap();
            s.synchronize().unwrap();
            devs.push(dev);
            streams.push(s);
            d_ins.push(d_in);
            d_outs.push(d_out);
            d_vals.push(d_val);
            d_ids.push(d_id);
            d_gvals.push(d_gval);
            d_gids.push(d_gid);
        }
        std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for (r, ((((((dev, s), d_in), d_out), d_val), d_id), (d_gval, d_gid))) in devs
                .iter()
                .zip(streams.iter())
                .zip(d_ins.iter())
                .zip(d_outs.iter())
                .zip(d_vals.iter())
                .zip(d_ids.iter())
                .zip(d_gvals.iter().zip(d_gids.iter()))
                .enumerate()
            {
                let c = &comm4;
                handles.push(sc.spawn(move || {
                    dev.make_current().unwrap();
                    let ex = s.exec(0, 0);
                    c.all_reduce_f16(&ex, d_in.addr(), d_out.addr(), r as i32, n as i32)
                        .unwrap();
                    c.all_gather_top1_f32_i32(
                        &ex,
                        d_val.addr(),
                        d_id.addr(),
                        d_gval.addr(),
                        d_gid.addr(),
                        r as i32,
                        1,
                    )
                    .unwrap();
                    s.synchronize().unwrap();
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        let mut ref_out = vec![0u16; n];
        d_outs[0].copy_to_host(&mut ref_out, &streams[0]).unwrap();
        streams[0].synchronize().unwrap();
        for r in 0..4 {
            let mut h_out = vec![0u16; n];
            let mut gid = [0i32; 1];
            d_outs[r].copy_to_host(&mut h_out, &streams[r]).unwrap();
            d_gids[r].copy_to_host(&mut gid, &streams[r]).unwrap();
            streams[r].synchronize().unwrap();
            assert_eq!(h_out, ref_out);
            assert_eq!(gid[0], 250_000);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_tp2_and_tp4_all_gather_topk_and_hc_combine() {
    let _g = GPU_LOCK.lock().unwrap();
    for tp in [2usize, 4usize] {
        let comm = Tp2Collectives::new_for_tp(tp, 1 << 20).unwrap();
        let streams: Vec<_> = (0..tp).map(|_| CudaStream::new(0).unwrap()).collect();

        // 1. Verify `all_gather_topk_f32_i32` (in-place and out-of-place, with tie-breaks)
        let rows = 3usize;
        let k = 4usize;
        let mut d_vals = Vec::new();
        let mut d_ids = Vec::new();
        for (r, s) in streams.iter().enumerate() {
            let mut vals = Vec::with_capacity(rows * k);
            let mut ids = Vec::with_capacity(rows * k);
            for row in 0..rows {
                for j in 0..k {
                    // Row 1 has exact ties at j=0 across ranks (`100.0`) to test deterministic lower-ID tie-breaking.
                    let v = if row == 1 && j == 0 {
                        100.0f32
                    } else {
                        ((rows - row) * 20 + (k - j) * 3 + r) as f32
                    };
                    vals.push(v);
                    ids.push((r * 1000 + row * 100 + j) as i32);
                }
            }
            let dv = DeviceBuffer::alloc(0, rows * k * 4).unwrap();
            let di = DeviceBuffer::alloc(0, rows * k * 4).unwrap();
            dv.copy_from_host(&vals, s).unwrap();
            di.copy_from_host(&ids, s).unwrap();
            d_vals.push(dv);
            d_ids.push(di);
        }

        // Also prepare fused `all_reduce_hc_combine_tap_f16` inputs `[num_tokens=2, hc_count=4, h=64]`
        let num_tokens = 2usize;
        let hc_count = 4usize;
        let h = 64usize;
        let res_in_host: Vec<u16> = (0..num_tokens * hc_count * h)
            .map(|i| f32_to_f16(((i % 19) as f32 - 9.0) * 0.1))
            .collect();
        let inj_host: Vec<f32> = (0..num_tokens * hc_count)
            .map(|i| ((i % 5) as f32 + 1.0) * 0.25)
            .collect();
        let mut d_res_fused = Vec::new();
        let mut d_blk_in = Vec::new();
        let mut d_inj = Vec::new();
        let mut d_tap_fused = Vec::new();
        for (r, s) in streams.iter().enumerate() {
            let blk: Vec<u16> = (0..num_tokens * h)
                .map(|i| f32_to_f16(((i + r * 7) % 13) as f32 * 0.125 - 0.5))
                .collect();
            let dr = DeviceBuffer::alloc(0, res_in_host.len() * 2).unwrap();
            let db = DeviceBuffer::alloc(0, blk.len() * 2).unwrap();
            let di = DeviceBuffer::alloc(0, inj_host.len() * 4).unwrap();
            let dt = DeviceBuffer::alloc(0, num_tokens * h * 2).unwrap();
            dr.copy_from_host(&res_in_host, s).unwrap();
            db.copy_from_host(&blk, s).unwrap();
            di.copy_from_host(&inj_host, s).unwrap();
            d_res_fused.push(dr);
            d_blk_in.push(db);
            d_inj.push(di);
            d_tap_fused.push(dt);
        }

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (r, ((((s, dv), di), dr), (db, (dinj, dt)))) in streams
                .iter()
                .zip(d_vals.iter())
                .zip(d_ids.iter())
                .zip(d_res_fused.iter())
                .zip(d_blk_in.iter().zip(d_inj.iter().zip(d_tap_fused.iter())))
                .enumerate()
            {
                let c = &comm;
                handles.push(scope.spawn(move || {
                    let _dev = CudaDevice::set_current(0).unwrap();
                    let ex = s.exec(0, 0);
                    c.all_gather_top1_f32_i32(
                        &ex,
                        dv.addr(),
                        di.addr(),
                        dv.addr(),
                        di.addr(),
                        r as i32,
                        1,
                    )
                    .unwrap();
                    c.broadcast_i32(&ex, di.addr(), di.addr(), r as i32, 1)
                        .unwrap();
                    c.all_gather_topk_f32_i32(
                        &ex,
                        dv.addr(),
                        di.addr(),
                        dv.addr(),
                        di.addr(),
                        r as i32,
                        rows as i32,
                        k as i32,
                    )
                    .unwrap();
                    // Variable-size all_reduce_f16 back-to-back (large 128 elems -> small 32 elems)
                    c.all_reduce_f16(&ex, dr.addr(), dt.addr(), r as i32, 128)
                        .unwrap();
                    c.all_reduce_f16(&ex, db.addr(), dt.addr(), r as i32, 32)
                        .unwrap();
                    c.all_reduce_hc_combine_tap_f16(
                        &ex,
                        dr.addr(),
                        db.addr(),
                        db.addr(),
                        dinj.addr(),
                        0,
                        dr.addr(),
                        dt.addr(),
                        0,
                        r as i32,
                        num_tokens as i32,
                        hc_count as i32,
                        h as i32,
                        h as i32,
                        0,
                    )
                    .unwrap();
                    s.synchronize().unwrap();
                }));
            }
            for h_join in handles {
                h_join.join().unwrap();
            }
        });

        // Verify `all_gather_topk_f32_i32`: all ranks must agree, row 1 top-1 must be rank 0's tie-broken ID `100`.
        let mut ref_ids = vec![0i32; rows * k];
        d_ids[0].copy_to_host(&mut ref_ids, &streams[0]).unwrap();
        streams[0].synchronize().unwrap();
        assert_eq!(ref_ids[k], 100);
        for r in 1..tp {
            let mut ids_r = vec![0i32; rows * k];
            d_ids[r].copy_to_host(&mut ids_r, &streams[r]).unwrap();
            streams[r].synchronize().unwrap();
            assert_eq!(ids_r, ref_ids);
        }

        // Verify `all_reduce_hc_combine_tap_f16`: compare against unfused `hc_combine_tap_f16` on rank 0's all-reduced `db`.
        let d_res_ref = DeviceBuffer::alloc(0, res_in_host.len() * 2).unwrap();
        let d_tap_ref = DeviceBuffer::alloc(0, num_tokens * h * 2).unwrap();
        d_res_ref.copy_from_host(&res_in_host, &streams[0]).unwrap();
        w34_host::ple::gpu_ops::hc_combine_tap_f16(
            &streams[0].exec(0, 0),
            d_res_ref.addr(),
            d_blk_in[0].addr(),
            d_inj[0].addr(),
            0,
            d_res_ref.addr(),
            d_tap_ref.addr(),
            0,
            num_tokens,
            hc_count,
            h,
            h as i32,
            0,
        )
        .unwrap();
        let mut expected_res = vec![0u16; res_in_host.len()];
        let mut expected_tap = vec![0u16; num_tokens * h];
        d_res_ref
            .copy_to_host(&mut expected_res, &streams[0])
            .unwrap();
        d_tap_ref
            .copy_to_host(&mut expected_tap, &streams[0])
            .unwrap();
        streams[0].synchronize().unwrap();

        for r in 0..tp {
            let mut got_res = vec![0u16; res_in_host.len()];
            let mut got_tap = vec![0u16; num_tokens * h];
            d_res_fused[r]
                .copy_to_host(&mut got_res, &streams[r])
                .unwrap();
            d_tap_fused[r]
                .copy_to_host(&mut got_tap, &streams[r])
                .unwrap();
            streams[r].synchronize().unwrap();
            assert_eq!(got_res, expected_res, "TP={tp} rank={r} residual mismatch");
            assert_eq!(got_tap, expected_tap, "TP={tp} rank={r} tap mismatch");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_tp2_alternating_cta_counts_preserve_staging_banks() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut dev_count = 0i32;
    // SAFETY: queries CUDA runtime device count.
    if unsafe { ffi::w34_get_device_count(&raw mut dev_count) } != 0 || dev_count < 2 {
        return;
    }
    for r in 0..2 {
        prepare_execution_plan(r).unwrap();
    }
    // M=16, H=1536 -> 24,576 elements = 48 KiB (< 64 KiB SM-TX path):
    // - `all_reduce_residual_gemma_rms_norm_f16` launches `blocks = 16` CTAs (row-major).
    // - `all_reduce_f16` launches `blocks = 6` CTAs (flat strided).
    let (m, h) = (16usize, 1536usize);
    let n = m * h;
    let comm = Tp2Collectives::new_for_tp_with_p2p(2, 64 * 1024, false).unwrap();

    let w_f32: Vec<f32> = (0..h).map(|i| 0.05 * ((i % 13) as f32 - 6.0)).collect();
    let w_f16: Vec<u16> = w_f32.iter().copied().map(f32_to_f16).collect();

    let streams = [CudaStream::new(0).unwrap(), CudaStream::new(1).unwrap()];
    let s_bg1 = CudaStream::new(1).unwrap();
    let d_w = [
        DeviceBuffer::alloc(0, h * 2).unwrap(),
        DeviceBuffer::alloc(1, h * 2).unwrap(),
    ];
    let d_in_a = [
        DeviceBuffer::alloc(0, n * 2).unwrap(),
        DeviceBuffer::alloc(1, n * 2).unwrap(),
    ];
    let d_res = [
        DeviceBuffer::alloc(0, n * 2).unwrap(),
        DeviceBuffer::alloc(1, n * 2).unwrap(),
    ];
    let d_norm_out = [
        DeviceBuffer::alloc(0, n * 2).unwrap(),
        DeviceBuffer::alloc(1, n * 2).unwrap(),
    ];
    let d_in_b = [
        DeviceBuffer::alloc(0, n * 2).unwrap(),
        DeviceBuffer::alloc(1, n * 2).unwrap(),
    ];
    let d_ar_out = [
        DeviceBuffer::alloc(0, n * 2).unwrap(),
        DeviceBuffer::alloc(1, n * 2).unwrap(),
    ];
    let d_delay = DeviceBuffer::alloc(1, 512 * 1024 * 2).unwrap();

    for r in 0..2 {
        let _dev = CudaDevice::set_current(r as i32).unwrap();
        d_w[r].copy_from_host(&w_f16, &streams[r]).unwrap();
        streams[r].synchronize().unwrap();
    }

    let steps = 8usize;
    let in_a0: Vec<u16> = (0..n)
        .map(|i| f32_to_f16(0.25 + 0.01 * ((i % 53) as f32)))
        .collect();
    let in_a1: Vec<u16> = (0..n)
        .map(|i| f32_to_f16(-0.15 + 0.01 * (((i * 3) % 47) as f32)))
        .collect();
    let res_init: Vec<u16> = (0..n)
        .map(|i| f32_to_f16(0.05 * ((i % 29) as f32) - 0.7))
        .collect();
    // Distinct giant values in `in_b` so any bank overwrite on rows 2..3 is unmistakable.
    let in_b0: Vec<u16> = (0..n).map(|i| f32_to_f16(50.0 + (i % 7) as f32)).collect();
    let in_b1: Vec<u16> = (0..n).map(|i| f32_to_f16(25.0 + (i % 5) as f32)).collect();

    for (r, (ia, ib)) in [(&in_a0, &in_b0), (&in_a1, &in_b1)].into_iter().enumerate() {
        let _dev = CudaDevice::set_current(r as i32).unwrap();
        d_in_a[r].copy_from_host(ia, &streams[r]).unwrap();
        d_res[r].copy_from_host(&res_init, &streams[r]).unwrap();
        d_in_b[r].copy_from_host(ib, &streams[r]).unwrap();
        streams[r].synchronize().unwrap();
    }

    // Stream all steps back-to-back without host barriers between collectives so Rank 0 can
    // enter Step `k` Call B (2 CTAs) while Rank 1's CTAs 2..3 are still finishing Step `k` Call A (4 CTAs).
    std::thread::scope(|sc| {
        for r in 0..2usize {
            let s = &streams[r];
            let c = &comm;
            let dw = &d_w[r];
            let dia = &d_in_a[r];
            let dres = &d_res[r];
            let dnorm = &d_norm_out[r];
            let dib = &d_in_b[r];
            let dar = &d_ar_out[r];
            let ddel = &d_delay;
            let bg = &s_bg1;
            sc.spawn(move || {
                let _dev = CudaDevice::set_current(r as i32).unwrap();
                let ex = s.exec(0, 0);
                if r == 1 {
                    let ex_bg = bg.exec(0, 0);
                    for _ in 0..64 {
                        // Saturate Rank 1 SMs concurrently so CTAs 6..15 of Call A lag behind Rank 0.
                        // SAFETY: `ddel` holds `512 * 1024` FP16s and `dw` holds `1536` FP16s on rank 1.
                        unsafe {
                            assert_eq!(
                                ffi::w34_gemma_rms_norm_f16(
                                    &ex_bg,
                                    ddel.addr() as *const c_void,
                                    dw.addr() as *const c_void,
                                    ddel.addr() as *mut c_void,
                                    core::ptr::null(),
                                    320,
                                    1536,
                                    1e-6,
                                ),
                                0
                            );
                        }
                    }
                }
                for _ in 0..steps {
                    // Call A: 16 CTAs (`M=16, H=1536`).
                    c.all_reduce_residual_gemma_rms_norm_f16(
                        &ex,
                        dia.addr(),
                        0,
                        dres.addr(),
                        dw.addr(),
                        dnorm.addr(),
                        0,
                        r as i32,
                        m as i32,
                        h as i32,
                        1e-6,
                        0,
                        0,
                    )
                    .unwrap();
                    // Call B: 6 CTAs (`num_elems = 24576`).
                    c.all_reduce_f16(&ex, dib.addr(), dar.addr(), r as i32, n as i32)
                        .unwrap();
                }
                s.synchronize().unwrap();
                if r == 1 {
                    bg.synchronize().unwrap();
                }
            });
        }
    });

    let mut expected_res = res_init.clone();
    for _ in 0..steps {
        for i in 0..n {
            let ar = f16_to_f32(f32_to_f16(f16_to_f32(in_a0[i]) + f16_to_f32(in_a1[i])));
            expected_res[i] = f32_to_f16(f16_to_f32(expected_res[i]) + ar);
        }
    }
    let mut expected_ar = vec![0u16; n];
    for i in 0..n {
        expected_ar[i] = f32_to_f16(f16_to_f32(in_b0[i]) + f16_to_f32(in_b1[i]));
    }

    for r in 0..2usize {
        let _dev = CudaDevice::set_current(r as i32).unwrap();
        let mut got_res = vec![0u16; n];
        let mut got_ar = vec![0u16; n];
        d_res[r].copy_to_host(&mut got_res, &streams[r]).unwrap();
        d_ar_out[r].copy_to_host(&mut got_ar, &streams[r]).unwrap();
        streams[r].synchronize().unwrap();
        for i in 0..n {
            let diff = (f16_to_f32(got_res[i]) - f16_to_f32(expected_res[i])).abs();
            assert!(
                diff <= 1e-2,
                "rank={r} row={} col={} res got={} expected={}",
                i / h,
                i % h,
                f16_to_f32(got_res[i]),
                f16_to_f32(expected_res[i]),
            );
            let ar_diff = (f16_to_f32(got_ar[i]) - f16_to_f32(expected_ar[i])).abs();
            assert!(
                ar_diff <= 5e-2,
                "rank={r} idx={i} ar got={} expected={}",
                f16_to_f32(got_ar[i]),
                f16_to_f32(expected_ar[i]),
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gdn_wy_extreme_decay_gates_produce_finite_outputs_and_states() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");

    let (hk, hv, dk, dv) = (2usize, 4usize, 128usize, 128usize);
    let conv_dim = 2 * hk * dk + hv * dv;
    let seq_len = 128usize;
    let num_slots = 4usize;
    let dims = (hk, dk, hv, dv, num_slots);

    let qkv_host: Vec<u16> = (0..(seq_len * conv_dim))
        .map(|i| f32_to_f16(((i % 89) as f32 - 44.0) * 0.01))
        .collect();
    let b_host: Vec<u16> = (0..(seq_len * hv))
        .map(|i| f32_to_f16(((i % 13) as f32 - 6.0) * 0.15))
        .collect();
    // Alternate strong positive and strongly negative `a` values while `a_log` includes `92.0`
    // (`-expf(92.0) = -inf`), exercising both `-inf - (-inf)` in `s_g` and `-inf * softplus(-120)`.
    let a_host: Vec<u16> = (0..(seq_len * hv))
        .map(|i| {
            if (i / hv) % 7 == 0 {
                f32_to_f16(4.0)
            } else {
                f32_to_f16(-60.0)
            }
        })
        .collect();
    let a_log = vec![92.0f32, 92.0f32, -1.2f32, 90.0f32];
    let dt_bias = vec![0.0f32, -60.0f32, -0.1f32, 2.0f32];

    let d_qkv = DeviceBuffer::alloc(0, qkv_host.len() * 2).unwrap();
    let d_b = DeviceBuffer::alloc(0, b_host.len() * 2).unwrap();
    let d_a = DeviceBuffer::alloc(0, a_host.len() * 2).unwrap();
    let d_alog = DeviceBuffer::alloc(0, a_log.len() * 4).unwrap();
    let d_dtb = DeviceBuffer::alloc(0, dt_bias.len() * 4).unwrap();
    d_qkv.copy_from_host(&qkv_host, &stream).unwrap();
    d_b.copy_from_host(&b_host, &stream).unwrap();
    d_a.copy_from_host(&a_host, &stream).unwrap();
    d_alog.copy_from_host(&a_log, &stream).unwrap();
    d_dtb.copy_from_host(&dt_bias, &stream).unwrap();

    let pool = StateBundlePool::new(0, &[0], num_slots, conv_dim, 4, hv, dk, dv).unwrap();
    let st = &pool.layers[0];
    let state_elems = hv * dv * dk;
    let ssm_init: Vec<f32> = (0..(num_slots * state_elems))
        .map(|i| ((i % 71) as f32 - 35.0) * 0.005)
        .collect();
    st.ssm_state.copy_from_host(&ssm_init, &stream).unwrap();

    let d_meta = DeviceBuffer::alloc(0, 256).unwrap();
    let d_ws = DeviceBuffer::alloc(0, 8 * 1024 * 1024).unwrap();
    let d_out_split = DeviceBuffer::alloc(0, seq_len * hv * dv * 2).unwrap();
    let d_out_chunk64 = DeviceBuffer::alloc(0, seq_len * hv * dv * 2).unwrap();

    // 1. Split-WY path (with 8 MiB workspace).
    let ex_split = stream.exec(d_ws.addr(), d_ws.len_bytes);
    run_gdn_batch(
        &ex_split,
        &stream,
        &d_meta,
        &[0, seq_len as i32],
        &[0],
        &[1],
        d_qkv.addr() as *const c_void,
        d_b.addr() as *const c_void,
        d_a.addr() as *const c_void,
        &d_alog,
        &d_dtb,
        st.ssm_state.addr(),
        d_out_split.addr(),
        dims,
    );

    // 2. Fallback chunk64-WY path (`scratch_size = 0`).
    let ex_chunk64 = stream.exec(0, 0);
    run_gdn_batch(
        &ex_chunk64,
        &stream,
        &d_meta,
        &[0, seq_len as i32],
        &[0],
        &[2],
        d_qkv.addr() as *const c_void,
        d_b.addr() as *const c_void,
        d_a.addr() as *const c_void,
        &d_alog,
        &d_dtb,
        st.ssm_state.addr(),
        d_out_chunk64.addr(),
        dims,
    );

    // 3. Single-step / decode recurrent path (`seq_len = 1`) with the same extreme gates.
    let d_out_step = DeviceBuffer::alloc(0, hv * dv * 2).unwrap();
    run_gdn_batch(
        &ex_chunk64,
        &stream,
        &d_meta,
        &[0, 1],
        &[0],
        &[3],
        d_qkv.addr() as *const c_void,
        d_b.addr() as *const c_void,
        d_a.addr() as *const c_void,
        &d_alog,
        &d_dtb,
        st.ssm_state.addr(),
        d_out_step.addr(),
        dims,
    );

    let mut out_split = vec![0u16; seq_len * hv * dv];
    let mut out_chunk64 = vec![0u16; seq_len * hv * dv];
    let mut out_step = vec![0u16; hv * dv];
    let mut ssm_got = vec![0.0f32; num_slots * state_elems];
    d_out_split.copy_to_host(&mut out_split, &stream).unwrap();
    d_out_chunk64
        .copy_to_host(&mut out_chunk64, &stream)
        .unwrap();
    d_out_step.copy_to_host(&mut out_step, &stream).unwrap();
    st.ssm_state.copy_to_host(&mut ssm_got, &stream).unwrap();
    stream.synchronize().unwrap();

    assert!(
        out_split.iter().all(|&x| f16_to_f32(x).is_finite()),
        "split-WY produced non-finite output under strong decay"
    );
    assert!(
        out_chunk64.iter().all(|&x| f16_to_f32(x).is_finite()),
        "chunk64-WY produced non-finite output under strong decay"
    );
    assert!(
        out_step.iter().all(|&x| f16_to_f32(x).is_finite()),
        "decode/recurrent GDN produced non-finite output under strong decay"
    );
    assert!(
        ssm_got[state_elems..4 * state_elems]
            .iter()
            .all(|x| x.is_finite()),
        "GDN produced non-finite ssm_state under strong decay"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_gdn_wy_duplicate_checkpoint_offsets_and_conv_kernel_size_validation() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");

    let (hk, hv, dk, dv) = (2usize, 4usize, 128usize, 128usize);
    let conv_dim = 2 * hk * dk + hv * dv;
    let seq_len = 128usize;
    let num_slots = 6usize;

    let qkv_host: Vec<u16> = (0..(seq_len * conv_dim))
        .map(|i| f32_to_f16(((i % 83) as f32 - 41.0) * 0.01))
        .collect();
    let b_host: Vec<u16> = (0..(seq_len * hv))
        .map(|i| f32_to_f16(((i % 11) as f32 - 5.0) * 0.1))
        .collect();
    let a_host: Vec<u16> = (0..(seq_len * hv))
        .map(|i| f32_to_f16(((i % 17) as f32 - 8.0) * 0.08))
        .collect();
    let a_log = vec![-1.2f32, -0.8f32, -1.5f32, -0.5f32];
    let dt_bias = vec![0.1f32, -0.1f32, 0.05f32, -0.05f32];

    let d_qkv = DeviceBuffer::alloc(0, qkv_host.len() * 2).unwrap();
    let d_b = DeviceBuffer::alloc(0, b_host.len() * 2).unwrap();
    let d_a = DeviceBuffer::alloc(0, a_host.len() * 2).unwrap();
    let d_alog = DeviceBuffer::alloc(0, a_log.len() * 4).unwrap();
    let d_dtb = DeviceBuffer::alloc(0, dt_bias.len() * 4).unwrap();
    d_qkv.copy_from_host(&qkv_host, &stream).unwrap();
    d_b.copy_from_host(&b_host, &stream).unwrap();
    d_a.copy_from_host(&a_host, &stream).unwrap();
    d_alog.copy_from_host(&a_log, &stream).unwrap();
    d_dtb.copy_from_host(&dt_bias, &stream).unwrap();

    // 1. Validate `w34_gdn_causal_conv1d_strided_f16` rejects unsupported `kernel_size = 5` / `0`.
    let pool = StateBundlePool::new(0, &[0], num_slots, conv_dim, 4, hv, dk, dv).unwrap();
    let st = &pool.layers[0];
    let d_qsl = DeviceBuffer::alloc(0, 8).unwrap();
    let d_si = DeviceBuffer::alloc(0, 4).unwrap();
    let d_so = DeviceBuffer::alloc(0, 4).unwrap();
    let d_hi = DeviceBuffer::alloc(0, 1).unwrap();
    d_qsl.copy_from_host(&[0i32, 16i32], &stream).unwrap();
    d_si.copy_from_host(&[0i32], &stream).unwrap();
    d_so.copy_from_host(&[1i32], &stream).unwrap();
    d_hi.copy_from_host(&[0u8], &stream).unwrap();
    let d_conv_w = DeviceBuffer::alloc(0, conv_dim * 5 * 2).unwrap();
    let d_conv_out = DeviceBuffer::alloc(0, seq_len * conv_dim * 2).unwrap();
    let ex0 = stream.exec(0, 0);
    let conv_batch = W34GdnBatch {
        query_start_loc: d_qsl.addr() as *const i32,
        state_in_indices: d_si.addr() as *const i32,
        state_out_indices: d_so.addr() as *const i32,
        has_initial_state: d_hi.addr() as *const u8,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 16,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    for bad_k in [0i32, 5i32] {
        // SAFETY: All device pointers are valid on device 0; `bad_k` tests boundary validation.
        let rc = unsafe {
            ffi::w34_gdn_causal_conv1d_strided_f16(
                &ex0,
                &conv_batch,
                d_qkv.addr() as *const c_void,
                d_conv_w.addr() as *const c_void,
                core::ptr::null(),
                st.conv_state.addr() as *mut c_void,
                d_conv_out.addr() as *mut c_void,
                conv_dim as i32,
                conv_dim as i32,
                bad_k,
                num_slots as i32,
                1,
            )
        };
        assert_ne!(rc, 0, "expected conv1d to reject kernel_size={bad_k}");
    }

    // 2. Duplicate checkpoint offsets `[64, 64]` writing to two distinct destination slots `[2, 3]`
    // must populate BOTH slots in both split-WY (`ws_size > 0`) and chunk64-WY (`ws_size == 0`).
    d_qsl
        .copy_from_host(&[0i32, seq_len as i32], &stream)
        .unwrap();
    let d_ckpt_dst = DeviceBuffer::alloc(0, 8).unwrap();
    let d_ckpt_off = DeviceBuffer::alloc(0, 8).unwrap();
    d_ckpt_dst.copy_from_host(&[2i32, 3i32], &stream).unwrap();
    d_ckpt_off.copy_from_host(&[64i32, 64i32], &stream).unwrap();

    let state_elems = hv * dv * dk;
    let d_ws = DeviceBuffer::alloc(0, 8 * 1024 * 1024).unwrap();
    let d_out = DeviceBuffer::alloc(0, seq_len * hv * dv * 2).unwrap();

    for use_split in [true, false] {
        let zero_ssm = vec![0.0f32; num_slots * state_elems];
        st.ssm_state.copy_from_host(&zero_ssm, &stream).unwrap();
        let ex = if use_split {
            stream.exec(d_ws.addr(), d_ws.len_bytes)
        } else {
            stream.exec(0, 0)
        };
        let batch = W34GdnBatch {
            query_start_loc: d_qsl.addr() as *const i32,
            state_in_indices: d_si.addr() as *const i32,
            state_out_indices: d_so.addr() as *const i32,
            has_initial_state: d_hi.addr() as *const u8,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: d_ckpt_dst.addr() as *const i32,
            checkpoint_token_offsets: d_ckpt_off.addr() as *const i32,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: seq_len as i32,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 2,
        };
        // SAFETY: All device buffers are live and sized for `[seq_len=128, hk=2, hv=4, d=128]`.
        let rc = unsafe {
            ffi::w34_gdn_recurrent_strided_f16(
                &ex,
                &batch,
                d_qkv.addr() as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                d_b.addr() as *const c_void,
                d_a.addr() as *const c_void,
                d_alog.addr() as *const f32,
                d_dtb.addr() as *const f32,
                st.ssm_state.addr() as *mut f32,
                d_out.addr() as *mut c_void,
                hk as i32,
                dk as i32,
                hv as i32,
                dv as i32,
                hv as i32,
                num_slots as i32,
            )
        };
        assert_eq!(rc, 0);
        let mut ssm_got = vec![0.0f32; num_slots * state_elems];
        st.ssm_state.copy_to_host(&mut ssm_got, &stream).unwrap();
        stream.synchronize().unwrap();
        let slot2 = &ssm_got[2 * state_elems..3 * state_elems];
        let slot3 = &ssm_got[3 * state_elems..4 * state_elems];
        let norm2: f32 = slot2.iter().map(|x| x * x).sum();
        assert!(
            norm2 > 1e-4,
            "expected non-zero checkpoint in slot 2 (split={use_split})"
        );
        assert_eq!(
            slot2, slot3,
            "duplicate checkpoint offset must populate both slot 2 and slot 3 (split={use_split})"
        );
    }
}

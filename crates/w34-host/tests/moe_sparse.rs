//! Sparse MoE parity, real-weight M4/M8/M16/M306 validation, shared expert, invalid routes, and TP1/TP2 execution.
#![cfg(feature = "cuda")]
#![allow(clippy::undocumented_unsafe_blocks)]
use std::{ffi::c_void, sync::Mutex};
use w34_host::{
    cache::KvFormatPolicy,
    cuda::{CublasHandle, CudaDevice, CudaStream, DeviceBuffer, f16_to_f32, f32_to_f16},
    loader::resolve_hf_model_dir,
    model::{FfnBlockWeights, Qwen3_5Engine},
};
use w34_kernels::ffi;

unsafe extern "C" {
    fn w34_moe_w4a16_select_regime(
        ex: *const ffi::W34Exec,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;
}

static GPU_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
struct SparseShape {
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    hidden: usize,
    interm: usize,
    group_size: usize,
    concentrated: bool,
    ws_cap: usize,
    expected_regime: i32,
    check_tp2_partition: bool,
}

#[test]
fn sparse_moe_parity_and_partition() {
    let base = SparseShape {
        num_tokens: 1,
        top_k: 8,
        num_experts: 256,
        hidden: 128,
        interm: 256,
        group_size: 128,
        concentrated: false,
        ws_cap: 4 * 1024 * 1024,
        expected_regime: 0,
        check_tp2_partition: true,
    };
    for m in [1, 2, 3] {
        check_sparse(SparseShape {
            num_tokens: m,
            expected_regime: 0,
            ..base
        });
    }
    for m in [4, 7] {
        check_sparse(SparseShape {
            num_tokens: m,
            expected_regime: 1,
            ..base
        });
    }
    // Wide hidden=1024, interm=512 exercises lpr=32 and lpr=16 warp-reduction cascades,
    // and num_experts=32 at M=4, top_k=8 hits the exact np == num_experts boundary.
    check_sparse(SparseShape {
        num_tokens: 4,
        num_experts: 32,
        hidden: 1024,
        interm: 512,
        expected_regime: 1,
        check_tp2_partition: false,
        ..base
    });
    check_sparse(SparseShape {
        num_tokens: 7,
        hidden: 1024,
        interm: 512,
        expected_regime: 1,
        check_tp2_partition: false,
        ..base
    });
    check_sparse(SparseShape {
        num_tokens: 4,
        concentrated: true,
        expected_regime: 1,
        ..base
    });
    // M = 8..128 exercises sparse_mma_8slot_4warp, including exact np == 4 * num_experts at M=128.
    for m in [8, 16, 32, 64, 128] {
        check_sparse(SparseShape {
            num_tokens: m,
            expected_regime: 2,
            check_tp2_partition: m <= 32,
            ..base
        });
    }
    check_sparse(SparseShape {
        num_tokens: 16,
        concentrated: true,
        expected_regime: 2,
        ..base
    });
    // Exceeding M=128 transitions cleanly to grouped_mma (regime 3).
    check_sparse(SparseShape {
        num_tokens: 129,
        expected_regime: 3,
        check_tp2_partition: false,
        ..base
    });
    // Force balanced multi-chunk grouped_mma inside a bounded 256 KiB workspace.
    check_sparse(SparseShape {
        num_tokens: 160,
        ws_cap: 256 * 1024,
        expected_regime: 3,
        check_tp2_partition: false,
        ..base
    });
    // Exercise group_size < 64 (32 and 16) pre-MMA FP16 scaling branch in grouped_mma.
    for gs in [32usize, 16] {
        check_sparse(SparseShape {
            num_tokens: 16,
            num_experts: 64,
            group_size: gs,
            expected_regime: 3,
            ..base
        });
    }
    // Insufficient workspace falls back cleanly to retained_gemv (regime 0) without touching scratch.
    check_sparse(SparseShape {
        num_tokens: 8,
        ws_cap: 128,
        expected_regime: 0,
        check_tp2_partition: false,
        ..base
    });
}

fn round_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

#[allow(clippy::too_many_arguments)]
fn cpu_marlin_moe_ref(
    x: &[u16],
    gu_packed: &[u32],
    gu_scales: &[u16],
    dn_packed: &[u32],
    dn_scales: &[u16],
    topk_ids: &[i32],
    topk_weights: &[f32],
    sh_out: &[u16],
    sh_gate: &[u16],
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    interm: usize,
    hidden: usize,
    group_size: usize,
) -> Vec<f32> {
    let gu_w_stride = hidden / 8;
    let gu_s_stride = hidden / group_size;
    let dn_w_stride = interm / 8;
    let dn_s_stride = interm / group_size;
    let mut out = vec![0.0f32; num_tokens * hidden];
    let mut act = vec![0.0f32; interm];

    for tok in 0..num_tokens {
        let mut slot_accum = vec![0.0f32; hidden];
        for s in 0..top_k {
            let pair = tok * top_k + s;
            let e_i32 = topk_ids[pair];
            if e_i32 < 0 || (e_i32 as usize) >= num_experts {
                continue;
            }
            let e = e_i32 as usize;
            let tw_h = round_f16(topk_weights[pair]);
            for (n, act_n) in act.iter_mut().enumerate() {
                let g_row = e * (2 * interm) + n;
                let u_row = g_row + interm;
                let mut g_acc = 0.0f32;
                let mut u_acc = 0.0f32;
                for w_col in 0..gu_w_stride {
                    let gw = gu_packed[g_row * gu_w_stride + w_col];
                    let uw = gu_packed[u_row * gu_w_stride + w_col];
                    let k0 = w_col * 8;
                    let gs = f16_to_f32(gu_scales[g_row * gu_s_stride + k0 / group_size]);
                    let us = f16_to_f32(gu_scales[u_row * gu_s_stride + k0 / group_size]);
                    for b in 0..8 {
                        let xv = f16_to_f32(x[tok * hidden + k0 + b]);
                        let gn = (((gw >> (4 * b)) & 0xF) as i32 - 8) as f32;
                        let un = (((uw >> (4 * b)) & 0xF) as i32 - 8) as f32;
                        g_acc += round_f16(gn * gs) * xv;
                        u_acc += round_f16(un * us) * xv;
                    }
                }
                let gh = round_f16(g_acc);
                let uh = round_f16(u_acc);
                let silu_h = round_f16(gh / (1.0 + (-gh).exp()));
                *act_n = round_f16(silu_h * uh);
            }
            for (col, slot_col) in slot_accum.iter_mut().enumerate() {
                let d_row = e * hidden + col;
                let mut d_acc = 0.0f32;
                for w_col in 0..dn_w_stride {
                    let dw = dn_packed[d_row * dn_w_stride + w_col];
                    let k0 = w_col * 8;
                    let ds = f16_to_f32(dn_scales[d_row * dn_s_stride + k0 / group_size]);
                    for b in 0..8 {
                        let av = act[k0 + b];
                        let dn = (((dw >> (4 * b)) & 0xF) as i32 - 8) as f32;
                        d_acc += round_f16(dn * ds) * av;
                    }
                }
                let weighted_h = round_f16(round_f16(d_acc) * tw_h);
                *slot_col += weighted_h;
            }
        }
        let gate_f = f16_to_f32(sh_gate[tok]);
        let sig_h = round_f16(1.0 / (1.0 + (-gate_f).exp()));
        for col in 0..hidden {
            let routed_h = round_f16(slot_accum[col]);
            let sh_h = round_f16(sig_h * f16_to_f32(sh_out[tok * hidden + col]));
            out[tok * hidden + col] = round_f16(routed_h + sh_h);
        }
    }
    out
}

fn check_sparse(shape: SparseShape) {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();

    let SparseShape {
        num_tokens,
        top_k,
        num_experts,
        hidden,
        interm,
        group_size,
        concentrated,
        ws_cap,
        expected_regime,
        check_tp2_partition,
    } = shape;

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
            topk_ids[t * top_k + s] = if concentrated { s as i32 } else { e };
            let w = 1.0f32 / ((s + 1) as f32);
            topk_weights[t * top_k + s] = w;
            w_sum += w;
        }
        for s in 0..top_k {
            topk_weights[t * top_k + s] /= w_sum;
        }
    }

    // Invalid routes must not leave observable uninitialized stage2 values.
    topk_ids[0] = -1;
    topk_ids[1] = num_experts as i32;

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

    // Reference (`scratch_ptr = 0` -> warp-per-row GEMV) vs dispatched (`d_ws`), plus
    // independent host-side donor-Marlin FP16 reference and sentinel workspace check.
    let sentinel = vec![0xA5u8; ws_cap];
    d_ws.copy_from_host(&sentinel, &stream).unwrap();
    let ex_ref = stream.exec(0, 0);
    let ex_mma = stream.exec(d_ws.addr(), ws_cap);
    assert_eq!(
        unsafe {
            w34_moe_w4a16_select_regime(
                &ex_ref,
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
    let actual_regime = unsafe {
        w34_moe_w4a16_select_regime(
            &ex_mma,
            num_tokens as i32,
            top_k as i32,
            num_experts as i32,
            interm as i32,
            hidden as i32,
            group_size as i32,
        )
    };
    assert_eq!(
        actual_regime, expected_regime,
        "Unexpected MoE dispatch regime for M={num_tokens} gs={group_size} ws={ws_cap}"
    );
    for (ex, d_interm, d_out) in [
        (&ex_ref, &d_interm_ref, &d_out_ref),
        (&ex_mma, &d_interm_mma, &d_out_mma),
    ] {
        assert_eq!(
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

    let mut ws_after = vec![0u8; ws_cap.min(64)];
    assert_eq!(
        unsafe {
            ffi::w34_memcpy_d2h_async(
                ws_after.as_mut_ptr() as *mut c_void,
                d_ws.addr() as *const c_void,
                ws_after.len(),
                stream.handle,
            )
        },
        0
    );
    stream.synchronize().unwrap();
    if expected_regime == 0 {
        assert!(
            ws_after.iter().all(|&b| b == 0xA5),
            "retained_gemv must not touch scratch workspace (M={num_tokens})"
        );
    } else {
        assert!(
            ws_after.iter().any(|&b| b != 0xA5),
            "non-GEMV regime {expected_regime} failed to write scratch workspace (M={num_tokens})"
        );
    }

    let mut h_ref = vec![0u16; num_tokens * hidden];
    let mut h_mma = vec![0u16; num_tokens * hidden];
    d_out_ref.copy_to_host(&mut h_ref, &stream).unwrap();
    d_out_mma.copy_to_host(&mut h_mma, &stream).unwrap();

    let cpu_ref = cpu_marlin_moe_ref(
        &x_host,
        &gu_packed,
        &gu_scales,
        &dn_packed,
        &dn_scales,
        &topk_ids,
        &topk_weights,
        &sh_out_host,
        &sh_gate_host,
        num_tokens,
        top_k,
        num_experts,
        interm,
        hidden,
        group_size,
    );
    for (label, gpu_out, rel_cap) in [("gemv", &h_ref, 2e-4f64), ("dispatched", &h_mma, 1e-3f64)] {
        let mut max_diff = 0.0f32;
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&c, &g) in cpu_ref.iter().zip(gpu_out.iter()) {
            let gf = f16_to_f32(g);
            let d = (c - gf).abs();
            max_diff = max_diff.max(d);
            diff_sq += f64::from(d).powi(2);
            ref_sq += f64::from(c).powi(2);
        }
        let rel_l2 = diff_sq.sqrt() / ref_sq.sqrt().max(1e-12);
        assert!(
            max_diff < 2e-3 && rel_l2 < rel_cap,
            "MoE {label} vs independent CPU Marlin FP16 ref drift at M={num_tokens} gs={group_size}: max_diff={max_diff} rel_l2={rel_l2}"
        );
    }

    if !check_tp2_partition {
        return;
    }

    // Sequential single-GPU TP intermediate partition check; no collectives.
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
            // Copy separate gate and up row partitions.
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

#[test]
fn real_weight_sparse_moe_m4_m8_m16_and_actual_tp1_tp2() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let model = "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4";
    if resolve_hf_model_dir(model).is_err() {
        return;
    }

    // 1. Validate real layer-0 MoE weights (including live router, shared expert, and shared gate)
    // across M = 4 (vec4), M = 8 & 16 & 64 (sparse MMA), and M = 306 (3-chunk Grouped MMA in 8 MiB scratch).
    let prompts_tp1 = [
        vec![151644, 872, 198, 785],
        vec![151644, 872, 198, 785, 6722, 315, 9625, 374],
        vec![
            151644, 872, 198, 785, 6722, 315, 9625, 374, 13, 151645, 198, 151644, 77091, 198,
            151667, 198,
        ],
    ];
    let mut tp1_l0 = Vec::new();
    let mut tp1_norms = Vec::new();
    let mut tp1_tokens = Vec::new();
    {
        let mut engine =
            Qwen3_5Engine::load(model, 1, KvFormatPolicy::Fp16, Some(32), true).unwrap();
        {
            let ffn = engine.ranks[0]
                .layers
                .iter()
                .find_map(|l| match &l.ffn {
                    FfnBlockWeights::Moe(_) => Some(&l.ffn),
                    FfnBlockWeights::Dense { .. } => None,
                })
                .expect("35B-A3B checkpoint must have MoE layers");
            let FfnBlockWeights::Moe(moe) = ffn else {
                unreachable!()
            };
            let h = engine.cfg.hidden_size;
            let k = moe.top_k;
            let e = moe.num_experts;
            let n_moe = moe.local_moe_interm;
            let n_sh = moe.local_shared_interm.max(8);
            let max_m = 306usize;

            let _dev = CudaDevice::set_current(0).unwrap();
            let stream = CudaStream::new(0).unwrap();
            let blas = CublasHandle::new(&stream).unwrap();
            let marlin_ints = (h.max(n_sh * 2).div_ceil(64) * 16).max(1024);
            let d_marlin = DeviceBuffer::alloc(0, marlin_ints * 4).unwrap();
            d_marlin.zero_async(&stream).unwrap();
            let d_x = DeviceBuffer::alloc(0, max_m * h * 2).unwrap();
            let d_out_ref = DeviceBuffer::alloc(0, max_m * h * 2).unwrap();
            let d_out_opt = DeviceBuffer::alloc(0, max_m * h * 2).unwrap();
            let d_gu = DeviceBuffer::alloc(0, 64).unwrap();
            let d_act = DeviceBuffer::alloc(0, 64).unwrap();
            let d_logits = DeviceBuffer::alloc(0, max_m * e * 2).unwrap();
            let d_tw = DeviceBuffer::alloc(0, max_m * k * 4).unwrap();
            let d_ids = DeviceBuffer::alloc(0, max_m * k * 4).unwrap();
            let d_interm = DeviceBuffer::alloc(0, max_m * k * n_moe * 2).unwrap();
            let d_sh_gu = DeviceBuffer::alloc(0, max_m * n_sh * 2 * 2).unwrap();
            let d_sh_act = DeviceBuffer::alloc(0, max_m * n_sh * 2).unwrap();
            let d_sh_out = DeviceBuffer::alloc(0, max_m * h * 2).unwrap();
            let d_sh_gate = DeviceBuffer::alloc(0, max_m * 2).unwrap();
            let ws_cap = 8 * 1024 * 1024; // Exact production scratch ceiling
            let d_ws = DeviceBuffer::alloc(0, ws_cap).unwrap();

            let ex_ref = stream.exec(0, 0);
            let ex_opt = stream.exec(d_ws.addr(), ws_cap);

            for (m, expected_regime) in [(4usize, 1i32), (8, 2), (16, 2), (64, 2), (306, 3)] {
                let got_regime = unsafe {
                    w34_moe_w4a16_select_regime(
                        &ex_opt,
                        m as i32,
                        k as i32,
                        e as i32,
                        n_moe as i32,
                        h as i32,
                        128,
                    )
                };
                assert_eq!(
                    got_regime, expected_regime,
                    "Real-weight MoE M={m} selected unexpected regime {got_regime}"
                );
                let mut x_host = Vec::with_capacity(m * h);
                for row in 0..m {
                    for col in 0..h {
                        let v = (((row * 251 + col * 31 + 7) % 97) as f32 - 48.0) * 0.015;
                        x_host.push(f32_to_f16(v));
                    }
                }
                d_x.copy_from_host(&x_host, &stream).unwrap();
                for (ex, d_out) in [(&ex_ref, &d_out_ref), (&ex_opt, &d_out_opt)] {
                    ffn.forward(
                        ex,
                        &blas,
                        d_marlin.addr(),
                        marlin_ints,
                        d_x.addr(),
                        d_out.addr(),
                        m,
                        h,
                        0,
                        d_gu.addr(),
                        d_act.addr(),
                        d_logits.addr(),
                        d_tw.addr(),
                        d_ids.addr(),
                        d_interm.addr(),
                        d_sh_gu.addr(),
                        d_sh_act.addr(),
                        d_sh_out.addr(),
                        d_sh_gate.addr(),
                    )
                    .unwrap();
                }
                let mut h_ref = vec![0u16; m * h];
                let mut h_opt = vec![0u16; m * h];
                d_out_ref.copy_to_host(&mut h_ref, &stream).unwrap();
                d_out_opt.copy_to_host(&mut h_opt, &stream).unwrap();
                let mut max_abs = 0.0f32;
                let mut diff_sq = 0.0f64;
                let mut ref_sq = 0.0f64;
                for (&r, &o) in h_ref.iter().zip(&h_opt) {
                    let rf = f16_to_f32(r);
                    let of = f16_to_f32(o);
                    assert!(rf.is_finite() && of.is_finite());
                    let d = (rf - of).abs();
                    max_abs = max_abs.max(d);
                    diff_sq += f64::from(d).powi(2);
                    ref_sq += f64::from(rf).powi(2);
                }
                let rel_l2 = diff_sq.sqrt() / ref_sq.sqrt().max(1e-12);
                assert!(
                    max_abs < 2e-3 && rel_l2 < 5e-4,
                    "Real-weight MoE M={m} drift too high: max_abs={max_abs} rel_l2={rel_l2}"
                );
            }
        }

        // Captured real-image vLLM 0.31.0 layer-0 M=8 parity gate (`rel_l2 <= 0.005`).
        {
            use sha2::{Digest, Sha256};
            use w34_host::loader::SafetensorsArchive;
            let golden_dir =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle/goldens");
            let meta: serde_json::Value = serde_json::from_slice(
                &std::fs::read(golden_dir.join("vllm_moe_layer0_m8.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(meta["vllm_version"], "0.31.0");
            let cfg_sha = format!(
                "{:x}",
                Sha256::digest(std::fs::read(engine.model_dir.join("config.json")).unwrap())
            );
            assert_eq!(cfg_sha, meta["config_sha256"].as_str().unwrap());
            let st_path = golden_dir.join(meta["safetensors_file"].as_str().unwrap());
            let st_sha = format!("{:x}", Sha256::digest(std::fs::read(&st_path).unwrap()));
            assert_eq!(st_sha, meta["safetensors_sha256"].as_str().unwrap());
            let archive = SafetensorsArchive::open(&st_path).unwrap();
            let limit = meta["layer0_rel_l2_limit"].as_f64().unwrap();
            assert!(limit.is_finite() && limit > 0.0);
            assert!(engine.verify_moe_layer0_golden(&archive, 0.0).is_err());
            assert!(engine.verify_moe_layer0_golden(&archive, f64::NAN).is_err());
            let rep = engine.verify_moe_layer0_golden(&archive, limit).unwrap();
            assert!(
                rep.passed,
                "Layer-0 MoE vLLM 0.31.0 real-image parity failed: {rep:?}"
            );
            assert_eq!(rep.router_slot_matches, rep.router_total_slots);
            let strict_fail = engine.verify_moe_layer0_golden(&archive, 1e-9).unwrap();
            assert!(
                !strict_fail.passed,
                "Layer-0 MoE parity check must fail closed on impossible threshold: {strict_fail:?}"
            );
        }

        if CudaDevice::count().unwrap_or(0) < 2 {
            return;
        }
        for p in &prompts_tp1 {
            let dump = engine.dump_prompt(p).unwrap();
            tp1_l0.push(dump.tensors_f32["layer_0_hidden_after"].1.clone());
            tp1_norms.push(dump.tensors_f32["final_norm"].1.clone());
            let m = engine.generate_with_radix_cache(p, 4, 64, false).unwrap();
            tp1_tokens.push(m.generated_token_ids);
        }
    }

    // 2. Actual two-GPU TP=2 execution across M=4, M=8, and M=16 MoE prefill + decode.
    let mut engine_tp2 =
        Qwen3_5Engine::load(model, 2, KvFormatPolicy::Fp16, Some(32), true).unwrap();
    {
        let st_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/goldens/vllm_moe_layer0_m8.safetensors");
        let archive = w34_host::loader::SafetensorsArchive::open(&st_path).unwrap();
        assert!(
            engine_tp2
                .verify_moe_layer0_golden(&archive, 0.005)
                .is_err()
        );
    }
    let rel_l2_fn = |a: &[f32], b: &[f32]| -> f64 {
        assert_eq!(a.len(), b.len());
        let err_sq: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        let ref_sq: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
        (err_sq / ref_sq.max(1e-30)).sqrt()
    };
    for (idx, p) in prompts_tp1.iter().enumerate() {
        let dump2 = engine_tp2.dump_prompt(p).unwrap();
        let l0_rel = rel_l2_fn(&tp1_l0[idx], &dump2.tensors_f32["layer_0_hidden_after"].1);
        let fn_rel = rel_l2_fn(&tp1_norms[idx], &dump2.tensors_f32["final_norm"].1);
        assert!(
            l0_rel < 1e-3,
            "Actual 35B-A3B MoE TP=1 vs TP=2 layer_0 drift at M={}: l0={l0_rel}",
            p.len()
        );
        assert!(
            fn_rel < 0.10,
            "Actual 35B-A3B MoE TP=1 vs TP=2 M={} final_norm rel_l2={fn_rel} >= 0.10",
            p.len()
        );
        let m2 = engine_tp2
            .generate_with_radix_cache(p, 4, 64, false)
            .unwrap();
        assert_eq!(
            tp1_tokens[idx],
            m2.generated_token_ids,
            "Actual 35B-A3B MoE TP=1 vs TP=2 greedy mismatch at M={}",
            p.len()
        );
    }
}

#[test]
fn dual_small_n_gemm_and_warp256_router_match_references() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();
    let ex = stream.exec(0, 0);

    // 1. Fused dual small-N GEMM (`N0 = 256` router + `N1 = 1` shared gate) must be 0-ULP
    // bit-identical to two separate `w34_small_n_gemm_f16` launches across M = 1, 4, 8.
    let size_k = 2048usize;
    let n0 = 256usize;
    let n1 = 1usize;
    let w0_host: Vec<u16> = (0..n0 * size_k)
        .map(|i| f32_to_f16(((i % 67) as f32 - 33.0) * 0.015625))
        .collect();
    let w1_host: Vec<u16> = (0..n1 * size_k)
        .map(|i| f32_to_f16(((i % 41) as f32 - 20.0) * 0.03125))
        .collect();
    let d_w0 = DeviceBuffer::alloc(dev.device_id, w0_host.len() * 2).unwrap();
    let d_w1 = DeviceBuffer::alloc(dev.device_id, w1_host.len() * 2).unwrap();
    d_w0.copy_from_host(&w0_host, &stream).unwrap();
    d_w1.copy_from_host(&w1_host, &stream).unwrap();

    let mut ref0_m23 = Vec::new();
    let mut ref1_m23 = Vec::new();
    for m in [23usize, 17, 16, 8, 7, 4, 1] {
        let x_host: Vec<u16> = (0..m * size_k)
            .map(|i| f32_to_f16(((i % 89) as f32 - 44.0) * 0.02))
            .collect();
        let d_x = DeviceBuffer::alloc(dev.device_id, x_host.len() * 2).unwrap();
        d_x.copy_from_host(&x_host, &stream).unwrap();
        let d_ref0 = DeviceBuffer::alloc(dev.device_id, m * n0 * 2).unwrap();
        let d_ref1 = DeviceBuffer::alloc(dev.device_id, m * n1 * 2).unwrap();
        let d_got0 = DeviceBuffer::alloc(dev.device_id, m * n0 * 2).unwrap();
        let d_got1 = DeviceBuffer::alloc(dev.device_id, m * n1 * 2).unwrap();

        assert_eq!(
            unsafe {
                ffi::w34_small_n_gemm_f16(
                    &ex,
                    d_x.addr() as *const c_void,
                    d_w0.addr() as *const c_void,
                    d_ref0.addr() as *mut c_void,
                    m as i32,
                    n0 as i32,
                    size_k as i32,
                )
            },
            0
        );
        assert_eq!(
            unsafe {
                ffi::w34_small_n_gemm_f16(
                    &ex,
                    d_x.addr() as *const c_void,
                    d_w1.addr() as *const c_void,
                    d_ref1.addr() as *mut c_void,
                    m as i32,
                    n1 as i32,
                    size_k as i32,
                )
            },
            0
        );
        assert_eq!(
            unsafe {
                ffi::w34_small_n_gemm_dual_f16(
                    &ex,
                    d_x.addr() as *const c_void,
                    d_w0.addr() as *const c_void,
                    d_got0.addr() as *mut c_void,
                    n0 as i32,
                    d_w1.addr() as *const c_void,
                    d_got1.addr() as *mut c_void,
                    n1 as i32,
                    m as i32,
                    size_k as i32,
                )
            },
            0
        );
        let mut ref0 = vec![0u16; m * n0];
        let mut ref1 = vec![0u16; m * n1];
        let mut got0 = vec![0u16; m * n0];
        let mut got1 = vec![0u16; m * n1];
        d_ref0.copy_to_host(&mut ref0, &stream).unwrap();
        d_ref1.copy_to_host(&mut ref1, &stream).unwrap();
        d_got0.copy_to_host(&mut got0, &stream).unwrap();
        d_got1.copy_to_host(&mut got1, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(got0, ref0, "dual_small_n out0 0-ULP mismatch at M={m}");
        assert_eq!(got1, ref1, "dual_small_n out1 0-ULP mismatch at M={m}");
        if m == 23 {
            ref0_m23 = ref0;
            ref1_m23 = ref1;
        } else {
            assert_eq!(
                &ref0[..],
                &ref0_m23[..m * n0],
                "small_n_gemm_f16 out0 0-ULP M-invariance mismatch at M={m} vs M=23"
            );
            assert_eq!(
                &ref1[..],
                &ref1_m23[..m * n1],
                "small_n_gemm_f16 out1 0-ULP M-invariance mismatch at M={m} vs M=23"
            );
        }
    }

    // 2. Single-warp `moe_topk_router_warp256_f16` (`num_experts = 256, top_k = 8`) vs
    // `cta128` (`num_experts = 257`, last column = -65504 so top-8 selection is identical).
    let rows = 5usize;
    let top_k = 8usize;
    let mut logits_256 = vec![0u16; rows * 256];
    let mut logits_257 = vec![0u16; rows * 257];
    for r in 0..rows {
        for e in 0..256 {
            let v = (((r * 73 + e * 19) % 512) as f32 - 256.0) * 0.03125;
            let bits = f32_to_f16(v);
            logits_256[r * 256 + e] = bits;
            logits_257[r * 257 + e] = bits;
        }
        logits_257[r * 257 + 256] = f32_to_f16(-65504.0);
    }
    let d_l256 = DeviceBuffer::alloc(dev.device_id, logits_256.len() * 2).unwrap();
    let d_l257 = DeviceBuffer::alloc(dev.device_id, logits_257.len() * 2).unwrap();
    d_l256.copy_from_host(&logits_256, &stream).unwrap();
    d_l257.copy_from_host(&logits_257, &stream).unwrap();
    let d_w256 = DeviceBuffer::alloc(dev.device_id, rows * top_k * 4).unwrap();
    let d_w257 = DeviceBuffer::alloc(dev.device_id, rows * top_k * 4).unwrap();
    let d_i256 = DeviceBuffer::alloc(dev.device_id, rows * top_k * 4).unwrap();
    let d_i257 = DeviceBuffer::alloc(dev.device_id, rows * top_k * 4).unwrap();

    for norm_topk in [0i32, 1] {
        for use_sigmoid in [0i32, 1] {
            // For softmax (`use_sigmoid == 0`), an extra `-65504` column adds exp(-65504) == 0.0f in FP32!
            assert_eq!(
                unsafe {
                    ffi::w34_moe_topk_router_f16(
                        &ex,
                        d_l256.addr() as *const c_void,
                        d_w256.addr() as *mut f32,
                        d_i256.addr() as *mut i32,
                        rows as i32,
                        256,
                        top_k as i32,
                        norm_topk,
                        use_sigmoid,
                    )
                },
                0
            );
            assert_eq!(
                unsafe {
                    ffi::w34_moe_topk_router_f16(
                        &ex,
                        d_l257.addr() as *const c_void,
                        d_w257.addr() as *mut f32,
                        d_i257.addr() as *mut i32,
                        rows as i32,
                        257,
                        top_k as i32,
                        norm_topk,
                        use_sigmoid,
                    )
                },
                0
            );
            let mut ids256 = vec![0i32; rows * top_k];
            let mut ids257 = vec![0i32; rows * top_k];
            let mut w256 = vec![0f32; rows * top_k];
            let mut w257 = vec![0f32; rows * top_k];
            d_i256.copy_to_host(&mut ids256, &stream).unwrap();
            d_i257.copy_to_host(&mut ids257, &stream).unwrap();
            d_w256.copy_to_host(&mut w256, &stream).unwrap();
            d_w257.copy_to_host(&mut w257, &stream).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(
                ids256, ids257,
                "router top-k IDs mismatch (norm={norm_topk}, sig={use_sigmoid})"
            );
            for (a, b) in w256.iter().zip(w257.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "router weight mismatch {a} vs {b} (norm={norm_topk}, sig={use_sigmoid})"
                );
            }
        }
    }
}

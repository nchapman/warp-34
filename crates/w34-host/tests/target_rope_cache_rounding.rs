//! Native installed vLLM fused_qk_norm_rope regression; no model execution.
#![cfg(feature = "cuda")]
use std::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer};
use w34_kernels::ffi;

#[test]
fn target_rope_uses_native_half_cache_coefficients() {
    // GPU1 capture: installed vLLM model_executor/layers/fused_qk_norm_rope.py.
    // get_rope with local Qwen/Qwen3.5-4B config (head_dim=256,
    // max_position_embeddings=262144, rope_theta=1e7, partial_rotary_factor=.25,
    // mrope_interleaved=true, mrope_section=[11,11,10]), under
    // set_current_vllm_config(VllmConfig()), torch default dtype float16.
    // Runtime: MRotaryEmbedding.dtype AND cos_sin_cache.dtype are float16.
    // q_gate=[ones(256), .5*ones(256)], k=-ones(256), weights=zeros(256),
    // positions=int64([[31],[47],[63]]), eps=1e-6, norm_beta=1.0.
    // Native fused outputs below are uint16 views, not a Python RoPE oracle.
    // RMSNorm rounds back to +/-1, isolating coefficient quantization.
    let native_q: [u16; 64] = [
        15686, 47850, 13606, 13596, 15376, 15658, 47999, 47710, 46961, 14578, 14650, 14797, 15204,
        15217, 15245, 15327, 15330, 15335, 15353, 15353, 15355, 15358, 15359, 15359, 15360, 15360,
        15360, 15360, 15360, 15360, 15360, 15360, 14358, 48250, 48514, 15747, 15328, 47264, 15421,
        15533, 15704, 15638, 15620, 15580, 15432, 15426, 15414, 15377, 15375, 15372, 15364, 15363,
        15363, 15361, 15361, 15361, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360,
    ];
    check_native_rope(&native_q, &[15360; 256], [31, 47, 63], 15360);
}

#[test]
fn target_rope_preserves_native_small_angle_coefficients() {
    // Same installed native fused operator as above, with positions [1,1,1]
    // and Q[32:64]=0. This exposes small sin terms hidden by all-ones inputs.
    let native_q = [
        14496, 15114, 15358, 15404, 15421, 15427, 15430, 15430, 15431, 15431, 15431, 15431, 15431,
        15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431,
        15431, 15431, 15431, 15431, 15431, 15431, 15154, 14557, 13852, 13181, 12429, 11650, 10922,
        10247, 9438, 8674, 7964, 7244, 6449, 5703, 5014, 4245, 3466, 2738, 2060, 1252, 756, 458,
        276, 167, 101, 61, 36, 22, 14, 9, 5, 3,
    ];
    let mut q = [15360; 256];
    q[32..64].fill(0);
    check_native_rope(&native_q, &q, [1, 1, 1], 15431);
    // Near a sine zero, pow(theta, -exponent) and the native reciprocal of
    // pow(theta, exponent) can produce distinct FP16 cache coefficients.
    let native_q = [
        15160, 14755, 48199, 48009, 15180, 48142, 15138, 13605, 14892, 15422, 15389, 15415, 15430,
        15429, 15430, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431, 15431,
        15431, 15431, 15431, 15431, 15431, 15431, 47255, 14960, 7678, 14349, 47223, 46448, 14520,
        15380, 14828, 12390, 13495, 12735, 10186, 11319, 10521, 7904, 9074, 8320, 5650, 6802, 6129,
        3420, 4557, 3842, 1210, 2335, 1584, 267, 579, 350, 59, 127,
    ];
    check_native_rope(&native_q, &q, [12, 43, 43], 15431);
}

fn check_native_rope(native_q: &[u16; 64], q: &[u16; 256], planes: [i32; 3], unrotated: u16) {
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let ex = stream.exec(0, 0);
    let upload = |bits: &[u16]| {
        let b = DeviceBuffer::alloc(device.device_id, bits.len() * 2).unwrap();
        b.copy_from_host(bits, &stream).unwrap();
        b
    };
    for m in [1usize, 16] {
        let mut qg = Vec::new();
        for _ in 0..m {
            qg.extend_from_slice(q);
            qg.extend_from_slice(&[14336; 256]);
        }
        let qg = upload(&qg);
        let k = upload(&q.iter().map(|v| v ^ 0x8000).collect::<Vec<_>>().repeat(m));
        let w = upload(&[0; 256]);
        let qout = upload(&vec![0; m * 256]);
        let kout = upload(&vec![0; m * 256]);
        let gout = upload(&vec![0; m * 256]);
        let positions: Vec<i32> = planes.into_iter().flat_map(|p| vec![p; m]).collect();
        let pos = DeviceBuffer::alloc(device.device_id, positions.len() * 4).unwrap();
        pos.copy_from_host(&positions, &stream).unwrap();
        // SAFETY: buffers sized for m rows, one Q/K head, 256 channels;
        // positions have three planes of stride m. All live until synchronization.
        let rc = unsafe {
            ffi::w34_fused_qk_rmsnorm_rope_gate_f16(
                &ex,
                qg.addr() as *const c_void,
                k.addr() as *const c_void,
                w.addr() as *const c_void,
                w.addr() as *const c_void,
                pos.addr() as *const i32,
                core::ptr::null(),
                qout.addr() as *mut c_void,
                kout.addr() as *mut c_void,
                gout.addr() as *mut c_void,
                m as i32,
                m as i32,
                1,
                1,
                256,
                64,
                11,
                11,
                10,
                10_000_000.0,
                1e-6,
            )
        };
        assert_eq!(rc, 0);
        for (name, buffer) in [("q", qout), ("k", kout), ("gate", gout)] {
            let mut got = vec![0u16; m * 256];
            buffer.copy_to_host(&mut got, &stream).unwrap();
            stream.synchronize().unwrap();
            for row in 0..m {
                for i in 0..256 {
                    let expected = if name == "gate" {
                        14336
                    } else {
                        let q = if i < 64 { native_q[i] } else { unrotated };
                        if name == "k" { q ^ 0x8000 } else { q }
                    };
                    assert_eq!(
                        got[row * 256 + i],
                        expected,
                        "M={m} {name} row={row} channel={i}"
                    );
                }
            }
        }
    }
}

#[test]
fn target_rope_vectorized_fast_path_matches_unaligned_fallback() {
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let ex = stream.exec(0, 0);

    let m = 7usize;
    let q_heads = 4usize;
    let kv_heads = 2usize;
    let hd = 256usize;
    let pad = 1usize; // +2B offset forces scalar fallback on inputs & outputs
    let mut qg = vec![0u16; pad + m * q_heads * 2 * hd];
    for (i, slot) in qg[pad..].iter_mut().enumerate() {
        *slot = (14000u16).wrapping_add(((i * 37) % 1024) as u16);
    }
    let mut k = vec![0u16; pad + m * kv_heads * hd];
    for (i, slot) in k[pad..].iter_mut().enumerate() {
        *slot = (46800u16).wrapping_add(((i * 53) % 1024) as u16);
    }
    let mut qw = vec![0u16; pad + hd];
    let mut kw = vec![0u16; pad + hd];
    for i in 0..hd {
        qw[pad + i] = (12000u16).wrapping_add(((i * 19) % 512) as u16);
        kw[pad + i] = (12100u16).wrapping_add(((i * 23) % 512) as u16);
    }
    let positions: Vec<i32> = (0..3 * m).map(|i| (i * 11 + 5) as i32).collect();
    let d_pos = DeviceBuffer::alloc(device.device_id, positions.len() * 4).unwrap();
    d_pos.copy_from_host(&positions, &stream).unwrap();

    let run = |off: usize| -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let d_qg = DeviceBuffer::alloc(device.device_id, qg.len() * 2).unwrap();
        let d_k = DeviceBuffer::alloc(device.device_id, k.len() * 2).unwrap();
        let d_qw = DeviceBuffer::alloc(device.device_id, qw.len() * 2).unwrap();
        let d_kw = DeviceBuffer::alloc(device.device_id, kw.len() * 2).unwrap();
        let d_qo = DeviceBuffer::alloc(device.device_id, (pad + m * q_heads * hd) * 2).unwrap();
        let d_ko = DeviceBuffer::alloc(device.device_id, (pad + m * kv_heads * hd) * 2).unwrap();
        let d_go = DeviceBuffer::alloc(device.device_id, (pad + m * q_heads * hd) * 2).unwrap();
        d_qg.copy_from_host(&qg[pad - off..], &stream).unwrap();
        d_k.copy_from_host(&k[pad - off..], &stream).unwrap();
        d_qw.copy_from_host(&qw[pad - off..], &stream).unwrap();
        d_kw.copy_from_host(&kw[pad - off..], &stream).unwrap();
        let bytes = (off * 2) as u64;
        // SAFETY: All device pointers shifted by `bytes` remain in-bounds for `(m, q_heads, kv_heads, hd)` until `synchronize()`.
        let rc = unsafe {
            ffi::w34_fused_qk_rmsnorm_rope_gate_f16(
                &ex,
                (d_qg.addr() + bytes) as *const c_void,
                (d_k.addr() + bytes) as *const c_void,
                (d_qw.addr() + bytes) as *const c_void,
                (d_kw.addr() + bytes) as *const c_void,
                d_pos.addr() as *const i32,
                core::ptr::null(),
                (d_qo.addr() + bytes) as *mut c_void,
                (d_ko.addr() + bytes) as *mut c_void,
                (d_go.addr() + bytes) as *mut c_void,
                m as i32,
                m as i32,
                q_heads as i32,
                kv_heads as i32,
                hd as i32,
                64,
                11,
                11,
                10,
                10_000_000.0,
                1e-6,
            )
        };
        assert_eq!(rc, 0);
        let mut qo = vec![0u16; pad + m * q_heads * hd];
        let mut ko = vec![0u16; pad + m * kv_heads * hd];
        let mut go = vec![0u16; pad + m * q_heads * hd];
        d_qo.copy_to_host(&mut qo, &stream).unwrap();
        d_ko.copy_to_host(&mut ko, &stream).unwrap();
        d_go.copy_to_host(&mut go, &stream).unwrap();
        stream.synchronize().unwrap();
        (
            qo[off..off + m * q_heads * hd].to_vec(),
            ko[off..off + m * kv_heads * hd].to_vec(),
            go[off..off + m * q_heads * hd].to_vec(),
        )
    };
    let (qo_fast, ko_fast, go_fast) = run(0);
    let (qo_scal, ko_scal, go_scal) = run(pad);
    assert_eq!(
        qo_fast, qo_scal,
        "vectorized q_out must match scalar fallback"
    );
    assert_eq!(
        ko_fast, ko_scal,
        "vectorized k_out must match scalar fallback"
    );
    assert_eq!(
        go_fast, go_scal,
        "vectorized gate_out must match scalar fallback"
    );
}

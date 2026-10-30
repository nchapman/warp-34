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
            qg.extend_from_slice(&[15360; 256]);
            qg.extend_from_slice(&[14336; 256]);
        }
        let qg = upload(&qg);
        let k = upload(&vec![48128; m * 256]);
        let w = upload(&[0; 256]);
        let qout = upload(&vec![0; m * 256]);
        let kout = upload(&vec![0; m * 256]);
        let gout = upload(&vec![0; m * 256]);
        let positions: Vec<i32> = [31, 47, 63].into_iter().flat_map(|p| vec![p; m]).collect();
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
                        let q = if i < 64 { native_q[i] } else { 15360 };
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

//! Native vLLM 0.31.0 Gemma residual boundary regression (GPU-only, no model).
#![cfg(feature = "cuda")]
use std::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, f32_to_f16};
use w34_kernels::ffi;

#[test]
fn gemma_normalizes_unrounded_sum_but_stores_half_residual() {
    // Captured on GPU1 from installed vLLM 0.31.0 under
    // set_current_vllm_config(VllmConfig()): GemmaRMSNorm(8).cuda().half(),
    // eps=1e-6. Both module(x, residual) and forward_cuda(x, residual) agree.
    // x=[1,1,2,-2,.5,-.5,4,-4]; residual=[2^-11,3*2^-11,2^-10,
    // -3*2^-10,2^-12,3*2^-12,2^-9,3*2^-9]; w=[0,.25,-.25,.5,-.5,.125,-.125,1].
    // Native output differs from rounding either numerator OR variance input.
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let ex = stream.exec(0, 0);
    let upload = |bits: &[u16]| {
        let buffer = DeviceBuffer::alloc(device.device_id, bits.len() * 2).unwrap();
        buffer.copy_from_host(bits, &stream).unwrap();
        buffer
    };
    let x = upload(&[15360, 15360, 16384, 49152, 14336, 47104, 17408, 50176]);
    let weight = upload(&[0, 13312, 46080, 14336, 47104, 12288, 45056, 15360]);
    for with_tap in [false, true] {
        let residual = upload(&[4096, 5632, 5120, 39424, 3072, 4608, 6144, 7680]);
        let output = upload(&[0; 8]);
        let tap = upload(&[0; 10]);
        // SAFETY: all buffers remain live through synchronization; x, residual,
        // weight and output hold 8 halves. Tap holds stride 10 with offset 1,
        // leaving room for 8 BF16 values. Null mask means this row is valid.
        let rc = unsafe {
            if with_tap {
                ffi::w34_fused_add_gemma_rms_norm_tap_f16(
                    &ex,
                    x.addr() as *const c_void,
                    residual.addr() as *mut c_void,
                    weight.addr() as *const c_void,
                    output.addr() as *mut c_void,
                    tap.addr() as *mut c_void,
                    core::ptr::null(),
                    1,
                    8,
                    1e-6,
                    10,
                    1,
                )
            } else {
                ffi::w34_fused_add_gemma_rms_norm_f16(
                    &ex,
                    x.addr() as *const c_void,
                    residual.addr() as *mut c_void,
                    weight.addr() as *const c_void,
                    output.addr() as *mut c_void,
                    core::ptr::null(),
                    1,
                    8,
                    1e-6,
                )
            }
        };
        assert_eq!(rc, 0);
        let mut got_res = [0u16; 8];
        let mut got_out = [0u16; 8];
        let mut got_tap = [0u16; 10];
        residual.copy_to_host(&mut got_res, &stream).unwrap();
        output.copy_to_host(&mut got_out, &stream).unwrap();
        tap.copy_to_host(&mut got_tap, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            got_res,
            [15360, 15362, 16384, 49154, 14336, 47101, 17408, 50173]
        );
        if with_tap {
            // Existing tap contract: BF16 of FP32 sum, not BF16 of stored half.
            assert_eq!(
                got_tap,
                [0, 16256, 16256, 16384, 49152, 16128, 48896, 16512, 49280, 0]
            );
        }
        assert_eq!(
            got_out,
            [14066, 14424, 14646, 48439, 12018, 46029, 15892, 49903]
        );
    }
}

#[test]
fn vec8_rms_norm_and_silu_mul_match_vec2_fallback_zero_ulp() {
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let ex = stream.exec(0, 0);

    // 1. Fused residual + GemmaRMSNorm: compare 16B-aligned `vec8` against +4B-offset scalar
    // fallback across `hidden_size = 2560` (4B), `4096`, and `5120` (27B), including a masked row.
    for hs in [2560usize, 4096, 5120] {
        let rows = 3usize;
        let pad_u16 = 2usize; // +4 bytes breaks 16B alignment while preserving 4B `vec2` alignment
        let mut x_host = vec![0u16; pad_u16 + rows * hs];
        let mut res_host = vec![0u16; pad_u16 + rows * hs];
        let mut w_host = vec![0u16; pad_u16 + hs];
        for i in 0..rows * hs {
            x_host[pad_u16 + i] = f32_to_f16(((i % 97) as f32 - 48.0) * 0.03125);
            res_host[pad_u16 + i] = f32_to_f16(((i % 53) as f32 - 26.0) * 0.015625);
        }
        for i in 0..hs {
            w_host[pad_u16 + i] = f32_to_f16(((i % 29) as f32 - 14.0) * 0.02);
        }
        let mask_host = [1u8, 0, 1];
        let d_mask = DeviceBuffer::alloc(device.device_id, mask_host.len()).unwrap();
        d_mask.copy_from_host(&mask_host, &stream).unwrap();

        let run_norm = |offset_u16: usize| -> (Vec<u16>, Vec<u16>) {
            let d_x = DeviceBuffer::alloc(device.device_id, x_host.len() * 2).unwrap();
            let d_r = DeviceBuffer::alloc(device.device_id, res_host.len() * 2).unwrap();
            let d_w = DeviceBuffer::alloc(device.device_id, w_host.len() * 2).unwrap();
            let d_o = DeviceBuffer::alloc(device.device_id, (pad_u16 + rows * hs) * 2).unwrap();
            let off_src = pad_u16 - offset_u16;
            d_x.copy_from_host(&x_host[off_src..], &stream).unwrap();
            d_r.copy_from_host(&res_host[off_src..], &stream).unwrap();
            d_w.copy_from_host(&w_host[off_src..], &stream).unwrap();
            let sentinel = vec![0x7BFFu16; pad_u16 + rows * hs];
            d_o.copy_from_host(&sentinel, &stream).unwrap();
            let byte_off = (offset_u16 * 2) as u64;
            // SAFETY: Device buffers remain live until `stream.synchronize()` below and hold `pad_u16 + rows * hs` elements.
            let rc = unsafe {
                ffi::w34_fused_add_gemma_rms_norm_f16(
                    &ex,
                    (d_x.addr() + byte_off) as *const c_void,
                    (d_r.addr() + byte_off) as *mut c_void,
                    (d_w.addr() + byte_off) as *const c_void,
                    (d_o.addr() + byte_off) as *mut c_void,
                    d_mask.addr() as *const u8,
                    rows as i32,
                    hs as i32,
                    1e-6,
                )
            };
            assert_eq!(rc, 0);
            let mut got_r = vec![0u16; pad_u16 + rows * hs];
            let mut got_o = vec![0u16; pad_u16 + rows * hs];
            d_r.copy_to_host(&mut got_r, &stream).unwrap();
            d_o.copy_to_host(&mut got_o, &stream).unwrap();
            stream.synchronize().unwrap();
            (
                got_r[offset_u16..offset_u16 + rows * hs].to_vec(),
                got_o[offset_u16..offset_u16 + rows * hs].to_vec(),
            )
        };
        let (r_vec8, o_vec8) = run_norm(0);
        let (r_vec2, o_vec2) = run_norm(pad_u16);
        assert_eq!(r_vec8, r_vec2, "0-ULP residual parity failed at hs={hs}");
        assert_eq!(o_vec8, o_vec2, "0-ULP output parity failed at hs={hs}");
        assert!(
            o_vec8[hs..2 * hs].iter().all(|&b| b == 0),
            "masked norm output row must be zeroed at hs={hs}"
        );
        assert_eq!(
            &r_vec8[hs..2 * hs],
            &res_host[pad_u16 + hs..pad_u16 + 2 * hs],
            "masked norm residual row must remain untouched at hs={hs}"
        );
    }

    // 2. SiLU-and-mul: compare 16B-aligned `vec8` against +4B-offset `vec2` across `d = 2560, 9216`.
    for d in [2560usize, 9216] {
        let rows = 3usize;
        let pad_u16 = 2usize;
        let mut gu_host = vec![0u16; pad_u16 + rows * 2 * d];
        for i in 0..rows * 2 * d {
            gu_host[pad_u16 + i] = f32_to_f16(((i % 113) as f32 - 56.0) * 0.04);
        }
        let mask_host = [1u8, 0, 1];
        let d_mask = DeviceBuffer::alloc(device.device_id, mask_host.len()).unwrap();
        d_mask.copy_from_host(&mask_host, &stream).unwrap();

        let run_silu = |offset_u16: usize| -> Vec<u16> {
            let d_in = DeviceBuffer::alloc(device.device_id, gu_host.len() * 2).unwrap();
            let d_out = DeviceBuffer::alloc(device.device_id, (pad_u16 + rows * d) * 2).unwrap();
            d_in.copy_from_host(&gu_host[pad_u16 - offset_u16..], &stream)
                .unwrap();
            let sentinel = vec![0x7BFFu16; pad_u16 + rows * d];
            d_out.copy_from_host(&sentinel, &stream).unwrap();
            let byte_off = (offset_u16 * 2) as u64;
            // SAFETY: Device buffers remain live until `stream.synchronize()` below and hold `pad_u16 + rows * 2 * d` / `pad_u16 + rows * d` elements.
            let rc = unsafe {
                ffi::w34_silu_and_mul_f16(
                    &ex,
                    (d_in.addr() + byte_off) as *const c_void,
                    (d_out.addr() + byte_off) as *mut c_void,
                    d_mask.addr() as *const u8,
                    rows as i32,
                    d as i32,
                )
            };
            assert_eq!(rc, 0);
            let mut got = vec![0u16; pad_u16 + rows * d];
            d_out.copy_to_host(&mut got, &stream).unwrap();
            stream.synchronize().unwrap();
            got[offset_u16..offset_u16 + rows * d].to_vec()
        };
        let s_vec8 = run_silu(0);
        let s_vec2 = run_silu(pad_u16);
        assert_eq!(s_vec8, s_vec2, "0-ULP silu_and_mul parity failed at d={d}");
        assert!(
            s_vec8[d..2 * d].iter().all(|&b| b == 0x7BFF),
            "masked silu_and_mul row must remain untouched at d={d}"
        );
    }
}

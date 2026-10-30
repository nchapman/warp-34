//! Native vLLM 0.31.0 Gemma residual boundary regression (GPU-only, no model).
#![cfg(feature = "cuda")]
use std::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer};
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

//! Preserve native FP16 materialization boundaries in fused vision operations.
#![cfg(feature = "cuda")]
// Preserve exact decimal representations of native FP16 oracle values.
#![allow(clippy::excessive_precision)]
#![allow(clippy::undocumented_unsafe_blocks)]
use std::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, f32_to_f16};
use w34_kernels::ffi;

#[test]
fn vision_bias_and_rope_preserve_native_half_rounding() {
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let scratch = DeviceBuffer::alloc(device.device_id, 4).unwrap();
    let ex = stream.exec(scratch.addr(), 4);
    let upload = |values: &[f32]| {
        let buffer = DeviceBuffer::alloc(device.device_id, values.len() * 2).unwrap();
        buffer
            .copy_from_host(
                &values.iter().copied().map(f32_to_f16).collect::<Vec<_>>(),
                &stream,
            )
            .unwrap();
        buffer
    };
    let x = upload(&[1.0]);
    let bias = upload(&[0.00048828125]);
    let residual = upload(&[-1.0]);
    let output = upload(&[0.0]);
    assert_eq!(
        // SAFETY: each live FP16 buffer has the declared single-element extent.
        unsafe {
            ffi::w34_vision_bias_add_f16(
                &ex,
                x.addr() as *const c_void,
                bias.addr() as *const c_void,
                residual.addr() as *const c_void,
                output.addr() as *mut c_void,
                1,
                1,
            )
        },
        0
    );
    let mut actual = [0u16];
    output.copy_to_host(&mut actual, &stream).unwrap();
    stream.synchronize().unwrap();
    // Native linear's FP16 result rounds 1 + half an ULP to 1 before residual add.
    assert_eq!(
        actual[0],
        f32_to_f16(0.0),
        "bias result must round before residual add"
    );
    for tanh in [0, 1] {
        assert_eq!(
            // SAFETY: live single-element FP16 buffers and a supported GELU mode.
            unsafe {
                ffi::w34_vision_bias_gelu_f16(
                    &ex,
                    x.addr() as *const c_void,
                    bias.addr() as *const c_void,
                    output.addr() as *mut c_void,
                    1,
                    1,
                    tanh,
                )
            },
            0
        );
        output.copy_to_host(&mut actual, &stream).unwrap();
        stream.synchronize().unwrap();
        // torch.nn.functional.gelu(torch.tensor([1.],dtype=torch.float16)), both approximations.
        assert_eq!(
            actual[0],
            f32_to_f16(0.8413086),
            "bias must materialize before GELU"
        );
    }
    let qkv = upload(&[1.0; 192]);
    let bias = upload(&[0.00048828125; 192]);
    let pos = DeviceBuffer::alloc(device.device_id, 8).unwrap();
    pos.copy_from_host(&[1i32, 1], &stream).unwrap();
    let seq = DeviceBuffer::alloc(device.device_id, 8).unwrap();
    seq.copy_from_host(&[0i32, 1], &stream).unwrap();
    let out = upload(&[0.0; 64]);
    assert_eq!(
        // SAFETY: QKV has 3 * 64 halves, output 64 halves, and valid position/sequence arrays.
        unsafe {
            ffi::w34_vision_qkv_rope_attn_f16(
                &ex,
                qkv.addr() as *mut c_void,
                bias.addr() as *const c_void,
                pos.addr() as *const i32,
                seq.addr() as *const i32,
                out.addr() as *mut c_void,
                1,
                1,
                1,
                64,
                10000.0,
            )
        },
        0
    );
    let mut rotated = [0u16; 192];
    qkv.copy_to_host(&mut rotated, &stream).unwrap();
    stream.synchronize().unwrap();
    // Native RotaryEmbedding stores cos/sin in FP16; ApplyRotaryEmb widens to FP32.
    let cos = 0.54052734f32;
    let sin = 0.8413086f32;
    assert_eq!(rotated[0], f32_to_f16(cos - sin));
    assert_eq!(rotated[32], f32_to_f16(cos + sin));
}

#[test]
fn vision_tanh_gelu_matches_native_activation() {
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let ex = stream.exec(0, 0);
    // Actual PyTorch 2.13 CUDA GELU(tanh), also verified at native ViT block
    // 0/12 activation boundaries. Fast tanhf loses these half rounding bits.
    let input = [0.02783203125, 1.2255859375, -0.36962890625];
    let expected = [0.01422119140625, 1.0908203125, -0.1314697265625];
    let x = DeviceBuffer::alloc(device.device_id, input.len() * 2).unwrap();
    x.copy_from_host(&input.map(f32_to_f16), &stream).unwrap();
    let out = DeviceBuffer::alloc(device.device_id, input.len() * 2).unwrap();
    assert_eq!(
        unsafe {
            ffi::w34_vision_bias_gelu_f16(
                &ex,
                x.addr() as *const c_void,
                std::ptr::null(),
                out.addr() as *mut c_void,
                1,
                input.len() as i32,
                1,
            )
        },
        0
    );
    let mut actual = [0u16; 3];
    out.copy_to_host(&mut actual, &stream).unwrap();
    assert_eq!(actual, expected.map(f32_to_f16));
}

#[test]
fn vision_rope_preserves_native_small_angle_coefficients() {
    let device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(device.device_id).unwrap();
    let scratch = DeviceBuffer::alloc(device.device_id, 4).unwrap();
    let ex = stream.exec(scratch.addr(), 4);
    let pos = DeviceBuffer::alloc(device.device_id, 8).unwrap();
    pos.copy_from_host(&[1i32, 1], &stream).unwrap();
    let seq = DeviceBuffer::alloc(device.device_id, 8).unwrap();
    seq.copy_from_host(&[0i32, 1], &stream).unwrap();
    // vLLM 0.31.0 RotaryEmbedding(D, D/2, 8192, 10000, true, float16),
    // CUDA cache at position 1, followed by ApplyRotaryEmb.forward_cuda.
    // Unit inputs isolate sin(theta) without attention or GEMM error.
    for (d, expected_sin) in [
        (64usize, 0.00017786026000976562),
        (72, 0.00016677379608154297),
    ] {
        let mut input = vec![0u16; 3 * d];
        for part in 0..3 {
            input[part * d..part * d + d / 2].fill(f32_to_f16(1.0));
        }
        let qkv = DeviceBuffer::alloc(device.device_id, input.len() * 2).unwrap();
        qkv.copy_from_host(&input, &stream).unwrap();
        let out = DeviceBuffer::alloc(device.device_id, d * 2).unwrap();
        assert_eq!(
            unsafe {
                ffi::w34_vision_qkv_rope_attn_f16(
                    &ex,
                    qkv.addr() as *mut c_void,
                    std::ptr::null(),
                    pos.addr() as *const i32,
                    seq.addr() as *const i32,
                    out.addr() as *mut c_void,
                    1,
                    1,
                    1,
                    d as i32,
                    10000.0,
                )
            },
            0
        );
        let mut actual = vec![0u16; 3 * d];
        qkv.copy_to_host(&mut actual, &stream).unwrap();
        stream.synchronize().unwrap();
        for part in 0..2 {
            for axis in 0..2 {
                let offset = part * d + d / 2 + (axis + 1) * (d / 4) - 1;
                assert_eq!(
                    actual[offset],
                    f32_to_f16(expected_sin),
                    "D={d}, part={part}, axis={axis}"
                );
            }
        }
    }
}

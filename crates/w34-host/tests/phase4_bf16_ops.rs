#![cfg(feature = "cuda")]
//! Replays actual SGLang component outputs; this is not a full drafter gate.
use core::ffi::c_void;
use w34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, check_rc};
use w34_host::loader::SafetensorsArchive;
use w34_kernels::ffi;

fn fixtures() -> SafetensorsArchive {
    SafetensorsArchive::open(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/oracle/goldens/sglang_dflash_bf16.safetensors"),
    )
    .expect("run sglang_dflash_bf16_dump.py in pinned SGLang image")
}
fn input(a: &SafetensorsArchive, name: &str, stream: &CudaStream) -> DeviceBuffer {
    let bytes = a.read_bytes(name).unwrap();
    let buffer = DeviceBuffer::alloc(0, bytes.len()).unwrap();
    buffer.copy_from_host(&bytes, stream).unwrap();
    buffer
}
fn compare(a: &SafetensorsArchive, name: &str, out: &DeviceBuffer, stream: &CudaStream) {
    let expected = a.read_f32(name).unwrap();
    let mut bits = vec![0u16; expected.len()];
    out.copy_to_host(&mut bits, stream).unwrap();
    let actual: Vec<f32> = bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let mut max_abs = 0.0f32;
    let mut bad = 0;
    for (&x, &y) in actual.iter().zip(&expected) {
        assert!(x.is_finite(), "{name}: nonfinite output");
        let diff = (x - y).abs();
        max_abs = max_abs.max(diff);
        if diff > 0.008 * y.abs() + 0.0002 {
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "{name}: {bad} values exceed BF16 rounding tolerance; max abs {max_abs}"
    );
}

#[test]
fn bf16_norm_and_fused_residual_match_sglang() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let a = fixtures();
    for hidden in [128, 256, 3072, 5120] {
        let x = input(&a, &format!("norm{hidden}.x"), &stream);
        let res = input(&a, &format!("norm{hidden}.residual"), &stream);
        let w = input(&a, &format!("norm{hidden}.weight"), &stream);
        let out = DeviceBuffer::alloc(0, 16 * hidden * 2).unwrap();
        // SAFETY: input/output allocations hold 16 BF16 rows of the declared width.
        unsafe {
            check_rc(
                ffi::w34_rms_norm_bf16(
                    &ex,
                    x.addr() as *const c_void,
                    core::ptr::null_mut(),
                    w.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    16,
                    hidden as i32,
                    1e-6,
                ),
                "BF16 norm",
            )
            .unwrap();
        }
        compare(&a, &format!("norm{hidden}.standalone"), &out, &stream);
        // SAFETY: the residual is a distinct allocation, as required by fused RMSNorm.
        unsafe {
            check_rc(
                ffi::w34_rms_norm_bf16(
                    &ex,
                    x.addr() as *const c_void,
                    res.addr() as *mut c_void,
                    w.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    16,
                    hidden as i32,
                    1e-6,
                ),
                "BF16 fused norm",
            )
            .unwrap();
        }
        compare(&a, &format!("norm{hidden}.fused"), &out, &stream);
        compare(&a, &format!("norm{hidden}.res_out"), &res, &stream);
    }
}

#[test]
fn bf16_silu_and_grouped_convolution_match_sglang() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let a = fixtures();
    let gu = input(&a, "silu.input", &stream);
    let out = DeviceBuffer::alloc(0, 32 * 128 * 2).unwrap();
    // SAFETY: packed gate/up input has 32 rows of width 256, output width 128.
    unsafe {
        check_rc(
            ffi::w34_silu_and_mul_bf16(
                &ex,
                gu.addr() as *const c_void,
                out.addr() as *mut c_void,
                32,
                128,
            ),
            "BF16 SiLU",
        )
        .unwrap();
    }
    compare(&a, "silu.output", &out, &stream);
    for block in [8, 16] {
        let x = input(&a, &format!("conv{block}.x"), &stream);
        let delta = input(&a, &format!("conv{block}.delta"), &stream);
        let base = input(&a, &format!("conv{block}.base"), &stream);
        let finish = DeviceBuffer::alloc(0, 32 * 128 * 2).unwrap();
        // SAFETY: fixture shapes are x=[32,128], delta=[32,2,3,2], base=[2,3,128].
        unsafe {
            check_rc(
                ffi::w34_dflash_conv_bf16(
                    &ex,
                    x.addr() as *const c_void,
                    delta.addr() as *const c_void,
                    base.addr() as *const c_void,
                    out.addr() as *mut c_void,
                    32,
                    128,
                    block,
                    64,
                    3,
                    0,
                ),
                "BF16 conv prepare",
            )
            .unwrap();
        }
        compare(&a, &format!("conv{block}.prepared"), &out, &stream);
        // SAFETY: output storage is separate; same fixture coefficients select side 1.
        unsafe {
            check_rc(
                ffi::w34_dflash_conv_bf16(
                    &ex,
                    out.addr() as *const c_void,
                    delta.addr() as *const c_void,
                    base.addr() as *const c_void,
                    finish.addr() as *mut c_void,
                    32,
                    128,
                    block,
                    64,
                    3,
                    1,
                ),
                "BF16 conv finish",
            )
            .unwrap();
        }
        compare(&a, &format!("conv{block}.finished"), &finish, &stream);
    }
}

#[test]
fn bf16_qk_norm_rope_and_packed_copy_match_sglang() {
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);
    let a = fixtures();
    let qkv = input(&a, "rope.input", &stream);
    let pos = input(&a, "rope.positions", &stream);
    let qw = input(&a, "rope.qw", &stream);
    let kw = input(&a, "rope.kw", &stream);
    let table = input(&a, "rope.table", &stream);
    // SAFETY: packed [16,12*128] tensor, Q=8,K=2,V=2; positions lie in [0,1024).
    unsafe {
        check_rc(
            ffi::w34_dflash_qk_norm_rope_bf16(
                &ex,
                qkv.addr() as *mut c_void,
                pos.addr() as *const i32,
                qw.addr() as *const c_void,
                kw.addr() as *const c_void,
                table.addr() as *const c_void,
                16,
                8,
                2,
                128,
                1024,
                1e-6,
            ),
            "BF16 QK norm/RoPE",
        )
        .unwrap();
    }
    compare(&a, "rope.output", &qkv, &stream);
    let v = DeviceBuffer::alloc(0, 16 * 2 * 128 * 2).unwrap();
    // SAFETY: copies the V columns [10*128,12*128) into separate dense rows.
    unsafe {
        check_rc(
            ffi::w34_copy_cols_bf16(
                &ex,
                qkv.addr() as *const c_void,
                v.addr() as *mut c_void,
                16,
                12 * 128,
                2 * 128,
                10 * 128,
                0,
                2 * 128,
            ),
            "BF16 column copy",
        )
        .unwrap();
    }
    let original = a.read_bf16_bits("rope.input").unwrap();
    let expected: Vec<u16> = original
        .chunks(12 * 128)
        .flat_map(|row| row[10 * 128..].iter().copied())
        .collect();
    let mut bits = vec![0u16; expected.len()];
    v.copy_to_host(&mut bits, &stream).unwrap();
    assert_eq!(bits, expected, "QK transform must leave V bit-identical");
}

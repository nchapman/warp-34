use super::*;
use crate::cuda::{CublasHandle, CudaDevice, CudaStream};

fn f32_to_bf16_rn(val: f32) -> u16 {
    let bits = val.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[test]
fn bf16_marlin_u4b8_and_hadamard_match_reference() {
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let blas = CublasHandle::new(&stream).unwrap();
    let ex = stream.exec(0, 0);

    let (m, n, k, gs) = (8usize, 128usize, 256usize, 64usize);
    let num_groups = k / gs;
    let packed: Vec<u32> = (0..(n * (k / 8)))
        .map(|i| {
            let mut w = 0u32;
            for b in 0..8 {
                let q = ((i * 13 + b * 7 + 3) % 16) as u32;
                w |= q << (b * 4);
            }
            w
        })
        .collect();
    let scales: Vec<u16> = (0..(n * num_groups))
        .map(|i| f32_to_bf16_rn(0.015625 * (1.0 + ((i * 11 + 5) % 17) as f32 * 0.125)))
        .collect();

    let mut w_deq_bf16 = vec![0u16; n * k];
    for r in 0..n {
        for c_word in 0..(k / 8) {
            let w = packed[r * (k / 8) + c_word];
            let col0 = c_word * 8;
            let s = bf16_to_f32(scales[r * num_groups + col0 / gs]);
            for b in 0..8 {
                let q = ((w >> (b * 4)) & 0xF) as i32 - 8;
                w_deq_bf16[r * k + col0 + b] = f32_to_bf16_rn((q as f32) * s);
            }
        }
    }

    let x_host: Vec<u16> = (0..(m * k))
        .map(|i| f32_to_bf16_rn((((i * 17 + 5) % 97) as f32 - 48.0) * 0.03125))
        .collect();
    let mut x_had_cpu = vec![0u16; m * k];
    for r in 0..m {
        for blk in (0..k).step_by(64) {
            let mut v: Vec<f32> = (0..64)
                .map(|j| bf16_to_f32(x_host[r * k + blk + j]))
                .collect();
            let mut h_step = 1usize;
            while h_step < 64 {
                for i0 in (0..64).step_by(h_step * 2) {
                    for j in i0..(i0 + h_step) {
                        let (a, b) = (v[j], v[j + h_step]);
                        v[j] = a + b;
                        v[j + h_step] = a - b;
                    }
                }
                h_step *= 2;
            }
            for j in 0..64 {
                x_had_cpu[r * k + blk + j] = f32_to_bf16_rn(v[j] * 0.125);
            }
        }
    }

    let x_dev = DeviceBuffer::alloc(0, m * k * 2).unwrap();
    x_dev.copy_from_host(&x_host, &stream).unwrap();
    let x_had_dev = DeviceBuffer::alloc(0, m * k * 2).unwrap();
    // SAFETY: x_dev and x_had_dev are valid BF16 device buffers of m * k elements on stream.
    let rc = unsafe {
        ffi::w34_block_hadamard_bf16(
            &ex,
            x_dev.addr() as *const c_void,
            x_had_dev.addr() as *mut c_void,
            m as i32,
            k as i32,
            k as i32,
            64,
        )
    };
    assert_eq!(rc, 0);
    let mut x_had_gpu = vec![0u16; m * k];
    x_had_dev.copy_to_host(&mut x_had_gpu, &stream).unwrap();
    assert_eq!(x_had_gpu, x_had_cpu);

    let w_deq_dev = DeviceBuffer::alloc(0, n * k * 2).unwrap();
    w_deq_dev.copy_from_host(&w_deq_bf16, &stream).unwrap();
    let y_cublas_deq = DeviceBuffer::alloc(0, m * n * 2).unwrap();
    blas.gemm_bf16(
        x_had_dev.addr(),
        w_deq_dev.addr(),
        y_cublas_deq.addr(),
        m as i32,
        n as i32,
        k as i32,
    )
    .unwrap();

    let tmp_p = DeviceBuffer::alloc(0, packed.len() * 4).unwrap();
    tmp_p.copy_from_host(&packed, &stream).unwrap();
    let w_marlin = DeviceBuffer::alloc(0, packed.len() * 4).unwrap();
    // SAFETY: tmp_p and w_marlin are valid u32 device buffers for [n, k/8] -> Marlin layout.
    let rc = unsafe {
        ffi::w34_marlin_repack_int4(
            &ex,
            tmp_p.addr() as *const u32,
            w_marlin.addr() as *mut u32,
            k as i64,
            n as i64,
        )
    };
    assert_eq!(rc, 0);
    let perm_s = marlin_permute_scales_from_n_major(&scales, k, n, gs);
    let s_marlin = DeviceBuffer::alloc(0, perm_s.len() * 2).unwrap();
    s_marlin.copy_from_host(&perm_s, &stream).unwrap();
    let ws_bytes = dev.props().unwrap().marlin_workspace_bytes();
    let ws = DeviceBuffer::alloc(0, ws_bytes).unwrap();
    ws.copy_from_host(&vec![0u8; ws_bytes], &stream).unwrap();
    let y_marlin = DeviceBuffer::alloc(0, m * n * 2).unwrap();
    // SAFETY: x_had_dev, w_marlin, s_marlin, y_marlin, and ws are valid device buffers on stream.
    let rc = unsafe {
        ffi::w34_marlin_gemm_bf16u4(
            &ex,
            x_had_dev.addr() as *const c_void,
            w_marlin.addr() as *const c_void,
            s_marlin.addr() as *const c_void,
            y_marlin.addr() as *mut c_void,
            ws.addr() as *mut i32,
            m as i64,
            n as i64,
            k as i64,
            (ws_bytes / 4) as i64,
            4,
        )
    };
    assert_eq!(rc, 0);

    let mut h_marlin = vec![0u16; m * n];
    let mut h_deq = vec![0u16; m * n];
    y_marlin.copy_to_host(&mut h_marlin, &stream).unwrap();
    y_cublas_deq.copy_to_host(&mut h_deq, &stream).unwrap();

    let (mut err, mut nrm) = (0.0f64, 0.0f64);
    for (&x, &y) in h_marlin.iter().zip(h_deq.iter()) {
        let (xf, yf) = (bf16_to_f32(x) as f64, bf16_to_f32(y) as f64);
        err += (xf - yf).powi(2);
        nrm += yf.powi(2);
    }
    let rel_l2 = (err / nrm).sqrt();
    assert!(rel_l2 < 2e-3, "marlin_vs_deq={rel_l2}");
}

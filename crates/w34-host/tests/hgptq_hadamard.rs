#![cfg(feature = "cuda")]

use std::ffi::c_void;
use w34_host::cuda::{
    CudaDevice, CudaEvent, CudaGraph, CudaStream, DeviceBuffer, f16_to_f32, f32_to_f16,
};
use w34_kernels::ffi;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn upload(values: &[u16], stream: &CudaStream) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(0, std::mem::size_of_val(values)).unwrap();
    buffer.copy_from_host(values, stream).unwrap();
    buffer
}

// Independent dense Sylvester matrix, not the production butterfly algorithm.
fn reference(input: &[u16], k: usize, padded_k: usize, block: usize) -> Vec<u16> {
    let mut out = vec![0; input.len() / k * padded_k];
    for (row, x) in input.chunks_exact(k).enumerate() {
        for base in (0..k).step_by(block) {
            for i in 0..block {
                let sum: f64 = (0..block)
                    .map(|j| {
                        let sign = if (i & j).count_ones().is_multiple_of(2) {
                            1.0
                        } else {
                            -1.0
                        };
                        f64::from(f16_to_f32(x[base + j])) * sign
                    })
                    .sum();
                out[row * padded_k + base + i] = f32_to_f16((sum / (block as f64).sqrt()) as f32);
            }
        }
    }
    out
}

fn launch(
    stream: &CudaStream,
    src: &DeviceBuffer,
    dst: &DeviceBuffer,
    rows: usize,
    k: usize,
    padded_k: usize,
    block: usize,
) {
    assert!(src.len_bytes >= rows * k * 2);
    assert!(dst.len_bytes >= rows * padded_k * 2);
    // SAFETY: disjoint live device buffers have the asserted extents; test widths
    // are block-aligned. The stream and its device remain live through completion.
    let rc = unsafe {
        ffi::w34_block_hadamard_f16(
            &stream.exec(0, 0),
            src.addr() as *const c_void,
            dst.addr() as *mut c_void,
            rows as i32,
            k as i32,
            padded_k as i32,
            block as i32,
        )
    };
    assert_eq!(rc, 0);
}

#[test]
fn block_hadamard_matches_dense_sylvester_preserves_input_and_zeros_padding() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for (block, k, padded_k, rows) in [
        (16, 16, 16, 1),
        (64, 64, 64, 1),
        (16, 48, 80, 3),
        (64, 192, 320, 3),
        (16, 4304, 4352, 17),
        (64, 2048, 2048, 129),
    ] {
        let input: Vec<u16> = (0..rows * k)
            .map(|i| f32_to_f16(((i * 73 + i / k * 11) % 509) as f32 / 32.0 - 8.0))
            .collect();
        let src = upload(&input, &stream);
        // Canary tail catches writes beyond the logical padded output allocation.
        let dst = upload(&vec![0x7bff; rows * padded_k + 64], &stream);
        launch(&stream, &src, &dst, rows, k, padded_k, block);
        let mut got = vec![0; rows * padded_k + 64];
        dst.copy_to_host(&mut got, &stream).unwrap();
        assert_eq!(
            &got[..rows * padded_k],
            reference(&input, k, padded_k, block)
        );
        assert_eq!(&got[rows * padded_k..], &[0x7bff; 64]);
        let mut unchanged = vec![0; input.len()];
        src.copy_to_host(&mut unchanged, &stream).unwrap();
        assert_eq!(unchanged, input);
    }
}

#[test]
fn block_hadamard_basis_and_large_intermediates_match_reference() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for block in [16, 64] {
        let rows = block + 3;
        let mut input = vec![0; rows * block];
        for i in 0..block {
            input[i * block + i] = f32_to_f16(1.0);
        }
        // Unnormalized partial sums overflow FP16, but final output is finite.
        input[block * block..(block + 1) * block].fill(f32_to_f16(8000.0));
        input[(block + 1) * block..(block + 2) * block].fill(1); // min subnormal
        let src = upload(&input, &stream);
        let dst = upload(&vec![0; input.len()], &stream);
        launch(&stream, &src, &dst, rows, block, block, block);
        let mut got = vec![0; input.len()];
        dst.copy_to_host(&mut got, &stream).unwrap();
        assert_eq!(got, reference(&input, block, block, block));
    }
}

#[test]
fn block_hadamard_graph_replay_reads_updated_inputs() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for (block, k, padded_k) in [(16, 4304, 4352), (64, 2048, 2048)] {
        let src = upload(&vec![0; 3 * k], &stream);
        let dst = upload(&vec![0; 3 * padded_k], &stream);
        launch(&stream, &src, &dst, 3, k, padded_k, block);
        stream.synchronize().unwrap();
        let graph = CudaGraph::capture(&stream, || {
            launch(&stream, &src, &dst, 3, k, padded_k, block);
            Ok(())
        })
        .unwrap();
        for seed in 1..=3 {
            let input: Vec<u16> = (0..3 * k)
                .map(|i| f32_to_f16(((i * seed) % 113) as f32 / 16.0))
                .collect();
            src.copy_from_host(&input, &stream).unwrap();
            graph.launch(&stream).unwrap();
            let mut got = vec![0; 3 * padded_k];
            dst.copy_to_host(&mut got, &stream).unwrap();
            assert_eq!(got, reference(&input, k, padded_k, block));
        }
    }
}

#[test]
fn block_hadamard_rejects_invalid_geometry_and_aliasing() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let src = upload(&[0; 256], &stream);
    let dst = upload(&[0; 256], &stream);
    let ex = stream.exec(0, 0);
    for (rows, k, padded_k, block) in [
        (-1, 64, 64, 64),
        (1, 0, 64, 64),
        (1, 64, 32, 64),
        (1, 64, 64, 32),
        (1, 63, 64, 64),
        (1, 64, 65, 64),
    ] {
        assert_ne!(
            // SAFETY: invalid geometry must be rejected without touching valid buffers.
            unsafe {
                ffi::w34_block_hadamard_f16(
                    &ex,
                    src.addr() as *const c_void,
                    dst.addr() as *mut c_void,
                    rows,
                    k,
                    padded_k,
                    block,
                )
            },
            0
        );
    }
    for output in [src.addr(), src.addr() + 2, src.addr() - 2] {
        assert_ne!(
            // SAFETY: overlapping ranges must be rejected before dereferencing.
            unsafe {
                ffi::w34_block_hadamard_f16(
                    &ex,
                    src.addr() as *const c_void,
                    output as *mut c_void,
                    1,
                    64,
                    64,
                    64,
                )
            },
            0
        );
    }
    assert_eq!(
        // SAFETY: an empty batch permits null data pointers and never launches.
        unsafe {
            ffi::w34_block_hadamard_f16(&ex, std::ptr::null(), std::ptr::null_mut(), 0, 64, 64, 64)
        },
        0
    );
}

#[test]
#[ignore = "manual GPU microbenchmark; run with --ignored --nocapture on an idle GPU"]
fn benchmark_block_hadamard() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for (block, k, padded_k) in [(64, 2048, 2048), (64, 6144, 6144), (16, 4304, 4352)] {
        for rows in [1, 8, 16, 512, 4096] {
            let src = upload(&vec![f32_to_f16(0.25); rows * k], &stream);
            let dst = upload(&vec![0; rows * padded_k], &stream);
            launch(&stream, &src, &dst, rows, k, padded_k, block);
            stream.synchronize().unwrap();
            let graph = CudaGraph::capture(&stream, || {
                for _ in 0..100 {
                    launch(&stream, &src, &dst, rows, k, padded_k, block);
                }
                Ok(())
            })
            .unwrap();
            graph.launch(&stream).unwrap();
            let start = CudaEvent::new(0, true).unwrap();
            let end = CudaEvent::new(0, true).unwrap();
            start.record(&stream).unwrap();
            for _ in 0..10 {
                graph.launch(&stream).unwrap();
            }
            end.record(&stream).unwrap();
            stream.synchronize().unwrap();
            let us = start.elapsed_ms(&end).unwrap(); // 1000 kernels: ms total = us/kernel
            eprintln!("H{block} M={rows} K={k} padded_K={padded_k}: {us:.3} us/kernel");
        }
    }
}

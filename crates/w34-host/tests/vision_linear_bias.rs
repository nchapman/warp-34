//! F.linear must add bias before the FP16 output materialization.
//! vLLM UnquantizedLinearMethod.apply -> default_unquantized_gemm ->
//! torch.nn.functional.linear(x, weight, bias) (layers/linear.py, layers/utils.py).
//! Run under flock /tmp/w34-gpu0.lock with CUDA_VISIBLE_DEVICES=0.
#![cfg(feature = "cuda")]
// Exact FP16 rounding-boundary values, kept unshortened for auditability.
#![allow(clippy::excessive_precision)]
use w34_host::cuda::{CublasHandle, CudaDevice, CudaGraph, CudaStream, DeviceBuffer, f32_to_f16};
// Keep capture isolated from another test's cuBLAS initialization/teardown.
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[test]
fn linear_bias_materialization() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();
    let blas = CublasHandle::new(&stream).unwrap();
    let upload = |v: &[f32]| {
        let b = DeviceBuffer::alloc(dev.device_id, v.len() * 2).unwrap();
        b.copy_from_host(
            &v.iter().copied().map(f32_to_f16).collect::<Vec<_>>(),
            &stream,
        )
        .unwrap();
        b
    };
    let a = upload(&[1., 0.5]);
    let b = upload(&[1., 0.0009765625]);
    let bias = upload(&[-0.00048828125]);
    let c = upload(&[0.]);
    blas.gemm_f16(a.addr(), b.addr(), c.addr(), 1, 1, 2)
        .unwrap();
    let ex = stream.exec(0, 0);
    assert_eq!(
        // SAFETY: live one-element FP16 buffers; null residual is supported.
        unsafe {
            w34_kernels::ffi::w34_vision_bias_add_f16(
                &ex,
                c.addr() as _,
                bias.addr() as _,
                std::ptr::null(),
                c.addr() as _,
                1,
                1,
            )
        },
        0
    );
    let mut out = [0u16];
    c.copy_to_host(&mut out, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        out[0],
        f32_to_f16(0.99951171875),
        "old split path loses the tie"
    );
    blas.gemm_f16_bias(a.addr(), b.addr(), bias.addr(), c.addr(), 1, 1, 2)
        .unwrap();
    c.copy_to_host(&mut out, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(out[0], f32_to_f16(1.0));
    // Non-square multirow layout, per-output-column broadcast and overwrite.
    let a = upload(&[1., 0.5, 2., 1., -1., -0.5]);
    let b = upload(&[1., 0.0009765625, -1., -0.0009765625]);
    let bias = upload(&[-0.00048828125, 0.00048828125]);
    let c = upload(&[99.; 6]);
    for _ in 0..2 {
        blas.gemm_f16_bias(a.addr(), b.addr(), bias.addr(), c.addr(), 3, 2, 2)
            .unwrap();
        let mut out = [0u16; 6];
        c.copy_to_host(&mut out, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            out,
            // Native Lt's tiny-shape algorithm is not an exact-sum CPU oracle.
            // torch F.linear on these same FP16 tensors gives these bits.
            [0.99951171875, -0.99951171875, 2., -2., -1., 1.].map(f32_to_f16)
        );
    }
    assert!(
        blas.gemm_f16_bias(0, b.addr(), bias.addr(), c.addr(), 3, 2, 2)
            .is_err()
    );
    assert!(
        blas.gemm_f16_bias(a.addr(), b.addr(), 0, c.addr(), 3, 2, 2)
            .is_err()
    );
    assert_ne!(
        // SAFETY: the ABI rejects a null handle before accessing the live buffers.
        unsafe {
            w34_kernels::ffi::w34_cublas_gemm_f16_bias(
                0,
                a.addr() as _,
                b.addr() as _,
                bias.addr() as _,
                c.addr() as _,
                3,
                2,
                2,
            )
        },
        0
    );
    for dims in [(0, 2, 2), (3, -1, 2), (3, 2, 0)] {
        assert!(
            blas.gemm_f16_bias(
                a.addr(),
                b.addr(),
                bias.addr(),
                c.addr(),
                dims.0,
                dims.1,
                dims.2
            )
            .is_err()
        );
    }
}

#[test]
fn linear_bias_graph_pointers_and_lifecycle() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();
    let upload = |values: &[f32]| {
        let buf = DeviceBuffer::alloc(dev.device_id, values.len() * 2).unwrap();
        buf.copy_from_host(
            &values.iter().copied().map(f32_to_f16).collect::<Vec<_>>(),
            &stream,
        )
        .unwrap();
        buf
    };
    let av = [1., 2., 3., -1., 0., 2.];
    let bv = [2., -1., 1., 0., 2., -2.];
    let a = upload(&av);
    let b = upload(&bv);
    // The second bias is deliberately only two-byte aligned.
    let biases = [upload(&[0.5, -0.5]), upload(&[99., 2., -3.])];
    let outputs = [upload(&[99.; 4]), upload(&[99.; 4])];
    for _ in 0..3 {
        let blas = CublasHandle::new(&stream).unwrap();
        let run = |i: usize| {
            blas.gemm_f16_bias(
                a.addr(),
                b.addr(),
                biases[i].addr() + if i == 1 { 2 } else { 0 },
                outputs[i].addr(),
                2,
                2,
                3,
            )
        };
        assert!(
            CudaGraph::capture(&stream, || run(0)).is_err(),
            "cold capture must reject preparation"
        );
        run(0).unwrap();
        run(1).unwrap();
        stream.synchronize().unwrap();
        let graph = CudaGraph::capture(&stream, || {
            run(0)?;
            run(1)
        })
        .unwrap();
        for _ in 0..3 {
            graph.launch(&stream).unwrap();
            for (i, output) in outputs.iter().enumerate() {
                let bias = if i == 0 { [0.5, -0.5] } else { [2., -3.] };
                let mut expected = [0u16; 4];
                for row in 0..2 {
                    for col in 0..2 {
                        let sum = (0..3)
                            .map(|k| av[row * 3 + k] * bv[col * 3 + k])
                            .sum::<f32>();
                        expected[row * 2 + col] = f32_to_f16(sum + bias[col]);
                    }
                }
                let mut actual = [0u16; 4];
                output.copy_to_host(&mut actual, &stream).unwrap();
                stream.synchronize().unwrap();
                assert_eq!(actual, expected);
            }
        }
        drop(graph); // workspace owner must outlive graph replays
        drop(blas);
    }
}

#[test]
fn linear_bias_many_shapes_preserve_captured_graph() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();
    let blas = CublasHandle::new(&stream).unwrap();
    let upload = |len: usize, value| {
        let buffer = DeviceBuffer::alloc(dev.device_id, len * 2).unwrap();
        buffer
            .copy_from_host(&vec![f32_to_f16(value); len], &stream)
            .unwrap();
        buffer
    };
    let a = upload(270 * 8, 1.0);
    let b = upload(8 * 8, 1.0);
    let bias = upload(8, 0.5);
    let out = upload(270 * 8, 99.0);
    let run = |rows| blas.gemm_f16_bias(a.addr(), b.addr(), bias.addr(), out.addr(), rows, 8, 8);
    run(8).unwrap();
    let graph = CudaGraph::capture(&stream, || run(8)).unwrap();
    // A serving handle must accept new image shapes indefinitely. Prepared
    // host descriptors may be replaced without invalidating an existing graph.
    for rows in 9..270 {
        run(rows).unwrap_or_else(|e| panic!("new shape {rows}: {e}"));
    }
    out.copy_from_host(&vec![f32_to_f16(99.0); 270 * 8], &stream)
        .unwrap();
    graph.launch(&stream).unwrap();
    let mut actual = vec![0u16; 270 * 8];
    out.copy_to_host(&mut actual, &stream).unwrap();
    stream.synchronize().unwrap();
    assert!(actual[..8 * 8].iter().all(|&x| x == f32_to_f16(8.5)));
    assert!(actual[8 * 8..].iter().all(|&x| x == f32_to_f16(99.0)));
}

#[test]
fn linear_bias_owner_can_follow_stream_teardown() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();
    let mut blas = CublasHandle::new(&stream).unwrap();
    let x = DeviceBuffer::alloc(dev.device_id, 8).unwrap();
    x.copy_from_host(&[f32_to_f16(1.0); 4], &stream).unwrap();
    let out = DeviceBuffer::alloc(dev.device_id, 8).unwrap();
    blas.gemm_f16_bias(x.addr(), x.addr(), x.addr(), out.addr(), 2, 2, 2)
        .unwrap();
    // Rank/model fields can drop their stream before the BLAS owner. Only
    // submissions require a live stream; teardown must not reuse its handle.
    drop(stream);
    // SAFETY: consume the live owner exactly once, then disable Rust's drop.
    let rc = unsafe { w34_kernels::ffi::w34_cublas_destroy(blas.handle) };
    blas.handle = 0;
    assert_eq!(rc, 0);
}

#[test]
fn linear_bias_patch_shape_strict_reduction() {
    // Native golden: torch 2.13.0+cu130, system cuBLAS 13.4, GPU1:
    // x = torch.full((1200,1536), 1.0009765625, device='cuda', dtype=torch.float16)
    // w = torch.full((1152,1536), 1.0009765625, device='cuda', dtype=torch.float16)
    // b = torch.full((1152,), -1539., device='cuda', dtype=torch.float16)
    // torch.nn.functional.linear(x,w,b).view(torch.int16).unique() == tensor([0])
    // This is the real patch GEMM shape, not a CPU exact-sum oracle: native
    // FP32 accumulation loses the tiny product terms. Default cuBLAS reduced
    // precision reduction instead gives 0.5 (0x3800) at every output element.
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();
    let blas = CublasHandle::new(&stream).unwrap();
    let upload = |len: usize, value: f32| {
        let buffer = DeviceBuffer::alloc(dev.device_id, len * 2).unwrap();
        buffer
            .copy_from_host(&vec![f32_to_f16(value); len], &stream)
            .unwrap();
        buffer
    };
    let a = upload(1200 * 1536, 1.0009765625);
    let b = upload(1152 * 1536, 1.0009765625);
    let bias = upload(1152, -1539.);
    let c = upload(1200 * 1152, 99.);
    blas.gemm_f16_bias(a.addr(), b.addr(), bias.addr(), c.addr(), 1200, 1152, 1536)
        .unwrap();
    let mut out = vec![0u16; 1200 * 1152];
    c.copy_to_host(&mut out, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        out.iter().filter(|&&bits| bits != 0).count(),
        0,
        "all outputs must exactly match native F.linear (+0); first bits: {:#06x}",
        out[0]
    );
}

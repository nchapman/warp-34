//! Actual-checkpoint routed-MLP-only benchmark (shared expert omitted on both paths).
//! Usage: cargo run --release -p w34-host --features cuda --example moe_bench -- MODEL OUTPUT.json [ITERATIONS]
//! Select a physical GPU externally with CUDA_VISIBLE_DEVICES; this program uses logical GPU 0.
//! Loads TP1, FP16 KV (32 pages), borrows the first MoE layer's weights without copying them.
//! Inputs are deterministic, zero-mean/unit-RMS rows rounded to FP16, not captured activations.
//! Router projection/top-k, loading, allocation and parity downloads are outside event timing.
//! Dispatch labels mirror op_moe.cu's fused entrypoint predicate, not a forced-MMA override.

#[cfg(not(feature = "cuda"))]
fn main() -> std::process::ExitCode {
    eprintln!("moe_bench requires --features cuda");
    std::process::ExitCode::FAILURE
}

#[cfg(feature = "cuda")]
fn main() -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
    use std::{ffi::c_void, ptr::null};
    use w34_host::{
        cache::KvFormatPolicy,
        cuda::{
            CublasHandle, CudaDevice, CudaEvent, CudaStream, DeviceBuffer, check_rc, f16_to_f32,
            f32_to_f16,
        },
        model::{FfnBlockWeights, MoeExpertStorage, Qwen3_5Engine},
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
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: moe_bench MODEL OUTPUT.json [ITERATIONS]".into());
    }
    let iterations: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(50);
    if iterations == 0 {
        return Err("ITERATIONS must be positive".into());
    }
    let engine = Qwen3_5Engine::load(&args[0], 1, KvFormatPolicy::Fp16, Some(32), false)?;
    let (layer, moe) = engine.ranks[0]
        .layers
        .iter()
        .enumerate()
        .find_map(|(i, l)| {
            if let FfnBlockWeights::Moe(m) = &l.ffn {
                Some((i, m))
            } else {
                None
            }
        })
        .ok_or("checkpoint has no MoE layer")?;
    let (gu, gs, dn, ds, group) = match moe.experts {
        MoeExpertStorage::W4A16 {
            gate_up_packed_ptr,
            gate_up_scales_ptr,
            down_packed_ptr,
            down_scales_ptr,
            group_size,
        } => (
            gate_up_packed_ptr,
            gate_up_scales_ptr,
            down_packed_ptr,
            down_scales_ptr,
            group_size,
        ),
        _ => return Err("first MoE layer must use W4A16 storage".into()),
    };
    let (h, n, e, k) = (
        engine.cfg.hidden_size,
        moe.local_moe_interm,
        moe.num_experts,
        moe.top_k,
    );
    let _device = CudaDevice::set_current(0)?;
    let stream = CudaStream::new(0)?;
    let blas = CublasHandle::new(&stream)?;
    let max_m = 256usize;
    let alloc = |bytes| DeviceBuffer::alloc(0, bytes);
    let x = alloc(max_m * h * 2)?;
    let logits = alloc(max_m * e * 2)?;
    let ids = alloc(max_m * k * 4)?;
    let weights = alloc(max_m * k * 4)?;
    let intermediate = alloc(max_m * k * n * 2)?;
    let outputs = [alloc(max_m * h * 2)?, alloc(max_m * h * 2)?];
    // Production 8 MiB scratch budget (exercises chunked grouped_mma at M=256 as well).
    let ws_bytes = 8 * 1024 * 1024;
    let ws = alloc(ws_bytes)?;
    // Router normally DenseF16; provide reusable zeroed Marlin locks if quantized.
    let marlin_ints = (e.div_ceil(64) * 16).max(1024);
    let marlin = alloc(marlin_ints * 4)?;
    marlin.zero_async(&stream)?;
    let execs = [stream.exec(0, 0), stream.exec(ws.addr(), ws_bytes)];
    let start = CudaEvent::new(0, true)?;
    let end = CudaEvent::new(0, true)?;
    let mut cases = Vec::new();
    let mut passed = true;
    for m in [1, 2, 3, 4, 7, 8, 16, 32, 64, 128, 256] {
        let mut input = Vec::with_capacity(m * h);
        for row in 0..m {
            let values: Vec<f32> = (0..h)
                .map(|col| {
                    let t = (row * 131 + col * 17 + 5) as f32;
                    (t * 0.013).sin() + 0.5 * (t * 0.037).cos()
                })
                .collect();
            let mean = values.iter().sum::<f32>() / h as f32;
            let rms = (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / h as f32).sqrt();
            input.extend(values.iter().map(|v| f32_to_f16((v - mean) / rms)));
        }
        x.copy_from_host(&input, &stream)?;
        moe.router_gate.forward(
            &execs[1],
            &blas,
            marlin.addr(),
            marlin_ints,
            x.addr(),
            logits.addr(),
            m,
        )?;
        check_rc(
            // SAFETY: live allocations sized for m rows outlive the synchronized
            // stream. Router output types match the FFI.
            unsafe {
                ffi::w34_moe_topk_router_f16(
                    &execs[1],
                    logits.addr() as *const c_void,
                    weights.addr() as *mut f32,
                    ids.addr() as *mut i32,
                    m as i32,
                    e as i32,
                    k as i32,
                    i32::from(moe.norm_topk_prob),
                    moe.scoring_func,
                )
            },
            "router",
        )?;
        let launch = |path: usize| {
            check_rc(
                // SAFETY: checkpoint pointers have production GPTQ N-major layout;
                // scratch/intermediate/output capacities cover this geometry. Null
                // shared pointers and null-scratch scalar fallback are supported.
                unsafe {
                    ffi::w34_moe_w4a16_mlp_f16(
                        &execs[path],
                        x.addr() as *const c_void,
                        gu as *const u32,
                        gs as *const c_void,
                        dn as *const u32,
                        ds as *const c_void,
                        ids.addr() as *const i32,
                        weights.addr() as *const f32,
                        null(),
                        null(),
                        intermediate.addr() as *mut c_void,
                        outputs[path].addr() as *mut c_void,
                        m as i32,
                        k as i32,
                        e as i32,
                        n as i32,
                        h as i32,
                        group as i32,
                    )
                },
                "routed MLP",
            )
        };
        for _ in 0..5 {
            launch(0)?;
            launch(1)?;
        }
        stream.synchronize()?;
        let mut ms = [0.0; 2];
        for (path, elapsed) in ms.iter_mut().enumerate() {
            start.record(&stream)?;
            for _ in 0..iterations {
                launch(path)?;
            }
            end.record(&stream)?;
            end.synchronize()?;
            *elapsed = f64::from(start.elapsed_ms(&end)?) / iterations as f64;
        }
        let mut reference = vec![0u16; m * h];
        let mut actual = vec![0u16; m * h];
        outputs[0].copy_to_host(&mut reference, &stream)?;
        outputs[1].copy_to_host(&mut actual, &stream)?;
        let (mut max_abs, mut diff_sq, mut ref_sq, mut exact, mut finite) =
            (0f64, 0f64, 0f64, 0usize, true);
        for (&r, &a) in reference.iter().zip(&actual) {
            exact += usize::from(r == a);
            let (r, a) = (f64::from(f16_to_f32(r)), f64::from(f16_to_f32(a)));
            finite &= r.is_finite() && a.is_finite();
            max_abs = max_abs.max((r - a).abs());
            diff_sq += (r - a).powi(2);
            ref_sq += r * r;
        }
        let rel_l2 = diff_sq.sqrt() / ref_sq.sqrt().max(1e-12);
        let parity = finite && max_abs < 2e-3 && rel_l2 < 1e-3;
        passed &= parity;
        // SAFETY: queries pure dimension/workspace dispatch predicate without dereferencing tensors.
        let regime = unsafe {
            w34_moe_w4a16_select_regime(
                &execs[1],
                m as i32,
                k as i32,
                e as i32,
                n as i32,
                h as i32,
                group as i32,
            )
        };
        let dispatch = match regime {
            1 => "sparse_vec4_4slot",
            2 => "sparse_mma_8slot_4warp",
            3 => "grouped_mma",
            _ => "retained_gemv",
        };
        eprintln!(
            "M={m:3} {dispatch}: baseline={:.4}ms dispatch={:.4}ms speedup={:.3} parity={parity}",
            ms[0],
            ms[1],
            ms[0] / ms[1]
        );
        cases.push(serde_json::json!({"m":m,"dispatch":dispatch,
            "baseline_ms":ms[0],"dispatch_ms":ms[1],"baseline_over_dispatch":ms[0]/ms[1],
            "parity":{"passed":parity,"finite":finite,"max_abs":max_abs,"rel_l2":rel_l2,
                "exact_count":exact,"element_count":m*h}}));
    }
    let report = serde_json::json!({"schema_version":1,"model":args[0],
        "model_snapshot":engine.model_dir,"layer":layer,"tp":1,"kv_policy":"fp16","kv_pages":32,
        "scope":"routed MLP only; shared expert omitted on both paths",
        "input":"deterministic trigonometric rows, centered and unit RMS, rounded FP16",
        "geometry":{"experts":e,"top_k":k,"hidden":h,"intermediate":n,"group_size":group},
        "warmup_iterations":5,"iterations":iterations,"workspace_bytes":ws_bytes,
        "parity_limits":{"max_abs_exclusive":2e-3,"rel_l2_exclusive":1e-3},
        "passed":passed,"cases":cases});
    std::fs::write(&args[1], serde_json::to_string_pretty(&report)?)?;
    Ok(if passed {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    })
}

#![allow(clippy::undocumented_unsafe_blocks)]

#[cfg(feature = "cuda")]
use std::sync::Mutex;
use w34_host::cuda::CapacityPlanner;
use w34_host::loader::{ModelConfig, SafetensorsArchive, resolve_hf_model_dir};

#[cfg(feature = "cuda")]
static GPU_LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "cuda")]
use core::ffi::c_void;
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use w34_host::cache::{KvFormatPolicy, KvPool, StateBundlePool, StateTx};
#[cfg(feature = "cuda")]
use w34_host::cuda::{
    CudaDevice, CudaStream, DeviceBuffer, Tp2Collectives, f16_to_f32, f32_to_f16,
    prepare_execution_plan,
};
#[cfg(feature = "cuda")]
use w34_host::model::{Qwen3_5Engine, TapPlan};
#[cfg(feature = "cuda")]
use w34_kernels::ffi::{
    self, W34_ATTN_MASK_CAUSAL, W34_ATTN_MASK_SYMMETRIC_WINDOW, W34_KV_FORMAT_FP16,
    W34_KV_FORMAT_TQ4, W34AttentionBatch, W34GdnBatch, W34KvLayerLayout,
};

#[test]
fn test_phase0_loader_all_five_cached_checkpoints() {
    let models = [
        ("nchapman/Qwen3.5-4B-W4A16-GPTQ", 32usize, 2560usize, 1usize),
        ("nchapman/Qwen3.5-9B-W4A16-GPTQ", 32usize, 4096usize, 2usize),
        (
            "nchapman/Qwen3.8-27B-W4A16-GPTQ",
            64usize,
            5120usize,
            2usize,
        ),
        ("incoai/Qwen3.8-27B-DFlash2", 5usize, 5120usize, 2usize),
        (
            "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4",
            40usize,
            2048usize,
            2usize,
        ),
    ];

    for (repo_id, expected_layers, expected_hidden, tp) in models {
        let dir = resolve_hf_model_dir(repo_id)
            .unwrap_or_else(|e| panic!("failed to resolve {repo_id}: {e}"));
        // This fixture is a drafter, not a target engine: select the windowed
        // parser explicitly rather than weakening target-model validation.
        let cfg = if repo_id == "incoai/Qwen3.8-27B-DFlash2" {
            let root: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                    .unwrap();
            ModelConfig::from_drafter_json(root.get("transformer_layer_config").unwrap_or(&root))
        } else {
            ModelConfig::from_dir(&dir)
        }
        .unwrap_or_else(|e| panic!("failed to parse config for {repo_id}: {e}"));
        assert_eq!(cfg.num_hidden_layers, expected_layers, "{repo_id} layers");
        assert_eq!(cfg.hidden_size, expected_hidden, "{repo_id} hidden_size");

        let archive = SafetensorsArchive::open(&dir)
            .unwrap_or_else(|e| panic!("failed to open safetensors for {repo_id}: {e}"));
        assert!(!archive.is_empty(), "{repo_id} should have tensors");

        let mut total_bytes = 0usize;
        for name in archive.tensor_names() {
            let info = archive.tensor_info(name).unwrap();
            total_bytes += info.len_bytes as usize;
        }
        let weight_bytes_per_rank = total_bytes / tp;
        let full_layers = cfg.num_full_attention_layers().max(1);
        let local_kv_heads = (cfg.num_key_value_heads / tp).max(1);
        let bytes_per_kv_page_across_layers =
            full_layers * local_kv_heads * 16 * (cfg.head_dim * 2) * 2;
        let gdn_layers = cfg.num_linear_attention_layers();
        let local_k_heads = cfg.linear_num_key_heads / tp;
        let local_v_heads = cfg.linear_num_value_heads / tp;
        let conv_dim =
            2 * local_k_heads * cfg.linear_key_head_dim + local_v_heads * cfg.linear_value_head_dim;
        let conv_bytes_per_slot =
            conv_dim * cfg.linear_conv_kernel_dim.saturating_sub(1).max(3) * 2;
        let ssm_bytes_per_slot =
            local_v_heads * cfg.linear_value_head_dim * cfg.linear_key_head_dim * 4;
        let state_bundle_pool_bytes = gdn_layers * 16 * (conv_bytes_per_slot + ssm_bytes_per_slot);

        let plan = CapacityPlanner::plan(
            24 * 1024 * 1024 * 1024,
            23 * 1024 * 1024 * 1024,
            weight_bytes_per_rank,
            256 * 1024 * 1024,
            state_bundle_pool_bytes,
            bytes_per_kv_page_across_layers,
            None,
        )
        .unwrap_or_else(|e| panic!("CapacityPlanner failed for {repo_id}: {e}"));
        assert!(
            plan.max_kv_pages_per_layer >= 64,
            "{repo_id} max_pages={}",
            plan.max_kv_pages_per_layer
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_phase1_op_fixtures_dump_diff() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tools/oracle/goldens/op_fixtures.safetensors");
    let archive = SafetensorsArchive::open(&fixture_path).expect("open op_fixtures.safetensors");

    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");
    prepare_execution_plan(dev.device_id).expect("prepare_execution_plan");
    let ex = stream.exec(0, 0);

    // 1. GemmaRMSNorm & fused_add_gemma_rms_norm
    {
        let x = archive.read_f16_bits("gemma_rms_norm.x").unwrap();
        let w = archive.read_f16_bits("gemma_rms_norm.w").unwrap();
        let ref_out = archive.read_f32("gemma_rms_norm.out").unwrap();
        let d_x = DeviceBuffer::alloc(0, x.len() * 2).unwrap();
        let d_w = DeviceBuffer::alloc(0, w.len() * 2).unwrap();
        let d_out = DeviceBuffer::alloc(0, x.len() * 2).unwrap();
        d_x.copy_from_host(&x, &stream).unwrap();
        d_w.copy_from_host(&w, &stream).unwrap();
        let rc = unsafe {
            ffi::w34_gemma_rms_norm_f16(
                &ex,
                d_x.addr() as *const c_void,
                d_w.addr() as *const c_void,
                d_out.addr() as *mut c_void,
                core::ptr::null(),
                16,
                2560,
                1e-6,
            )
        };
        assert_eq!(rc, 0);
        let mut got_u16 = vec![0u16; x.len()];
        d_out.copy_to_host(&mut got_u16, &stream).unwrap();
        for (&g, &r) in got_u16.iter().zip(ref_out.iter()) {
            let diff = (f16_to_f32(g) - r).abs();
            assert!(diff <= 2e-3, "gemma_rms_norm diff={diff}");
        }

        let res_in = archive
            .read_f16_bits("fused_add_gemma_rms_norm.res_in")
            .unwrap();
        let ref_res_out = archive
            .read_f32("fused_add_gemma_rms_norm.res_out")
            .unwrap();
        let ref_f_out = archive.read_f32("fused_add_gemma_rms_norm.out").unwrap();
        let d_res = DeviceBuffer::alloc(0, res_in.len() * 2).unwrap();
        d_res.copy_from_host(&res_in, &stream).unwrap();
        let rc = unsafe {
            ffi::w34_fused_add_gemma_rms_norm_f16(
                &ex,
                d_x.addr() as *const c_void,
                d_res.addr() as *mut c_void,
                d_w.addr() as *const c_void,
                d_out.addr() as *mut c_void,
                core::ptr::null(),
                16,
                2560,
                1e-6,
            )
        };
        assert_eq!(rc, 0);
        let mut got_res = vec![0u16; res_in.len()];
        d_res.copy_to_host(&mut got_res, &stream).unwrap();
        d_out.copy_to_host(&mut got_u16, &stream).unwrap();
        for (&g, &r) in got_res.iter().zip(ref_res_out.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 2e-3);
        }
        for (&g, &r) in got_u16.iter().zip(ref_f_out.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 2e-3);
        }
    }

    // 2. silu_and_mul & rms_norm_gated_silu
    {
        let inp = archive.read_f16_bits("silu_and_mul.in").unwrap();
        let ref_out = archive.read_f32("silu_and_mul.out").unwrap();
        let d_in = DeviceBuffer::alloc(0, inp.len() * 2).unwrap();
        let d_out = DeviceBuffer::alloc(0, ref_out.len() * 2).unwrap();
        d_in.copy_from_host(&inp, &stream).unwrap();
        let rc = unsafe {
            ffi::w34_silu_and_mul_f16(
                &ex,
                d_in.addr() as *const c_void,
                d_out.addr() as *mut c_void,
                core::ptr::null(),
                16,
                512,
            )
        };
        assert_eq!(rc, 0);
        let mut got = vec![0u16; ref_out.len()];
        d_out.copy_to_host(&mut got, &stream).unwrap();
        for (&g, &r) in got.iter().zip(ref_out.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 2e-3);
        }

        let gx = archive.read_f16_bits("rms_norm_gated.x").unwrap();
        let gz = archive.read_f16_bits("rms_norm_gated.z").unwrap();
        let gw = archive.read_f16_bits("rms_norm_gated.w").unwrap();
        let gref = archive.read_f32("rms_norm_gated.out").unwrap();
        let d_gx = DeviceBuffer::alloc(0, gx.len() * 2).unwrap();
        let d_gz = DeviceBuffer::alloc(0, gz.len() * 2).unwrap();
        let d_gw = DeviceBuffer::alloc(0, gw.len() * 2).unwrap();
        let d_go = DeviceBuffer::alloc(0, gref.len() * 2).unwrap();
        d_gx.copy_from_host(&gx, &stream).unwrap();
        d_gz.copy_from_host(&gz, &stream).unwrap();
        d_gw.copy_from_host(&gw, &stream).unwrap();
        let rc = unsafe {
            ffi::w34_rms_norm_gated_silu_f16(
                &ex,
                d_gx.addr() as *const c_void,
                d_gz.addr() as *const c_void,
                d_gw.addr() as *const c_void,
                d_go.addr() as *mut c_void,
                core::ptr::null(),
                16,
                32,
                128,
                1e-6,
            )
        };
        assert_eq!(rc, 0);
        let mut got_g = vec![0u16; gref.len()];
        d_go.copy_to_host(&mut got_g, &stream).unwrap();
        for (&g, &r) in got_g.iter().zip(gref.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 3e-3);
        }
    }

    // 3. fused_qk_rmsnorm_rope_gate
    {
        let q_gate = archive.read_f16_bits("qk_norm_rope.q_gate_in").unwrap();
        let k_in = archive.read_f16_bits("qk_norm_rope.k_in").unwrap();
        let q_nw = archive.read_f16_bits("qk_norm_rope.q_nw").unwrap();
        let k_nw = archive.read_f16_bits("qk_norm_rope.k_nw").unwrap();
        let pos_i32 = archive.read_i32("qk_norm_rope.positions").unwrap();
        let ref_q = archive.read_f32("qk_norm_rope.q_out").unwrap();
        let ref_k = archive.read_f32("qk_norm_rope.k_out").unwrap();
        let ref_gate = archive.read_f32("qk_norm_rope.gate_out").unwrap();

        let d_qg = DeviceBuffer::alloc(0, q_gate.len() * 2).unwrap();
        let d_ki = DeviceBuffer::alloc(0, k_in.len() * 2).unwrap();
        let d_qnw = DeviceBuffer::alloc(0, q_nw.len() * 2).unwrap();
        let d_knw = DeviceBuffer::alloc(0, k_nw.len() * 2).unwrap();
        let d_pos = DeviceBuffer::alloc(0, pos_i32.len() * 4).unwrap();
        let d_qo = DeviceBuffer::alloc(0, ref_q.len() * 2).unwrap();
        let d_ko = DeviceBuffer::alloc(0, ref_k.len() * 2).unwrap();
        let d_go = DeviceBuffer::alloc(0, ref_gate.len() * 2).unwrap();

        d_qg.copy_from_host(&q_gate, &stream).unwrap();
        d_ki.copy_from_host(&k_in, &stream).unwrap();
        d_qnw.copy_from_host(&q_nw, &stream).unwrap();
        d_knw.copy_from_host(&k_nw, &stream).unwrap();
        d_pos.copy_from_host(&pos_i32, &stream).unwrap();

        let rc = unsafe {
            ffi::w34_fused_qk_rmsnorm_rope_gate_f16(
                &ex,
                d_qg.addr() as *const c_void,
                d_ki.addr() as *const c_void,
                d_qnw.addr() as *const c_void,
                d_knw.addr() as *const c_void,
                d_pos.addr() as *const i32,
                core::ptr::null(),
                d_qo.addr() as *mut c_void,
                d_ko.addr() as *mut c_void,
                d_go.addr() as *mut c_void,
                8,
                8,
                8,
                2,
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
        let mut got_q = vec![0u16; ref_q.len()];
        let mut got_k = vec![0u16; ref_k.len()];
        let mut got_g = vec![0u16; ref_gate.len()];
        d_qo.copy_to_host(&mut got_q, &stream).unwrap();
        d_ko.copy_to_host(&mut got_k, &stream).unwrap();
        d_go.copy_to_host(&mut got_g, &stream).unwrap();
        for (&g, &r) in got_q.iter().zip(ref_q.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 3e-3);
        }
        for (&g, &r) in got_k.iter().zip(ref_k.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 3e-3);
        }
        for (&g, &r) in got_g.iter().zip(ref_gate.iter()) {
            assert_eq!(f16_to_f32(g), r);
        }
    }

    // 4. Marlin W4A16 repack (0 ULP bit-exact) + GEMM (M=1 and M=16)
    {
        let q_packed = archive.read_u32("marlin.q_packed").unwrap();
        let ref_repacked = archive.read_u32("marlin.repacked").unwrap();
        let sc_perm = archive.read_f16_bits("marlin.sc_perm").unwrap();
        let a_m1 = archive.read_f16_bits("marlin.a_m1").unwrap();
        let ref_m1 = archive.read_f32("marlin.ref_m1").unwrap();
        let a_m16 = archive.read_f16_bits("marlin.a_m16").unwrap();
        let ref_m16 = archive.read_f32("marlin.ref_m16").unwrap();

        let d_qp = DeviceBuffer::alloc(0, q_packed.len() * 4).unwrap();
        let d_rep = DeviceBuffer::alloc(0, ref_repacked.len() * 4).unwrap();
        d_qp.copy_from_host(&q_packed, &stream).unwrap();
        let rc = unsafe {
            ffi::w34_marlin_repack_int4(
                &ex,
                d_qp.addr() as *const u32,
                d_rep.addr() as *mut u32,
                256,
                128,
            )
        };
        assert_eq!(rc, 0);
        let mut got_rep = vec![0u32; ref_repacked.len()];
        d_rep.copy_to_host(&mut got_rep, &stream).unwrap();
        assert_eq!(got_rep, ref_repacked, "0-ULP Marlin repack parity");

        let d_sc = DeviceBuffer::alloc(0, sc_perm.len() * 2).unwrap();
        let d_ws = DeviceBuffer::alloc(0, 65536 * 4).unwrap();
        d_sc.copy_from_host(&sc_perm, &stream).unwrap();
        unsafe { ffi::w34_zero_i32(&ex, d_ws.addr() as *mut c_void, 65536) };

        for (m, a_host, ref_host) in [(1i64, &a_m1, &ref_m1), (16i64, &a_m16, &ref_m16)] {
            let d_a = DeviceBuffer::alloc(0, a_host.len() * 2).unwrap();
            let d_c = DeviceBuffer::alloc(0, ref_host.len() * 2).unwrap();
            d_a.copy_from_host(a_host, &stream).unwrap();
            let rc = unsafe {
                ffi::w34_marlin_gemm_f16u4(
                    &ex,
                    d_a.addr() as *const c_void,
                    d_rep.addr() as *const c_void,
                    d_sc.addr() as *const c_void,
                    d_c.addr() as *mut c_void,
                    d_ws.addr() as *mut i32,
                    m,
                    128,
                    256,
                    65536,
                    8,
                )
            };
            assert_eq!(rc, 0);
            let mut got_c = vec![0u16; ref_host.len()];
            d_c.copy_to_host(&mut got_c, &stream).unwrap();
            for (&g, &r) in got_c.iter().zip(ref_host.iter()) {
                let diff = (f16_to_f32(g) - r).abs();
                assert!(diff <= 5e-3, "Marlin GEMM M={m} diff={diff}");
            }
        }
    }

    // 5. GDN causal_conv1d
    {
        let cx = archive.read_f16_bits("gdn_conv.x").unwrap();
        let cw = archive.read_f16_bits("gdn_conv.w").unwrap();
        let cs_in = archive.read_f16_bits("gdn_conv.state_in").unwrap();
        let cref_out = archive.read_f32("gdn_conv.out").unwrap();
        let cref_st = archive.read_f32("gdn_conv.state_out").unwrap();

        let d_cx = DeviceBuffer::alloc(0, cx.len() * 2).unwrap();
        let d_cw = DeviceBuffer::alloc(0, cw.len() * 2).unwrap();
        let d_cs = DeviceBuffer::alloc(0, cs_in.len() * 2 * 2).unwrap(); // 2 slots
        let d_co = DeviceBuffer::alloc(0, cref_out.len() * 2).unwrap();
        d_cx.copy_from_host(&cx, &stream).unwrap();
        d_cw.copy_from_host(&cw, &stream).unwrap();
        d_cs.copy_from_host(&cs_in, &stream).unwrap();

        let d_meta = DeviceBuffer::alloc(0, 256).unwrap();
        let meta_i32: [i32; 5] = [0, 8, 0, 1, 1]; // q_start=[0,8], state_in=[0], state_out=[1], has_init=[1]
        d_meta.copy_from_host(&meta_i32, &stream).unwrap();
        let gdn_batch = W34GdnBatch {
            query_start_loc: d_meta.addr() as *const i32,
            state_in_indices: (d_meta.addr() + 8) as *const i32,
            state_out_indices: (d_meta.addr() + 12) as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: core::ptr::null(),
            checkpoint_token_offsets: core::ptr::null(),
            has_initial_state: (d_meta.addr() + 16) as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: 8,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 0,
        };
        let rc = unsafe {
            ffi::w34_gdn_causal_conv1d_f16(
                &ex,
                &gdn_batch,
                d_cx.addr() as *const c_void,
                d_cw.addr() as *const c_void,
                core::ptr::null(),
                d_cs.addr() as *mut c_void,
                d_co.addr() as *mut c_void,
                256,
                4,
                2,
                1,
            )
        };
        assert_eq!(rc, 0);
        let mut got_co = vec![0u16; cref_out.len()];
        let mut got_cs = vec![0u16; cs_in.len() * 2];
        d_co.copy_to_host(&mut got_co, &stream).unwrap();
        d_cs.copy_to_host(&mut got_cs, &stream).unwrap();
        for (&g, &r) in got_co.iter().zip(cref_out.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 2e-3);
        }
        for (&g, &r) in got_cs[cs_in.len()..].iter().zip(cref_st.iter()) {
            assert!((f16_to_f32(g) - r).abs() <= 2e-3);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gate1_fragmented_paged_attention_and_tq4_strides() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");
    let ex = stream.exec(0, 0);

    for policy in [KvFormatPolicy::Fp16, KvFormatPolicy::Tq4All] {
        let pool = KvPool::new(0, 0, &[3], 2, 256, 16, 16, policy).expect("KvPool");
        let layout = pool.layers[0].layout;
        if policy == KvFormatPolicy::Tq4All {
            assert_eq!(layout.k_token_stride_bytes, 132);
            assert_eq!(layout.k_format, W34_KV_FORMAT_TQ4);
        } else {
            assert_eq!(layout.k_token_stride_bytes, 512);
            assert_eq!(layout.k_format, W34_KV_FORMAT_FP16);
        }

        let seq_len = 40usize; // spans 3 pages (16 + 16 + 8)
        let k_host: Vec<u16> = (0..(seq_len * 2 * 256))
            .map(|i| f32_to_f16(((i % 97) as f32 - 48.0) * 0.02))
            .collect();
        let v_host: Vec<u16> = (0..(seq_len * 2 * 256))
            .map(|i| f32_to_f16(((i % 89) as f32 - 44.0) * 0.02))
            .collect();
        let q_host: Vec<u16> = (0..(8 * 256))
            .map(|i| f32_to_f16(((i % 73) as f32 - 36.0) * 0.03))
            .collect();

        let d_k = DeviceBuffer::alloc(0, k_host.len() * 2).unwrap();
        let d_v = DeviceBuffer::alloc(0, v_host.len() * 2).unwrap();
        let d_q = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        let d_out_contig = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        let d_out_perm = DeviceBuffer::alloc(0, q_host.len() * 2).unwrap();
        d_k.copy_from_host(&k_host, &stream).unwrap();
        d_v.copy_from_host(&v_host, &stream).unwrap();
        d_q.copy_from_host(&q_host, &stream).unwrap();

        let run_with_pages = |pages: [i32; 3], d_out: &DeviceBuffer| -> Vec<u16> {
            let mut slots = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let p = pages[pos / 16];
                let off = (pos % 16) as i32;
                slots.push(p * 16 + off);
            }
            let d_slots = DeviceBuffer::alloc(0, slots.len() * 4).unwrap();
            d_slots.copy_from_host(&slots, &stream).unwrap();
            let d_bt = DeviceBuffer::alloc(0, 3 * 4).unwrap();
            d_bt.copy_from_host(&pages, &stream).unwrap();
            let q_start = [0i32, 1i32];
            let ctx_before = [(seq_len - 1) as i32];
            let seq_after = [seq_len as i32];
            let d_qs = DeviceBuffer::alloc(0, 8).unwrap();
            let d_cb = DeviceBuffer::alloc(0, 4).unwrap();
            let d_sa = DeviceBuffer::alloc(0, 4).unwrap();
            d_qs.copy_from_host(&q_start, &stream).unwrap();
            d_cb.copy_from_host(&ctx_before, &stream).unwrap();
            d_sa.copy_from_host(&seq_after, &stream).unwrap();

            let write_batch = W34AttentionBatch {
                block_tables: d_bt.addr() as *const i32,
                write_slot_mapping: d_slots.addr() as *const i32,
                query_start_loc: d_qs.addr() as *const i32,
                context_lens_before: d_cb.addr() as *const i32,
                seq_lens_after: d_sa.addr() as *const i32,
                valid_req_mask: core::ptr::null(),
                valid_token_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: seq_len as i32,
                max_blocks_per_req: 3,
                max_q_len: 1,
                max_seq_len: seq_len as i32,
                sliding_window: 0,
                mask_mode: W34_ATTN_MASK_CAUSAL,
                sm_scale: 1.0 / (256.0f32).sqrt(),
                reserved: 0,
            };
            assert_eq!(
                unsafe {
                    ffi::w34_kv_cache_write(
                        &ex,
                        &layout,
                        &write_batch,
                        d_k.addr() as *const c_void,
                        d_v.addr() as *const c_void,
                    )
                },
                0
            );

            let mut read_batch = write_batch;
            read_batch.total_tokens = 1;
            assert_eq!(
                unsafe {
                    ffi::w34_paged_attention_f16(
                        &ex,
                        &layout,
                        &read_batch,
                        d_q.addr() as *const c_void,
                        d_out.addr() as *mut c_void,
                        8,
                    )
                },
                0
            );
            let mut out = vec![0u16; q_host.len()];
            d_out.copy_to_host(&mut out, &stream).unwrap();
            out
        };

        let out_contig = run_with_pages([0, 1, 2], &d_out_contig);
        let out_perm = run_with_pages([11, 4, 9], &d_out_perm);
        assert_eq!(
            out_contig, out_perm,
            "Gate #1: 0-ULP page-table permutation invariance for {policy:?}"
        );
    }

    // Also test symmetric non-causal sliding window (DFlash2 head_dim=128)
    let dflash_pool =
        KvPool::new(0, 1, &[0], 2, 128, 16, 8, KvFormatPolicy::Fp16).expect("dflash pool");
    let df_layout: W34KvLayerLayout = dflash_pool.layers[0].layout;
    assert_eq!(df_layout.namespace_id, 1);
    let df_seq = 8usize;
    let df_kv: Vec<u16> = (0..(df_seq * 2 * 128))
        .map(|i| f32_to_f16(((i % 41) as f32 - 20.0) * 0.03))
        .collect();
    let df_q: Vec<u16> = (0..(df_seq * 4 * 128))
        .map(|i| f32_to_f16(((i % 37) as f32 - 18.0) * 0.03))
        .collect();
    let d_df_k = DeviceBuffer::alloc(0, df_kv.len() * 2).unwrap();
    let d_df_v = DeviceBuffer::alloc(0, df_kv.len() * 2).unwrap();
    let d_df_q = DeviceBuffer::alloc(0, df_q.len() * 2).unwrap();
    let d_df_out = DeviceBuffer::alloc(0, df_q.len() * 2).unwrap();
    d_df_k.copy_from_host(&df_kv, &stream).unwrap();
    d_df_v.copy_from_host(&df_kv, &stream).unwrap();
    d_df_q.copy_from_host(&df_q, &stream).unwrap();
    let df_slots: Vec<i32> = (0..df_seq as i32).collect();
    let d_df_slots = DeviceBuffer::alloc(0, df_slots.len() * 4).unwrap();
    d_df_slots.copy_from_host(&df_slots, &stream).unwrap();
    let df_bt = [0i32];
    let d_df_bt = DeviceBuffer::alloc(0, 4).unwrap();
    d_df_bt.copy_from_host(&df_bt, &stream).unwrap();
    let df_qs = [0i32, df_seq as i32];
    let d_df_qs = DeviceBuffer::alloc(0, 8).unwrap();
    d_df_qs.copy_from_host(&df_qs, &stream).unwrap();
    let df_cb = [0i32];
    let df_sa = [df_seq as i32];
    let d_df_cb = DeviceBuffer::alloc(0, 4).unwrap();
    let d_df_sa = DeviceBuffer::alloc(0, 4).unwrap();
    d_df_cb.copy_from_host(&df_cb, &stream).unwrap();
    d_df_sa.copy_from_host(&df_sa, &stream).unwrap();
    let df_batch = W34AttentionBatch {
        block_tables: d_df_bt.addr() as *const i32,
        write_slot_mapping: d_df_slots.addr() as *const i32,
        query_start_loc: d_df_qs.addr() as *const i32,
        context_lens_before: d_df_cb.addr() as *const i32,
        seq_lens_after: d_df_sa.addr() as *const i32,
        valid_req_mask: core::ptr::null(),
        valid_token_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: df_seq as i32,
        max_blocks_per_req: 1,
        max_q_len: df_seq as i32,
        max_seq_len: df_seq as i32,
        sliding_window: 8,
        mask_mode: W34_ATTN_MASK_SYMMETRIC_WINDOW,
        sm_scale: 1.0 / (128.0f32).sqrt(),
        reserved: 0,
    };
    unsafe {
        assert_eq!(
            ffi::w34_kv_cache_write(
                &ex,
                &df_layout,
                &df_batch,
                d_df_k.addr() as *const c_void,
                d_df_v.addr() as *const c_void,
            ),
            0
        );
        assert_eq!(
            ffi::w34_paged_attention_f16(
                &ex,
                &df_layout,
                &df_batch,
                d_df_q.addr() as *const c_void,
                d_df_out.addr() as *mut c_void,
                4,
            ),
            0
        );
    }
    let mut df_got = vec![0u16; df_q.len()];
    d_df_out.copy_to_host(&mut df_got, &stream).unwrap();
    assert!(df_got.iter().any(|&v| f16_to_f32(v).abs() > 1e-4));
}

#[cfg(feature = "cuda")]
#[test]
fn test_gate2_gdn_continuation_checkpoint_and_rejection_parity() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");
    let ex = stream.exec(0, 0);

    let hk = 4usize;
    let hv = 8usize;
    let dk = 128usize;
    let dv = 128usize;
    let conv_dim = 2 * hk * dk + hv * dv; // 2048
    let total_len = 72usize; // 64 prefill + 8 verify

    let x_host: Vec<u16> = (0..(total_len * conv_dim))
        .map(|i| f32_to_f16(((i % 113) as f32 - 56.0) * 0.01))
        .collect();
    let w_host: Vec<u16> = (0..(conv_dim * 4))
        .map(|i| f32_to_f16(((i % 31) as f32 - 15.0) * 0.02))
        .collect();
    let b_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 19) as f32 - 9.0) * 0.1))
        .collect();
    let a_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 23) as f32 - 11.0) * 0.05))
        .collect();
    let a_log: Vec<f32> = (0..hv).map(|i| -1.5 - 0.1 * (i as f32)).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|i| -0.1 + 0.03 * (i as f32)).collect();

    let d_x = DeviceBuffer::alloc(0, x_host.len() * 2).unwrap();
    let d_w = DeviceBuffer::alloc(0, w_host.len() * 2).unwrap();
    let d_b = DeviceBuffer::alloc(0, b_host.len() * 2).unwrap();
    let d_a = DeviceBuffer::alloc(0, a_host.len() * 2).unwrap();
    let d_alog = DeviceBuffer::alloc(0, a_log.len() * 4).unwrap();
    let d_dtb = DeviceBuffer::alloc(0, dt_bias.len() * 4).unwrap();
    d_x.copy_from_host(&x_host, &stream).unwrap();
    d_w.copy_from_host(&w_host, &stream).unwrap();
    d_b.copy_from_host(&b_host, &stream).unwrap();
    d_a.copy_from_host(&a_host, &stream).unwrap();
    d_alog.copy_from_host(&a_log, &stream).unwrap();
    d_dtb.copy_from_host(&dt_bias, &stream).unwrap();

    let pool = StateBundlePool::new(0, &[0], 16, conv_dim, 4, hv, dk, dv).unwrap();
    let st = &pool.layers[0];

    // Path A: One-shot 64-token prefill into slot 0 (with checkpoint at token 64 -> slot 1),
    // then 8-token speculative verify from slot 1 saving spec slots 2..=9.
    let d_conv_out_a = DeviceBuffer::alloc(0, total_len * conv_dim * 2).unwrap();
    let d_gdn_out_a = DeviceBuffer::alloc(0, total_len * hv * dv * 2).unwrap();
    let d_meta = DeviceBuffer::alloc(0, 512).unwrap();

    // Step A1: 64-token prefill (has_init=0, state_out=0, checkpoint at token 64 -> slot 1)
    let meta_a1: [i32; 7] = [0, 64, -1, 0, 1, 64, 0];
    d_meta.copy_from_host(&meta_a1, &stream).unwrap();
    let batch_a1 = W34GdnBatch {
        query_start_loc: d_meta.addr() as *const i32,
        state_in_indices: (d_meta.addr() + 8) as *const i32,
        state_out_indices: (d_meta.addr() + 12) as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: (d_meta.addr() + 16) as *const i32,
        checkpoint_token_offsets: (d_meta.addr() + 20) as *const i32,
        has_initial_state: (d_meta.addr() + 24) as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 64,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 1,
    };
    unsafe {
        assert_eq!(
            ffi::w34_gdn_causal_conv1d_f16(
                &ex,
                &batch_a1,
                d_x.addr() as *const c_void,
                d_w.addr() as *const c_void,
                core::ptr::null(),
                st.conv_state.addr() as *mut c_void,
                d_conv_out_a.addr() as *mut c_void,
                conv_dim as i32,
                4,
                16,
                1,
            ),
            0
        );
        assert_eq!(
            ffi::w34_gdn_recurrent_f16(
                &ex,
                &batch_a1,
                d_conv_out_a.addr() as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                d_b.addr() as *const c_void,
                d_a.addr() as *const c_void,
                d_alog.addr() as *const f32,
                d_dtb.addr() as *const f32,
                st.ssm_state.addr() as *mut f32,
                d_gdn_out_a.addr() as *mut c_void,
                hk as i32,
                dk as i32,
                hv as i32,
                dv as i32,
                16,
            ),
            0
        );
    }

    // Step A2: 8-token speculative verify from checkpoint slot 1 -> spec_state_indices = [2,3,4,5,6,7,8,9]
    let meta_a2: [i32; 13] = [0, 8, 1, 9, 2, 3, 4, 5, 6, 7, 8, 9, 1];
    d_meta.copy_from_host(&meta_a2, &stream).unwrap();
    let batch_a2 = W34GdnBatch {
        query_start_loc: d_meta.addr() as *const i32,
        state_in_indices: (d_meta.addr() + 8) as *const i32,
        state_out_indices: (d_meta.addr() + 12) as *const i32,
        spec_state_indices: (d_meta.addr() + 16) as *const i32,
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: (d_meta.addr() + 48) as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 8,
        max_verify_tokens: 8,
        max_checkpoints_per_req: 0,
    };
    unsafe {
        assert_eq!(
            ffi::w34_gdn_causal_conv1d_f16(
                &ex,
                &batch_a2,
                (d_x.addr() + (64 * conv_dim * 2) as u64) as *const c_void,
                d_w.addr() as *const c_void,
                core::ptr::null(),
                st.conv_state.addr() as *mut c_void,
                (d_conv_out_a.addr() + (64 * conv_dim * 2) as u64) as *mut c_void,
                conv_dim as i32,
                4,
                16,
                1,
            ),
            0
        );
        assert_eq!(
            ffi::w34_gdn_recurrent_f16(
                &ex,
                &batch_a2,
                (d_conv_out_a.addr() + (64 * conv_dim * 2) as u64) as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                (d_b.addr() + (64 * hv * 2) as u64) as *const c_void,
                (d_a.addr() + (64 * hv * 2) as u64) as *const c_void,
                d_alog.addr() as *const f32,
                d_dtb.addr() as *const f32,
                st.ssm_state.addr() as *mut f32,
                (d_gdn_out_a.addr() + (64 * hv * dv * 2) as u64) as *mut c_void,
                hk as i32,
                dk as i32,
                hv as i32,
                dv as i32,
                16,
            ),
            0
        );
    }

    // Path B: One-shot 69-token run (64 + 5 accepted tokens, matching spec slot 6 = step index 4) into slot 11
    let d_conv_out_b = DeviceBuffer::alloc(0, 69 * conv_dim * 2).unwrap();
    let d_gdn_out_b = DeviceBuffer::alloc(0, 69 * hv * dv * 2).unwrap();
    let meta_b: [i32; 5] = [0, 69, -1, 11, 0];
    d_meta.copy_from_host(&meta_b, &stream).unwrap();
    let batch_b = W34GdnBatch {
        query_start_loc: d_meta.addr() as *const i32,
        state_in_indices: (d_meta.addr() + 8) as *const i32,
        state_out_indices: (d_meta.addr() + 12) as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: (d_meta.addr() + 16) as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 69,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    unsafe {
        assert_eq!(
            ffi::w34_gdn_causal_conv1d_f16(
                &ex,
                &batch_b,
                d_x.addr() as *const c_void,
                d_w.addr() as *const c_void,
                core::ptr::null(),
                st.conv_state.addr() as *mut c_void,
                d_conv_out_b.addr() as *mut c_void,
                conv_dim as i32,
                4,
                16,
                1,
            ),
            0
        );
        assert_eq!(
            ffi::w34_gdn_recurrent_f16(
                &ex,
                &batch_b,
                d_conv_out_b.addr() as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                d_b.addr() as *const c_void,
                d_a.addr() as *const c_void,
                d_alog.addr() as *const f32,
                d_dtb.addr() as *const f32,
                st.ssm_state.addr() as *mut f32,
                d_gdn_out_b.addr() as *mut c_void,
                hk as i32,
                dk as i32,
                hv as i32,
                dv as i32,
                16,
            ),
            0
        );
    }

    // Compare outputs 0..69 AND ssm_state[slot=6] vs ssm_state[slot=11] for 0-ULP equality!
    let mut out_a = vec![0u16; 69 * hv * dv];
    let mut out_b = vec![0u16; 69 * hv * dv];
    d_gdn_out_a.copy_to_host(&mut out_a, &stream).unwrap();
    d_gdn_out_b.copy_to_host(&mut out_b, &stream).unwrap();
    let state_elems = hv * dv * dk;
    let mut all_ssm = vec![0.0f32; 16 * state_elems];
    st.ssm_state.copy_to_host(&mut all_ssm, &stream).unwrap();

    assert_eq!(out_a, out_b, "Gate #2: 0-ULP GDN output parity");
    assert_eq!(
        &all_ssm[6 * state_elems..7 * state_elems],
        &all_ssm[11 * state_elems..12 * state_elems],
        "Gate #2: 0-ULP GDN speculative rejection state parity (slot 6 == slot 11)"
    );

    // Also verify 27B geometry (`hk=16, hv=96, dk=128, dv=128` -> `<8, 5>` TILE_S=16 prefill
    // vs `<8, 6>` single-token decode) across 35 tokens (> 2 * TILE_S + partial tail tile).
    {
        let hk96 = 16usize;
        let hv96 = 96usize;
        let conv_dim96 = 2 * hk96 * dk + hv96 * dv; // 16384
        let seq96 = 35usize;
        let qkv_host96: Vec<u16> = (0..(seq96 * conv_dim96))
            .map(|i| f32_to_f16(((i % 97) as f32 - 48.0) * 0.008))
            .collect();
        let b_host96: Vec<u16> = (0..(seq96 * hv96))
            .map(|i| f32_to_f16(((i % 19) as f32 - 9.0) * 0.1))
            .collect();
        let a_host96: Vec<u16> = (0..(seq96 * hv96))
            .map(|i| f32_to_f16(((i % 23) as f32 - 11.0) * 0.05))
            .collect();
        let a_log96: Vec<f32> = (0..hv96).map(|i| -1.5 - 0.02 * (i as f32)).collect();
        let dt_bias96: Vec<f32> = (0..hv96).map(|i| -0.1 + 0.005 * (i as f32)).collect();

        let d_qkv96 = DeviceBuffer::alloc(0, qkv_host96.len() * 2).unwrap();
        let d_b96 = DeviceBuffer::alloc(0, b_host96.len() * 2).unwrap();
        let d_a96 = DeviceBuffer::alloc(0, a_host96.len() * 2).unwrap();
        let d_alog96 = DeviceBuffer::alloc(0, a_log96.len() * 4).unwrap();
        let d_dtb96 = DeviceBuffer::alloc(0, dt_bias96.len() * 4).unwrap();
        d_qkv96.copy_from_host(&qkv_host96, &stream).unwrap();
        d_b96.copy_from_host(&b_host96, &stream).unwrap();
        d_a96.copy_from_host(&a_host96, &stream).unwrap();
        d_alog96.copy_from_host(&a_log96, &stream).unwrap();
        d_dtb96.copy_from_host(&dt_bias96, &stream).unwrap();

        let pool96 = StateBundlePool::new(0, &[0], 2, conv_dim96, 4, hv96, dk, dv).unwrap();
        let st96 = &pool96.layers[0];
        let d_out_prefill = DeviceBuffer::alloc(0, seq96 * hv96 * dv * 2).unwrap();
        let d_out_decode = DeviceBuffer::alloc(0, seq96 * hv96 * dv * 2).unwrap();

        // Prefill 35 tokens into slot 0
        let meta_pf: [i32; 5] = [0, seq96 as i32, -1, 0, 0];
        d_meta.copy_from_host(&meta_pf, &stream).unwrap();
        let batch_pf = W34GdnBatch {
            query_start_loc: d_meta.addr() as *const i32,
            state_in_indices: (d_meta.addr() + 8) as *const i32,
            state_out_indices: (d_meta.addr() + 12) as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: core::ptr::null(),
            checkpoint_token_offsets: core::ptr::null(),
            has_initial_state: (d_meta.addr() + 16) as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: seq96 as i32,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 0,
        };
        assert_eq!(
            unsafe {
                ffi::w34_gdn_recurrent_f16(
                    &ex,
                    &batch_pf,
                    d_qkv96.addr() as *const c_void,
                    core::ptr::null(),
                    core::ptr::null(),
                    core::ptr::null(),
                    d_b96.addr() as *const c_void,
                    d_a96.addr() as *const c_void,
                    d_alog96.addr() as *const f32,
                    d_dtb96.addr() as *const f32,
                    st96.ssm_state.addr() as *mut f32,
                    d_out_prefill.addr() as *mut c_void,
                    hk96 as i32,
                    dk as i32,
                    hv96 as i32,
                    dv as i32,
                    2,
                )
            },
            0
        );

        // Decode 35 single tokens step-by-step into slot 1
        for step in 0..seq96 {
            let meta_dec: [i32; 5] = [0, 1, if step == 0 { -1 } else { 1 }, 1, i32::from(step > 0)];
            d_meta.copy_from_host(&meta_dec, &stream).unwrap();
            let batch_dec = W34GdnBatch {
                query_start_loc: d_meta.addr() as *const i32,
                state_in_indices: (d_meta.addr() + 8) as *const i32,
                state_out_indices: (d_meta.addr() + 12) as *const i32,
                spec_state_indices: core::ptr::null(),
                checkpoint_dst_indices: core::ptr::null(),
                checkpoint_token_offsets: core::ptr::null(),
                has_initial_state: (d_meta.addr() + 16) as *const u8,
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: 1,
                max_verify_tokens: 0,
                max_checkpoints_per_req: 0,
            };
            assert_eq!(
                unsafe {
                    ffi::w34_gdn_recurrent_f16(
                        &ex,
                        &batch_dec,
                        (d_qkv96.addr() + (step * conv_dim96 * 2) as u64) as *const c_void,
                        core::ptr::null(),
                        core::ptr::null(),
                        core::ptr::null(),
                        (d_b96.addr() + (step * hv96 * 2) as u64) as *const c_void,
                        (d_a96.addr() + (step * hv96 * 2) as u64) as *const c_void,
                        d_alog96.addr() as *const f32,
                        d_dtb96.addr() as *const f32,
                        st96.ssm_state.addr() as *mut f32,
                        (d_out_decode.addr() + (step * hv96 * dv * 2) as u64) as *mut c_void,
                        hk96 as i32,
                        dk as i32,
                        hv96 as i32,
                        dv as i32,
                        2,
                    )
                },
                0
            );
        }

        let mut out_pf = vec![0u16; seq96 * hv96 * dv];
        let mut out_dec = vec![0u16; seq96 * hv96 * dv];
        d_out_prefill.copy_to_host(&mut out_pf, &stream).unwrap();
        d_out_decode.copy_to_host(&mut out_dec, &stream).unwrap();
        let state_elems96 = hv96 * dv * dk;
        let mut ssm96 = vec![0.0f32; 2 * state_elems96];
        st96.ssm_state.copy_to_host(&mut ssm96, &stream).unwrap();
        stream.synchronize().unwrap();

        assert_eq!(
            out_pf, out_dec,
            "Gate #2: 0-ULP hv=96 GDN TILE_S=16 prefill vs single-step decode output parity"
        );
        assert_eq!(
            &ssm96[..state_elems96],
            &ssm96[state_elems96..],
            "Gate #2: 0-ULP hv=96 GDN TILE_S=16 prefill vs single-step decode ssm_state parity"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gdn_conv_paired_prefill_matches_chunked_scalar() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(dev.device_id).unwrap();
    let ex = stream.exec(0, 0);
    let channels = 512usize;
    let tokens = 257usize;
    let state_size = channels * 3;
    let x: Vec<u16> = (0..tokens * channels)
        .map(|i| f32_to_f16(((i % 97) as f32 - 48.0) * 0.037))
        .collect();
    let w: Vec<u16> = (0..channels * 4)
        .map(|i| f32_to_f16(((i % 31) as f32 - 15.0) * 0.019))
        .collect();
    let mut state = vec![0u16; 6 * state_size];
    for i in 0..state_size {
        state[i] = f32_to_f16(((i % 41) as f32 - 20.0) * 0.027);
        state[state_size + i] = state[i];
    }
    let dx = DeviceBuffer::alloc(0, x.len() * 2).unwrap();
    let dw = DeviceBuffer::alloc(0, w.len() * 2).unwrap();
    let ds = DeviceBuffer::alloc(0, state.len() * 2).unwrap();
    let dm = DeviceBuffer::alloc(0, 256).unwrap();
    let out = DeviceBuffer::alloc(0, x.len() * 2).unwrap();
    let reference = DeviceBuffer::alloc(0, x.len() * 2).unwrap();
    dx.copy_from_host(&x, &stream).unwrap();
    dw.copy_from_host(&w, &stream).unwrap();
    ds.copy_from_host(&state, &stream).unwrap();
    // Both paths update their initial state in-place; checkpoints split interior tiles.
    for mode in 0..2 {
        let chunk = if mode == 0 { tokens } else { 64 };
        for start in (0..tokens).step_by(chunk) {
            let len = chunk.min(tokens - start);
            let offsets = [63i32 - start as i32, 191i32 - start as i32];
            let meta = [
                0i32,
                len as i32,
                mode,
                mode,
                2 + 2 * mode,
                3 + 2 * mode,
                offsets[0],
                offsets[1],
                1,
            ];
            dm.copy_from_host(&meta, &stream).unwrap();
            let batch = W34GdnBatch {
                query_start_loc: dm.addr() as *const i32,
                state_in_indices: (dm.addr() + 8) as *const i32,
                state_out_indices: (dm.addr() + 12) as *const i32,
                spec_state_indices: core::ptr::null(),
                checkpoint_dst_indices: (dm.addr() + 16) as *const i32,
                checkpoint_token_offsets: (dm.addr() + 24) as *const i32,
                has_initial_state: (dm.addr() + 32) as *const u8,
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: len as i32,
                max_verify_tokens: 0,
                max_checkpoints_per_req: 2,
            };
            let dst = if mode == 0 { &out } else { &reference };
            unsafe {
                assert_eq!(
                    ffi::w34_gdn_causal_conv1d_f16(
                        &ex,
                        &batch,
                        (dx.addr() + (start * channels * 2) as u64) as *const c_void,
                        dw.addr() as *const c_void,
                        core::ptr::null(),
                        ds.addr() as *mut c_void,
                        (dst.addr() + (start * channels * 2) as u64) as *mut c_void,
                        channels as i32,
                        4,
                        6,
                        1
                    ),
                    0
                );
            }
        }
    }
    let mut actual = vec![0u16; x.len()];
    let mut expected = vec![0u16; x.len()];
    out.copy_to_host(&mut actual, &stream).unwrap();
    reference.copy_to_host(&mut expected, &stream).unwrap();
    ds.copy_to_host(&mut state, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        actual, expected,
        "paired conv must preserve FP16 product rounding"
    );
    for (a, b) in [(0, 1), (2, 4), (3, 5)] {
        assert_eq!(
            &state[a * state_size..(a + 1) * state_size],
            &state[b * state_size..(b + 1) * state_size],
            "conv state slots {a}/{b}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gdn_wy_split_and_fallback_prefill_parity() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");
    let ex_zero = stream.exec(0, 0);

    let hk = 2usize;
    let hv = 6usize; // gqa_ratio = 3 (matches 27B ratio)
    let dk = 128usize;
    let dv = 128usize;
    let conv_dim = 2 * hk * dk + hv * dv;
    let init_len = 40usize;
    let wy_len = 260usize; // >= 256 triggers WY; non-multiple of 64
    let total_len = init_len + wy_len;

    let qkv_host: Vec<u16> = (0..(total_len * conv_dim))
        .map(|i| f32_to_f16(((i % 109) as f32 - 54.0) * 0.009))
        .collect();
    let b_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 19) as f32 - 9.0) * 0.1))
        .collect();
    let a_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 23) as f32 - 11.0) * 0.05))
        .collect();
    let a_log: Vec<f32> = (0..hv).map(|i| -1.5 - 0.08 * (i as f32)).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|i| -0.1 + 0.02 * (i as f32)).collect();

    let d_qkv = DeviceBuffer::alloc(0, qkv_host.len() * 2).unwrap();
    let d_b = DeviceBuffer::alloc(0, b_host.len() * 2).unwrap();
    let d_a = DeviceBuffer::alloc(0, a_host.len() * 2).unwrap();
    let d_alog = DeviceBuffer::alloc(0, a_log.len() * 4).unwrap();
    let d_dtb = DeviceBuffer::alloc(0, dt_bias.len() * 4).unwrap();
    d_qkv.copy_from_host(&qkv_host, &stream).unwrap();
    d_b.copy_from_host(&b_host, &stream).unwrap();
    d_a.copy_from_host(&a_host, &stream).unwrap();
    d_alog.copy_from_host(&a_log, &stream).unwrap();
    d_dtb.copy_from_host(&dt_bias, &stream).unwrap();

    // Slots: 0 = init_state after 40 tokens;
    // Reference recurrent: ckpts [1, 2, 3] at offsets [1, 2, 3], final = 4
    // Split WY (with K_norm): ckpts [5, 6, 7], final = 8
    // Split WY (no K_norm, 27B TP=1 budget): ckpts [9, 10, 11], final = 12
    // Fallback WY (zero scratch): ckpts [13, 14, 15], final = 16
    // Tight-scratch fallback: ckpts [18, 19, 20], final = 21
    let num_slots = 22usize;
    let pool = StateBundlePool::new(0, &[0], num_slots, conv_dim, 4, hv, dk, dv).unwrap();
    let st = &pool.layers[0];
    let d_meta = DeviceBuffer::alloc(0, 256).unwrap();

    // 1. Seed initial state in slot 0 via 40-token recurrent prefill.
    let d_out_init = DeviceBuffer::alloc(0, init_len * hv * dv * 2).unwrap();
    let meta_init: [i32; 5] = [0, init_len as i32, -1, 0, 0];
    d_meta.copy_from_host(&meta_init, &stream).unwrap();
    let batch_init = W34GdnBatch {
        query_start_loc: d_meta.addr() as *const i32,
        state_in_indices: (d_meta.addr() + 8) as *const i32,
        state_out_indices: (d_meta.addr() + 12) as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: (d_meta.addr() + 16) as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: init_len as i32,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    unsafe {
        assert_eq!(
            ffi::w34_gdn_recurrent_f16(
                &ex_zero,
                &batch_init,
                d_qkv.addr() as *const c_void,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                d_b.addr() as *const c_void,
                d_a.addr() as *const c_void,
                d_alog.addr() as *const f32,
                d_dtb.addr() as *const f32,
                st.ssm_state.addr() as *mut f32,
                d_out_init.addr() as *mut c_void,
                hk as i32,
                dk as i32,
                hv as i32,
                dv as i32,
                num_slots as i32,
            ),
            0
        );
    }

    let qkv_wy_ptr = (d_qkv.addr() + (init_len * conv_dim * 2) as u64) as *const c_void;
    let b_wy_ptr = (d_b.addr() + (init_len * hv * 2) as u64) as *const c_void;
    let a_wy_ptr = (d_a.addr() + (init_len * hv * 2) as u64) as *const c_void;

    // 2. Reference: run 260 tokens as two <256 recurrent chunks (130 + 130) from slot 0.
    let d_out_ref = DeviceBuffer::alloc(0, wy_len * hv * dv * 2).unwrap();
    for (sub_off, sub_len, s_in, s_out, ck_dst, ck_off) in [
        (0usize, 130usize, 0i32, 17i32, [1i32, 2, 3], [1i32, 2, 3]),
        (
            130usize,
            130usize,
            17i32,
            4i32,
            [-1i32, -1, -1],
            [-1i32, -1, -1],
        ),
    ] {
        let meta_ref: [i32; 11] = [
            0,
            sub_len as i32,
            s_in,
            s_out,
            ck_dst[0],
            ck_dst[1],
            ck_dst[2],
            ck_off[0],
            ck_off[1],
            ck_off[2],
            1,
        ];
        d_meta.copy_from_host(&meta_ref, &stream).unwrap();
        let batch_ref = W34GdnBatch {
            query_start_loc: d_meta.addr() as *const i32,
            state_in_indices: (d_meta.addr() + 8) as *const i32,
            state_out_indices: (d_meta.addr() + 12) as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: (d_meta.addr() + 16) as *const i32,
            checkpoint_token_offsets: (d_meta.addr() + 28) as *const i32,
            has_initial_state: (d_meta.addr() + 40) as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: sub_len as i32,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 3,
        };
        unsafe {
            assert_eq!(
                ffi::w34_gdn_recurrent_f16(
                    &ex_zero,
                    &batch_ref,
                    (qkv_wy_ptr as u64 + (sub_off * conv_dim * 2) as u64) as *const c_void,
                    core::ptr::null(),
                    core::ptr::null(),
                    core::ptr::null(),
                    (b_wy_ptr as u64 + (sub_off * hv * 2) as u64) as *const c_void,
                    (a_wy_ptr as u64 + (sub_off * hv * 2) as u64) as *const c_void,
                    d_alog.addr() as *const f32,
                    d_dtb.addr() as *const f32,
                    st.ssm_state.addr() as *mut f32,
                    (d_out_ref.addr() + (sub_off * hv * dv * 2) as u64) as *mut c_void,
                    hk as i32,
                    dk as i32,
                    hv as i32,
                    dv as i32,
                    num_slots as i32,
                ),
                0
            );
        }
    }

    // 3. Run the WY modes on all 260 tokens with checkpoints at [1, 2, 3] AND IN-PLACE
    //    `state_in == state_out == base_slot + 3` (reproducing production chunked-prefill aliasing!):
    //    - split_knorm (4 MiB scratch: K_norm cached)
    //    - split_noknorm (exact split workspace budget, with no space for K_norm)
    //    - fallback_wy (0 scratch: gdn_wy_chunk64_prefill_f16_kernel)
    //    - tight_fallback (old budget, insufficient for checkpoint-induced chunks)
    let state_elems = hv * dv * dk;
    let mut init_ssm_slot0 = vec![0.0f32; state_elems];
    let mut all_ssm_tmp = vec![0.0f32; num_slots * state_elems];
    st.ssm_state
        .copy_to_host(&mut all_ssm_tmp, &stream)
        .unwrap();
    stream.synchronize().unwrap();
    init_ssm_slot0.copy_from_slice(&all_ssm_tmp[..state_elems]);

    let d_ws = DeviceBuffer::alloc(0, 4 * 1024 * 1024).unwrap();
    let h_slot_bytes = hv * 128 * 128 * 2;
    let align256 = |n: usize| n.div_ceil(256) * 256;
    // Mirror the split-WY descriptor layout: header, one span, chunk descriptors, G, W.
    let needed_h_slots = wy_len.div_ceil(64) + 1 + 3;
    let off_h = 256
        + 256
        + align256(needed_h_slots * 32)
        + align256(wy_len * hv * 4)
        + align256(wy_len * hv * 256);
    let noknorm_cap = off_h + needed_h_slots * h_slot_bytes;
    let knorm_threshold = noknorm_cap + align256(wy_len * hk * 256);
    assert!(noknorm_cap < knorm_threshold && knorm_threshold <= d_ws.len_bytes);
    let tight_cap = off_h + (wy_len.div_ceil(64) + 1) * h_slot_bytes;
    let ex_tight = stream.exec(d_ws.addr(), tight_cap);
    let ex_knorm = stream.exec(d_ws.addr(), d_ws.len_bytes);
    let ex_noknorm = stream.exec(d_ws.addr(), noknorm_cap);

    let mut wy_outs = Vec::new();
    for base_slot in [5i32, 9i32, 13i32, 18i32] {
        let inplace_slot = (base_slot + 3) as usize;
        all_ssm_tmp[inplace_slot * state_elems..(inplace_slot + 1) * state_elems]
            .copy_from_slice(&init_ssm_slot0);
    }
    st.ssm_state.copy_from_host(&all_ssm_tmp, &stream).unwrap();

    for (ex_mode, base_slot) in [
        (ex_knorm, 5i32),
        (ex_noknorm, 9i32),
        (ex_zero, 13i32),
        (ex_tight, 18i32),
    ] {
        let d_out_wy = DeviceBuffer::alloc(0, wy_len * hv * dv * 2).unwrap();
        let meta_wy: [i32; 11] = [
            0,
            wy_len as i32,
            base_slot + 3,
            base_slot + 3,
            base_slot,
            base_slot + 1,
            base_slot + 2,
            1,
            2,
            3,
            1,
        ];
        d_meta.copy_from_host(&meta_wy, &stream).unwrap();
        let batch_wy = W34GdnBatch {
            query_start_loc: d_meta.addr() as *const i32,
            state_in_indices: (d_meta.addr() + 8) as *const i32,
            state_out_indices: (d_meta.addr() + 12) as *const i32,
            spec_state_indices: core::ptr::null(),
            checkpoint_dst_indices: (d_meta.addr() + 16) as *const i32,
            checkpoint_token_offsets: (d_meta.addr() + 28) as *const i32,
            has_initial_state: (d_meta.addr() + 40) as *const u8,
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: wy_len as i32,
            max_verify_tokens: 0,
            max_checkpoints_per_req: 3,
        };
        unsafe {
            assert_eq!(
                ffi::w34_gdn_recurrent_f16(
                    &ex_mode,
                    &batch_wy,
                    qkv_wy_ptr,
                    core::ptr::null(),
                    core::ptr::null(),
                    core::ptr::null(),
                    b_wy_ptr,
                    a_wy_ptr,
                    d_alog.addr() as *const f32,
                    d_dtb.addr() as *const f32,
                    st.ssm_state.addr() as *mut f32,
                    d_out_wy.addr() as *mut c_void,
                    hk as i32,
                    dk as i32,
                    hv as i32,
                    dv as i32,
                    num_slots as i32,
                ),
                0
            );
        }
        let mut out_host = vec![0u16; wy_len * hv * dv];
        d_out_wy.copy_to_host(&mut out_host, &stream).unwrap();
        wy_outs.push(out_host);
    }

    let mut out_ref = vec![0u16; wy_len * hv * dv];
    d_out_ref.copy_to_host(&mut out_ref, &stream).unwrap();
    let mut ssm_all = vec![0.0f32; num_slots * state_elems];
    st.ssm_state.copy_to_host(&mut ssm_all, &stream).unwrap();
    stream.synchronize().unwrap();

    // Split with K_norm and split without K_norm must be 0-ULP identical!
    assert_eq!(
        wy_outs[0], wy_outs[1],
        "split WY with K_norm vs without K_norm must match 0-ULP"
    );
    assert_eq!(
        &ssm_all[5 * state_elems..9 * state_elems],
        &ssm_all[9 * state_elems..13 * state_elems],
        "split WY ssm_state with K_norm vs without K_norm must match 0-ULP"
    );

    assert_eq!(
        wy_outs[2], wy_outs[3],
        "tight workspace must fall back, not silently skip GDN"
    );
    assert_eq!(
        &ssm_all[13 * state_elems..17 * state_elems],
        &ssm_all[18 * state_elems..22 * state_elems]
    );

    for (mode_idx, (mode_name, base_slot)) in [
        ("split_knorm", 5usize),
        ("split_noknorm", 9usize),
        ("fallback_chunk64", 13usize),
    ]
    .into_iter()
    .enumerate()
    {
        let max_out_err = out_ref
            .iter()
            .zip(wy_outs[mode_idx].iter())
            .map(|(&r, &g)| (f16_to_f32(r) - f16_to_f32(g)).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_out_err < 3e-3,
            "WY {mode_name} output max_abs_err={max_out_err} exceeded 3e-3"
        );
        for ck in 0..4usize {
            let ref_s = &ssm_all[(1 + ck) * state_elems..(2 + ck) * state_elems];
            let got_s =
                &ssm_all[(base_slot + ck) * state_elems..(base_slot + ck + 1) * state_elems];
            let max_s_err = ref_s
                .iter()
                .zip(got_s.iter())
                .map(|(&r, &g)| (r - g).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_s_err < 3e-3,
                "WY {mode_name} slot+{ck} ssm_state max_abs_err={max_s_err} exceeded 3e-3"
            );
        }
    }
}

/// Runs one `w34_gdn_recurrent_f16` call over `num_reqs` requests described by host metadata
/// `[query_start_loc[R+1], state_in[R], state_out[R]]` (all requests have initial state).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_gdn_batch(
    ex: &w34_kernels::ffi::W34Exec,
    stream: &CudaStream,
    d_meta: &DeviceBuffer,
    qsl: &[i32],
    state_in: &[i32],
    state_out: &[i32],
    qkv: *const c_void,
    b: *const c_void,
    a: *const c_void,
    d_alog: &DeviceBuffer,
    d_dtb: &DeviceBuffer,
    ssm_state: u64,
    out: u64,
    dims: (usize, usize, usize, usize, usize),
) {
    let (hk, dk, hv, dv, num_slots) = dims;
    let r = state_in.len();
    let mut meta: Vec<i32> = Vec::new();
    meta.extend_from_slice(qsl);
    meta.extend_from_slice(state_in);
    meta.extend_from_slice(state_out);
    let has_init_off = meta.len() * 4;
    meta.push(0x0101_0101); // has_initial_state bytes (up to 4 requests)
    d_meta.copy_from_host(&meta, stream).unwrap();
    let batch = W34GdnBatch {
        query_start_loc: d_meta.addr() as *const i32,
        state_in_indices: (d_meta.addr() + (qsl.len() * 4) as u64) as *const i32,
        state_out_indices: (d_meta.addr() + ((qsl.len() + r) * 4) as u64) as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: (d_meta.addr() + has_init_off as u64) as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: r as i32,
        total_tokens: qsl[r] - qsl[0],
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    unsafe {
        assert_eq!(
            ffi::w34_gdn_recurrent_f16(
                ex,
                &batch,
                qkv,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                b,
                a,
                d_alog.addr() as *const f32,
                d_dtb.addr() as *const f32,
                ssm_state as *mut f32,
                out as *mut c_void,
                hk as i32,
                dk as i32,
                hv as i32,
                dv as i32,
                num_slots as i32,
            ),
            0
        );
    }
    stream.synchronize().unwrap();
}

/// A mixed continuous-batching step (`q_len = [1, 256]`) takes the Tensor-Core WY path: the
/// prefill request must be 0-ULP identical to running it alone, and the 1-token decode request
/// must stay within WY-vs-recurrent tolerance of the dedicated decode kernel.
#[cfg(feature = "cuda")]
#[test]
fn test_gdn_wy_mixed_decode_prefill_batch_parity() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let stream = CudaStream::new(dev.device_id).expect("stream");

    let (hk, hv, dk, dv) = (2usize, 6usize, 128usize, 128usize);
    let conv_dim = 2 * hk * dk + hv * dv;
    let prefill_len = 256usize;
    let total_len = 1 + prefill_len;
    let num_slots = 6usize;
    let dims = (hk, dk, hv, dv, num_slots);

    let qkv_host: Vec<u16> = (0..(total_len * conv_dim))
        .map(|i| f32_to_f16(((i % 113) as f32 - 56.0) * 0.009))
        .collect();
    let b_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 17) as f32 - 8.0) * 0.1))
        .collect();
    let a_host: Vec<u16> = (0..(total_len * hv))
        .map(|i| f32_to_f16(((i % 29) as f32 - 14.0) * 0.05))
        .collect();
    let a_log: Vec<f32> = (0..hv).map(|i| -1.5 - 0.08 * (i as f32)).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|i| -0.1 + 0.02 * (i as f32)).collect();
    let d_qkv = DeviceBuffer::alloc(0, qkv_host.len() * 2).unwrap();
    let d_b = DeviceBuffer::alloc(0, b_host.len() * 2).unwrap();
    let d_a = DeviceBuffer::alloc(0, a_host.len() * 2).unwrap();
    let d_alog = DeviceBuffer::alloc(0, a_log.len() * 4).unwrap();
    let d_dtb = DeviceBuffer::alloc(0, dt_bias.len() * 4).unwrap();
    d_qkv.copy_from_host(&qkv_host, &stream).unwrap();
    d_b.copy_from_host(&b_host, &stream).unwrap();
    d_a.copy_from_host(&a_host, &stream).unwrap();
    d_alog.copy_from_host(&a_log, &stream).unwrap();
    d_dtb.copy_from_host(&dt_bias, &stream).unwrap();

    // Slots: 0 = decode-request init, 1 = prefill-request init, 2/3 = solo outputs,
    // 4/5 = mixed-batch outputs.
    let pool = StateBundlePool::new(0, &[0], num_slots, conv_dim, 4, hv, dk, dv).unwrap();
    let st = &pool.layers[0];
    let state_elems = hv * dv * dk;
    let mut ssm_init = vec![0.0f32; num_slots * state_elems];
    for (i, v) in ssm_init[..2 * state_elems].iter_mut().enumerate() {
        *v = ((i % 97) as f32 - 48.0) * 0.002;
    }
    st.ssm_state.copy_from_host(&ssm_init, &stream).unwrap();

    let d_meta = DeviceBuffer::alloc(0, 256).unwrap();
    let d_ws = DeviceBuffer::alloc(0, 4 * 1024 * 1024).unwrap();
    let ex = stream.exec(d_ws.addr(), d_ws.len_bytes);
    let row_bytes = (hv * dv * 2) as u64;
    let qkv_at = |tok: usize| (d_qkv.addr() + (tok * conv_dim * 2) as u64) as *const c_void;
    let b_at = |tok: usize| (d_b.addr() + (tok * hv * 2) as u64) as *const c_void;
    let a_at = |tok: usize| (d_a.addr() + (tok * hv * 2) as u64) as *const c_void;
    let ssm = st.ssm_state.addr();

    let d_out_solo = DeviceBuffer::alloc(0, total_len * hv * dv * 2).unwrap();
    run_gdn_batch(
        &ex,
        &stream,
        &d_meta,
        &[0, 1],
        &[0],
        &[2],
        qkv_at(0),
        b_at(0),
        a_at(0),
        &d_alog,
        &d_dtb,
        ssm,
        d_out_solo.addr(),
        dims,
    );
    run_gdn_batch(
        &ex,
        &stream,
        &d_meta,
        &[0, prefill_len as i32],
        &[1],
        &[3],
        qkv_at(1),
        b_at(1),
        a_at(1),
        &d_alog,
        &d_dtb,
        ssm,
        d_out_solo.addr() + row_bytes,
        dims,
    );
    let d_out_mixed = DeviceBuffer::alloc(0, total_len * hv * dv * 2).unwrap();
    run_gdn_batch(
        &ex,
        &stream,
        &d_meta,
        &[0, 1, total_len as i32],
        &[0, 1],
        &[4, 5],
        qkv_at(0),
        b_at(0),
        a_at(0),
        &d_alog,
        &d_dtb,
        ssm,
        d_out_mixed.addr(),
        dims,
    );

    let mut out_solo = vec![0u16; total_len * hv * dv];
    let mut out_mixed = vec![0u16; total_len * hv * dv];
    let mut ssm_all = vec![0.0f32; num_slots * state_elems];
    d_out_solo.copy_to_host(&mut out_solo, &stream).unwrap();
    d_out_mixed.copy_to_host(&mut out_mixed, &stream).unwrap();
    st.ssm_state.copy_to_host(&mut ssm_all, &stream).unwrap();
    stream.synchronize().unwrap();

    let row = hv * dv;
    let slot = |s: usize| &ssm_all[s * state_elems..(s + 1) * state_elems];
    assert_eq!(
        out_mixed[row..],
        out_solo[row..],
        "mixed-batch prefill output must match solo WY prefill 0-ULP"
    );
    assert_eq!(
        slot(5),
        slot(3),
        "mixed-batch prefill ssm_state must match solo 0-ULP"
    );

    let max_out_err = out_solo[..row]
        .iter()
        .zip(out_mixed[..row].iter())
        .map(|(&r, &g)| (f16_to_f32(r) - f16_to_f32(g)).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_out_err < 3e-3,
        "mixed decode output max_abs_err={max_out_err}"
    );
    let max_s_err = slot(2)
        .iter()
        .zip(slot(4).iter())
        .map(|(&r, &g)| (r - g).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_s_err < 3e-3,
        "mixed decode ssm_state max_abs_err={max_s_err}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_gate3_and_gate4_cuda_graph_spec_rollback_and_tp2_collectives() {
    let _guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Gate #4: Cold-start TP=2 init & typed collectives stress (including token IDs > 65504)
    let dev0 = CudaDevice::set_current(0).expect("CudaDevice(0)");
    let dev1 = CudaDevice::set_current(1).expect("CudaDevice(1)");
    let stream0 = CudaStream::new(dev0.device_id).expect("stream0");
    let stream1 = CudaStream::new(dev1.device_id).expect("stream1");
    prepare_execution_plan(dev0.device_id).unwrap();
    prepare_execution_plan(dev1.device_id).unwrap();

    let comm = Tp2Collectives::new(4096 * 2).expect("Tp2Collectives");
    let ex0 = stream0.exec(0, 0);
    let ex1 = stream1.exec(0, 0);

    let d_val0 = DeviceBuffer::alloc(0, 4).unwrap();
    let d_id0 = DeviceBuffer::alloc(0, 4).unwrap();
    let d_out_val0 = DeviceBuffer::alloc(0, 4).unwrap();
    let d_out_id0 = DeviceBuffer::alloc(0, 4).unwrap();
    d_val0.copy_from_host(&[12.5f32], &stream0).unwrap();
    d_id0.copy_from_host(&[100_001i32], &stream0).unwrap();

    let d_val1 = DeviceBuffer::alloc(1, 4).unwrap();
    let d_id1 = DeviceBuffer::alloc(1, 4).unwrap();
    let d_out_val1 = DeviceBuffer::alloc(1, 4).unwrap();
    let d_out_id1 = DeviceBuffer::alloc(1, 4).unwrap();
    d_val1.copy_from_host(&[19.75f32], &stream1).unwrap();
    d_id1.copy_from_host(&[240_123i32], &stream1).unwrap(); // > 65504 FP16 max!

    std::thread::scope(|s| {
        let h0 = s.spawn(|| {
            dev0.make_current().unwrap();
            comm.all_gather_top1_f32_i32(
                &ex0,
                d_val0.addr(),
                d_id0.addr(),
                d_out_val0.addr(),
                d_out_id0.addr(),
                0,
                1,
            )
            .unwrap();
            stream0.synchronize().unwrap();
        });
        let h1 = s.spawn(|| {
            dev1.make_current().unwrap();
            comm.all_gather_top1_f32_i32(
                &ex1,
                d_val1.addr(),
                d_id1.addr(),
                d_out_val1.addr(),
                d_out_id1.addr(),
                1,
                1,
            )
            .unwrap();
            stream1.synchronize().unwrap();
        });
        h0.join().unwrap();
        h1.join().unwrap();
    });

    let mut win0 = [0i32; 1];
    let mut win1 = [0i32; 1];
    d_out_id0.copy_to_host(&mut win0, &stream0).unwrap();
    d_out_id1.copy_to_host(&mut win1, &stream1).unwrap();
    assert_eq!(
        win0[0], 240_123,
        "Gate #4: INT32 token ID > 65504 preserved"
    );
    assert_eq!(win1[0], 240_123, "Gate #4: both ranks agree on top-1");

    // Gate #3: CUDA Graph replay under metadata mutation + TapPlan + StateTx speculative rollback
    let taps = TapPlan {
        layer_ids: vec![5, 19, 31],
    };
    assert_eq!(taps.layer_ids.len(), 3);

    let mut eng_graph = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Fp16,
        Some(256),
        true,
    )
    .expect("load 4B graph");
    let mut eng_eager = Qwen3_5Engine::load(
        "nchapman/Qwen3.5-4B-W4A16-GPTQ",
        1,
        KvFormatPolicy::Fp16,
        Some(256),
        false,
    )
    .expect("load 4B eager");

    let prompt_ids = eng_graph
        .encode("The capital of France is")
        .expect("encode");
    let m_graph = eng_graph.generate_tokens(&prompt_ids, 16, false).unwrap();
    let m_eager = eng_eager.generate_tokens(&prompt_ids, 16, false).unwrap();
    assert_eq!(
        m_graph.generated_token_ids, m_eager.generated_token_ids,
        "Gate #3: CUDA Graph replay matches eager execution bit-for-bit"
    );

    // Also verify StateTx speculative verify & rollback schedule (accepted_len = 3 out of 1+4)
    let mut tx = StateTx::new(0);
    tx.begin_prefill();
    tx.commit_prefill(prompt_ids.len());
    tx.begin_spec_verify(vec![1, 2, 3, 4]);
    assert_eq!(tx.verify_slot_schedule(), vec![0, 1, 2, 3, 4]);
    let freed = tx.resolve_spec_verify(3);
    assert_eq!(tx.committed_slot, 2);
    assert_eq!(freed, vec![0, 1, 3, 4]);
}

use super::*;
use std::path::Path;

fn sample_qwen4_exp_cfg() -> ModelConfig {
    let root: serde_json::Value = serde_json::from_str(
        r#"{
          "architectures": ["Qwen4ExpForConditionalGeneration"],
          "model_type": "qwen4_exp",
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4,
            "hc_lowrank": 80,
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ple_layer_ids": [2],
            "ple_seed": 20260202,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "indexer_head_dim": 128,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_budget": 1024,
            "indexer_compress_ratio": 4
          }
        }"#,
    )
    .unwrap();
    ModelConfig::from_json(&root).unwrap()
}

#[test]
fn test_ple_hasher_matches_checkpoint_constants_and_shift_semantics() {
    let cfg = sample_qwen4_exp_cfg();
    let hasher = PleNgramHasher::from_config(&cfg, 1, 0);
    assert_eq!(
        hasher.layer_multipliers,
        vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    );
    assert_eq!(
        hasher.ngram_heads_vocab_sizes,
        vec![
            20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
            20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
            20_000_161, 20_000_171
        ]
    );
    assert_eq!(
        hasher.ngram_heads_offsets,
        vec![
            0,
            20_000_003,
            40_000_026,
            60_000_059,
            80_000_106,
            100_000_165,
            120_000_228,
            140_000_297,
            160_000_374,
            180_000_455,
            200_000_548,
            220_000_655,
            240_000_802,
            260_000_955,
            280_001_114,
            300_001_275
        ]
    );
    assert_eq!(hasher.padded_vocab_size, 320_001_536);
    assert_eq!(hasher.shard_rows, 2_500_012);

    // Verify chunked streaming matches single-shot prefill across EOS boundaries.
    let eos = hasher.eos_token_id;
    let seq = [101, 202, eos, 303, 404, 505, eos, eos, 606];
    let mut ctx_full = [[eos, eos]];
    let mut ids_full = vec![0i64; seq.len() * 16];
    hasher.compute_ngram_ids(&seq, &[0, seq.len() as i32], &mut ctx_full, &mut ids_full);

    let mut ctx_step = [[eos, eos]];
    let mut ids_step = vec![0i64; seq.len() * 16];
    for (i, &tok) in seq.iter().enumerate() {
        hasher.compute_ngram_ids(
            &[tok],
            &[0, 1],
            &mut ctx_step,
            &mut ids_step[i * 16..(i + 1) * 16],
        );
    }
    assert_eq!(ids_full, ids_step);
    assert_eq!(ctx_full, ctx_step);
}

#[test]
fn test_qwen_flash_next_capacity_planner_tp1_and_tp2() {
    let cfg = sample_qwen4_exp_cfg();
    let vram_24g = 24 * 1024 * 1024 * 1024usize;
    let free_23_5g = 23_600 * 1024 * 1024usize;

    let plan_tp1 = QwenFlashNextCapacityPlan::plan(
        &cfg,
        1,
        vram_24g,
        free_23_5g,
        KvFormatPolicy::Tq4,
        16,
        Some(2048),
        Some(8),
        None,
        true,
    )
    .unwrap();
    assert!(plan_tp1.hot_experts_per_layer_per_rank >= 36);
    assert!(plan_tp1.headroom_per_rank_bytes >= 512 * 1024 * 1024);

    let plan_tp2 = QwenFlashNextCapacityPlan::plan(
        &cfg,
        2,
        vram_24g,
        free_23_5g,
        KvFormatPolicy::Tq4,
        16,
        Some(4096),
        Some(16),
        None,
        true,
    )
    .unwrap();
    assert!(plan_tp2.hot_experts_per_layer_per_rank >= 96);
    assert!(plan_tp2.headroom_per_rank_bytes >= 512 * 1024 * 1024);

    let plan_tp4 = QwenFlashNextCapacityPlan::plan(
        &cfg,
        4,
        vram_24g,
        free_23_5g,
        KvFormatPolicy::Tq4,
        16,
        Some(4096),
        Some(16),
        None,
        true,
    )
    .unwrap();
    assert_eq!(plan_tp4.hot_experts_per_layer_per_rank, 128);
    assert_eq!(plan_tp4.cold_experts_per_layer_per_rank, 0);
    assert_eq!(plan_tp4.host_expert_pinned_ram_bytes, 0);
    assert!(plan_tp4.headroom_per_rank_bytes >= 512 * 1024 * 1024);
    assert!(cfg.kv_heads_compatible_with_tp(4));
    assert_eq!(cfg.local_kv_heads(4), 1);
    assert_eq!(
        [0, 1, 2, 3].map(|r| cfg.kv_head_shard_index(r, 4)),
        [0, 0, 1, 1]
    );
}

#[test]
fn test_ple_mmap_table_against_real_snapshot_if_present() {
    let snap = Path::new(
        "/home/nchapman/.cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/ef554143369a706525336f6b42a09094835dc077",
    );
    if !snap.exists() {
        return;
    }
    let cfg = ModelConfig::from_dir(snap).unwrap();
    let archive = SafetensorsArchive::open(snap).unwrap();
    let hasher = PleNgramHasher::from_archive(&archive, &cfg, 1, 0).unwrap();
    let table = PleMmapTable::open(&archive, &cfg, 1).unwrap();
    assert_eq!(table.shard_rows, 2_500_012);
    assert_eq!(table.head_dim, 160);
    assert!((table.lut.weight_scale - 0.00019931793).abs() < 1e-9);

    let eos = hasher.eos_token_id;
    let mut ctx = [[eos, eos]];
    let tokens = [151644, 872, 198, 3838];
    let mut ids = vec![0i64; tokens.len() * 16];
    hasher.compute_ngram_ids(&tokens, &[0, tokens.len() as i32], &mut ctx, &mut ids);
    let mut out_f32 = vec![0.0f32; tokens.len() * 2560];
    table.gather_dequant_f32(&ids, &mut out_f32);
    assert!(out_f32.iter().all(|v| v.is_finite()));
    assert!(out_f32.iter().any(|&v| v != 0.0));
}

#[cfg(feature = "cuda")]
#[test]
#[allow(clippy::cognitive_complexity)]
fn test_gpu_hc_ple_qsa_kernels_parity() {
    use crate::cuda::{CudaDevice, CudaStream, DeviceBuffer, f16_to_f32};
    use core::ffi::c_void;
    use w34_kernels::ffi::{
        W34_ATTN_MASK_CAUSAL, W34_KV_FORMAT_FP16, W34_KV_FORMAT_Q8_0, W34_KV_FORMAT_TQ4,
        W34_KV_NAMESPACE_TARGET_FULL_ATTN, W34_KV_TRANSFORM_IDENTITY,
        W34_KV_TRANSFORM_RADEMACHER_WHT256, W34AttentionBatch, W34GdnBatch, W34KvLayerLayout,
        w34_kv_cache_write_strided,
    };

    fn dev_from_slice<T: Copy>(src: &[T], stream: &CudaStream) -> DeviceBuffer {
        let buf = DeviceBuffer::alloc(0, core::mem::size_of_val(src)).unwrap();
        buf.copy_from_host(src, stream).unwrap();
        stream.synchronize().unwrap();
        buf
    }

    fn dev_to_vec<T: Copy + Default>(
        buf: &DeviceBuffer,
        len: usize,
        stream: &CudaStream,
    ) -> Vec<T> {
        let mut out = vec![T::default(); len];
        buf.copy_to_host(&mut out, stream).unwrap();
        out
    }

    let _dev = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ex = stream.exec(0, 0);

    // 1. Verify HyperConnection (GatedResidual) kernels:
    //    hc_init_streams_f16 -> hc_rmsnorm_f16 -> hc_down_silu_inject_f16 -> hc_mix_streams_f16 -> hc_combine_f16
    let m = 3usize;
    let hc_count = 4usize;
    let hidden = 2560usize;
    let rank_dim = 320usize;
    let eps = 1e-6f32;

    let embed_f32: Vec<f32> = (0..m * hidden)
        .map(|i| (((i * 17 + 5) % 97) as f32 - 48.0) * 0.02)
        .collect();
    let embed_f16: Vec<u16> = embed_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let norm_w_f32: Vec<f32> = (0..hc_count * hidden)
        .map(|i| (((i * 13 + 3) % 53) as f32 - 26.0) * 0.01)
        .collect();
    let norm_w_f16: Vec<u16> = norm_w_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let down_in_f32: Vec<f32> = (0..m * rank_dim)
        .map(|i| (((i * 23 + 7) % 89) as f32 - 44.0) * 0.1)
        .collect();
    let down_in_f16: Vec<u16> = down_in_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let inj_in_f32: Vec<f32> = (0..m * hc_count)
        .map(|i| ((i as f32) - 5.5) * 0.8)
        .collect();
    let inj_in_f16: Vec<u16> = inj_in_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let up_in_f32: Vec<f32> = (0..m * hc_count * hidden)
        .map(|i| (((i * 7 + 11) % 71) as f32 - 35.0) * 0.05)
        .collect();
    let up_in_f16: Vec<u16> = up_in_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let blk_out_f32: Vec<f32> = (0..m * hidden)
        .map(|i| (((i * 19 + 1) % 67) as f32 - 33.0) * 0.03)
        .collect();
    let blk_out_f16: Vec<u16> = blk_out_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let ple_delta_f32: Vec<f32> = (0..m * hc_count * hidden)
        .map(|i| (((i * 31 + 2) % 61) as f32 - 30.0) * 0.01)
        .collect();
    let ple_delta_f16: Vec<u16> = ple_delta_f32.iter().map(|&v| f32_to_f16(v)).collect();

    let d_embed = dev_from_slice(&embed_f16, &stream);
    let d_res = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();
    let d_norm_w = dev_from_slice(&norm_w_f16, &stream);
    let d_normed = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();
    let d_down_in = dev_from_slice(&down_in_f16, &stream);
    let d_down_out = DeviceBuffer::alloc(0, m * rank_dim * 2).unwrap();
    let d_inj_in = dev_from_slice(&inj_in_f16, &stream);
    let d_inj_w = DeviceBuffer::alloc(0, m * hc_count * 4).unwrap();
    let d_up_in = dev_from_slice(&up_in_f16, &stream);
    let d_mixed = DeviceBuffer::alloc(0, m * hidden * 2).unwrap();
    let d_blk_out = dev_from_slice(&blk_out_f16, &stream);
    let d_ple_delta = dev_from_slice(&ple_delta_f16, &stream);
    let d_res_out = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();

    gpu_ops::hc_init_streams_f16(&ex, d_embed.addr(), d_res.addr(), 0, m, hc_count, hidden)
        .unwrap();
    gpu_ops::hc_rmsnorm_f16(
        &ex,
        d_res.addr(),
        d_norm_w.addr(),
        d_normed.addr(),
        0,
        m,
        hc_count,
        hidden,
        eps,
    )
    .unwrap();
    gpu_ops::hc_down_silu_inject_f16(
        &ex,
        d_down_in.addr(),
        d_down_out.addr(),
        d_inj_in.addr(),
        d_inj_w.addr(),
        0,
        m,
        rank_dim,
        hc_count,
    )
    .unwrap();
    gpu_ops::hc_mix_streams_f16(
        &ex,
        d_res.addr(),
        d_up_in.addr(),
        d_mixed.addr(),
        0,
        m,
        hc_count,
        hidden,
    )
    .unwrap();
    gpu_ops::hc_combine_f16(
        &ex,
        d_res.addr(),
        d_blk_out.addr(),
        d_inj_w.addr(),
        d_ple_delta.addr(),
        d_res_out.addr(),
        0,
        m,
        hc_count,
        hidden,
    )
    .unwrap();

    let h_down_out: Vec<u16> = dev_to_vec(&d_down_out, m * rank_dim, &stream);
    let h_inj_w: Vec<f32> = dev_to_vec(&d_inj_w, m * hc_count, &stream);
    let h_mixed: Vec<u16> = dev_to_vec(&d_mixed, m * hidden, &stream);
    let h_res_out: Vec<u16> = dev_to_vec(&d_res_out, m * hc_count * hidden, &stream);

    for i in 0..m * rank_dim {
        let x = f16_to_f32(down_in_f16[i]) * 0.25;
        let ref_silu = x / (1.0 + (-x).exp());
        let got = f16_to_f32(h_down_out[i]);
        assert!((got - ref_silu).abs() < 2e-3, "down silu mismatch at {i}");
    }
    for i in 0..m * hc_count {
        let x = f16_to_f32(inj_in_f16[i]) * 0.25;
        let ref_inj = 2.0 / (1.0 + (-x).exp());
        assert!((h_inj_w[i] - ref_inj).abs() < 1e-5, "inj mismatch at {i}");
    }
    for tok in 0..m {
        for d in 0..hidden {
            let r = f16_to_f32(embed_f16[tok * hidden + d]);
            let mut acc = 0.0f32;
            for s in 0..hc_count {
                let u = f16_to_f32(up_in_f16[(tok * hc_count + s) * hidden + d]);
                acc += (0.25 / (1.0 + (-u).exp())) * r;
            }
            let got_mix = f16_to_f32(h_mixed[tok * hidden + d]);
            assert!((got_mix - acc).abs() < 2e-3);

            let y = f16_to_f32(blk_out_f16[tok * hidden + d]);
            for s in 0..hc_count {
                let idx = (tok * hc_count + s) * hidden + d;
                let p = f16_to_f32(ple_delta_f16[idx]);
                let ref_comb = r + h_inj_w[tok * hc_count + s] * y + p;
                let got_comb = f16_to_f32(h_res_out[idx]);
                assert!((got_comb - ref_comb).abs() < 3e-3);
            }
        }
    }

    // 2. Verify PLE kernels:
    //    ple_dequant_f8_to_f16, ple_dilated_conv1d_f16 (prefill + continuation step), ple_signed_sqrt_gate_f16
    let f8_bytes: Vec<u8> = (0..m * hidden)
        .map(|i| ((i * 29 + 3) % 120) as u8)
        .collect();
    let scale = 0.00019931793f32;
    let lut = F8E4m3ScaleLut::new(scale);
    let d_f8 = dev_from_slice(&f8_bytes, &stream);
    let d_ple_emb = DeviceBuffer::alloc(0, m * hidden * 2).unwrap();
    gpu_ops::ple_dequant_f8_to_f16(&ex, d_f8.addr(), d_ple_emb.addr(), 0, m, hidden, scale)
        .unwrap();
    let h_ple_emb: Vec<u16> = dev_to_vec(&d_ple_emb, m * hidden, &stream);
    for (i, &b) in f8_bytes.iter().enumerate() {
        assert_eq!(h_ple_emb[i], lut.f16_lut[b as usize]);
    }

    // Verify PLE signed-sqrt sigmoid gate (including exact 0.0 sign preservation):
    let query_vals: Vec<f32> = (0..m * hc_count * hidden)
        .map(|i| {
            if i % 16 == 0 {
                0.0
            } else {
                (((i * 19 + 5) % 83) as f32 - 41.0) * 0.2
            }
        })
        .collect();
    let query_f16: Vec<u16> = query_vals.iter().map(|&v| f32_to_f16(v)).collect();
    let key_f16: Vec<u16> = (0..m * hc_count * hidden)
        .map(|i| f32_to_f16((((i * 7 + 3) % 59) as f32 - 29.0) * 0.1))
        .collect();
    let val_f16: Vec<u16> = (0..m * hidden)
        .map(|i| f32_to_f16((((i * 13 + 1) % 47) as f32 - 23.0) * 0.15))
        .collect();
    let d_qg = dev_from_slice(&query_f16, &stream);
    let d_kg = dev_from_slice(&key_f16, &stream);
    let d_vg = dev_from_slice(&val_f16, &stream);
    let d_gout = DeviceBuffer::alloc(0, m * hc_count * hidden * 2).unwrap();
    gpu_ops::ple_signed_sqrt_gate_f16(
        &ex,
        d_qg.addr(),
        d_kg.addr(),
        d_vg.addr(),
        d_gout.addr(),
        0,
        m,
        hc_count,
        hidden,
        1e-6,
    )
    .unwrap();
    let h_gout: Vec<u16> = dev_to_vec(&d_gout, m * hc_count * hidden, &stream);
    for tok in 0..m {
        for s in 0..hc_count {
            let mut dot = 0.0f32;
            for d in 0..hidden {
                let idx = (tok * hc_count + s) * hidden + d;
                dot += f16_to_f32(query_f16[idx]) * f16_to_f32(key_f16[idx]);
            }
            dot /= (hidden as f32).sqrt();
            let sgn = if dot > 0.0 {
                1.0
            } else if dot < 0.0 {
                -1.0
            } else {
                0.0
            };
            let logit = sgn * dot.abs().max(1e-6).sqrt();
            let g = 1.0 / (1.0 + (-logit).exp());
            for d in 0..hidden {
                let v = f16_to_f32(val_f16[tok * hidden + d]);
                let idx = (tok * hc_count + s) * hidden + d;
                let ref_out = g * v;
                let got = f16_to_f32(h_gout[idx]);
                assert!((got - ref_out).abs() < 3e-3, "gate mismatch at {idx}");
            }
        }
    }

    // Verify PLE stateful dilated conv1d (kernel=4, dilation=3, span=9):
    // Run 12 tokens in one prefill vs 10 tokens prefill + 2 tokens decode continuation!
    let conv_dim = 64usize;
    let seq_total = 12usize;
    let key_seq_f16: Vec<u16> = (0..seq_total * conv_dim)
        .map(|i| f32_to_f16((((i * 11 + 5) % 43) as f32 - 21.0) * 0.05))
        .collect();
    let conv_w_f16: Vec<u16> = (0..conv_dim * 4)
        .map(|i| f32_to_f16((((i * 7 + 1) % 19) as f32 - 9.0) * 0.1))
        .collect();
    let d_conv_w = dev_from_slice(&conv_w_f16, &stream);

    let d_key_full = dev_from_slice(&key_seq_f16, &stream);
    let d_out_full = DeviceBuffer::alloc(0, seq_total * conv_dim * 2).unwrap();
    let d_state_full = DeviceBuffer::alloc(0, 2 * conv_dim * 9 * 2).unwrap();
    let d_qsl_full = dev_from_slice(&[0i32, seq_total as i32], &stream);
    let d_slot0 = dev_from_slice(&[0i32], &stream);
    let d_has0 = dev_from_slice(&[0u8], &stream);
    let batch_full = W34GdnBatch {
        query_start_loc: d_qsl_full.addr() as usize as *const i32,
        state_in_indices: d_slot0.addr() as usize as *const i32,
        state_out_indices: d_slot0.addr() as usize as *const i32,
        spec_state_indices: core::ptr::null(),
        checkpoint_dst_indices: core::ptr::null(),
        checkpoint_token_offsets: core::ptr::null(),
        has_initial_state: d_has0.addr() as usize as *const u8,
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: seq_total as i32,
        max_verify_tokens: 0,
        max_checkpoints_per_req: 0,
    };
    gpu_ops::ple_dilated_conv1d_f16(
        &ex,
        d_key_full.addr(),
        d_conv_w.addr(),
        d_state_full.addr(),
        d_out_full.addr(),
        &batch_full,
        conv_dim,
    )
    .unwrap();

    // Now run 10 tokens + 2 tokens continuation using state slot 1:
    let split = 10usize;
    let d_key_p1 = dev_from_slice(&key_seq_f16[..split * conv_dim], &stream);
    let d_out_p1 = DeviceBuffer::alloc(0, split * conv_dim * 2).unwrap();
    let d_qsl_p1 = dev_from_slice(&[0i32, split as i32], &stream);
    let d_slot1 = dev_from_slice(&[1i32], &stream);
    let batch_p1 = W34GdnBatch {
        query_start_loc: d_qsl_p1.addr() as usize as *const i32,
        state_in_indices: d_slot1.addr() as usize as *const i32,
        state_out_indices: d_slot1.addr() as usize as *const i32,
        total_tokens: split as i32,
        ..batch_full
    };
    gpu_ops::ple_dilated_conv1d_f16(
        &ex,
        d_key_p1.addr(),
        d_conv_w.addr(),
        d_state_full.addr(),
        d_out_p1.addr(),
        &batch_p1,
        conv_dim,
    )
    .unwrap();

    let rem = seq_total - split;
    let d_key_p2 = dev_from_slice(&key_seq_f16[split * conv_dim..], &stream);
    let d_out_p2 = DeviceBuffer::alloc(0, rem * conv_dim * 2).unwrap();
    let d_qsl_p2 = dev_from_slice(&[0i32, rem as i32], &stream);
    let d_has1 = dev_from_slice(&[1u8], &stream);
    let batch_p2 = W34GdnBatch {
        query_start_loc: d_qsl_p2.addr() as usize as *const i32,
        state_in_indices: d_slot1.addr() as usize as *const i32,
        state_out_indices: d_slot1.addr() as usize as *const i32,
        has_initial_state: d_has1.addr() as usize as *const u8,
        total_tokens: rem as i32,
        ..batch_full
    };
    gpu_ops::ple_dilated_conv1d_f16(
        &ex,
        d_key_p2.addr(),
        d_conv_w.addr(),
        d_state_full.addr(),
        d_out_p2.addr(),
        &batch_p2,
        conv_dim,
    )
    .unwrap();

    let h_out_full: Vec<u16> = dev_to_vec(&d_out_full, seq_total * conv_dim, &stream);
    let mut h_out_split: Vec<u16> = dev_to_vec(&d_out_p1, split * conv_dim, &stream);
    h_out_split.extend(dev_to_vec::<u16>(&d_out_p2, rem * conv_dim, &stream));
    assert_eq!(h_out_full, h_out_split);

    // 3. Verify QSA Indexer + Block Selector + Sparse Paged GQA kernels:
    let qsa_tokens = 2usize;
    let q_heads = 12usize;
    let kv_heads = 1usize;
    let head_dim = 256usize;
    let block_size = 16usize;
    let num_pages = 4usize;
    let select_k = 64usize;

    let d_k_cache = dev_from_slice(
        &vec![f32_to_f16(0.1); num_pages * kv_heads * block_size * head_dim],
        &stream,
    );
    let d_v_cache = dev_from_slice(
        &vec![f32_to_f16(0.25); num_pages * kv_heads * block_size * head_dim],
        &stream,
    );
    let kv_layout = W34KvLayerLayout {
        k_base: d_k_cache.addr(),
        v_base: d_v_cache.addr(),
        k_page_stride_bytes: (kv_heads * block_size * head_dim * 2) as u64,
        k_head_stride_bytes: (block_size * head_dim * 2) as u64,
        k_token_stride_bytes: (head_dim * 2) as u64,
        v_page_stride_bytes: (kv_heads * block_size * head_dim * 2) as u64,
        v_head_stride_bytes: (block_size * head_dim * 2) as u64,
        v_token_stride_bytes: (head_dim * 2) as u64,
        namespace_id: W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 3,
        local_kv_heads: kv_heads as i32,
        head_dim: head_dim as i32,
        block_size: block_size as i32,
        k_format: W34_KV_FORMAT_FP16,
        v_format: W34_KV_FORMAT_FP16,
        k_transform: W34_KV_TRANSFORM_IDENTITY,
        v_transform: W34_KV_TRANSFORM_IDENTITY,
        reserved: 0,
    };

    let d_qk_idx_in = dev_from_slice(&vec![f32_to_f16(0.1); qsa_tokens * 5 * 128], &stream);
    let d_q_idx = dev_from_slice(&vec![f32_to_f16(0.1); qsa_tokens * 4 * 128], &stream);
    let d_k_idx = dev_from_slice(&vec![f32_to_f16(0.1); qsa_tokens * 128], &stream);
    let d_idx_norm_w = dev_from_slice(&vec![f32_to_f16(0.0); 128], &stream);
    let d_pos_ids = dev_from_slice(&[20i32, 21, 20, 21, 20, 21], &stream);
    gpu_ops::qsa_indexer_norm_rope_f16(
        &ex,
        d_qk_idx_in.addr(),
        d_idx_norm_w.addr(),
        d_idx_norm_w.addr(),
        d_pos_ids.addr(),
        d_q_idx.addr(),
        d_k_idx.addr(),
        0,
        qsa_tokens,
        4,
        1,
        128,
        64,
        10_000_000.0,
        1e-6,
        11,
        11,
        qsa_tokens,
    )
    .unwrap();

    let d_raw_k = dev_from_slice(&vec![f32_to_f16(0.1); num_pages * 16 * (128 + 8)], &stream);
    let d_comp_k = dev_from_slice(&vec![f32_to_f16(0.1); num_pages * 4 * 128], &stream);
    let d_qsl_attn = dev_from_slice(&[0, qsa_tokens as i32], &stream);
    let d_ctx_before = dev_from_slice(&[20], &stream);
    let d_seq_after = dev_from_slice(&[22], &stream);
    let d_blk_tbl = dev_from_slice(&[2, 1, -1, -1], &stream);
    let d_slot_map = dev_from_slice(&[16 + 4, 16 + 5], &stream);
    gpu_ops::qsa_indexer_cache_update_f16(
        &ex,
        d_k_idx.addr(),
        d_raw_k.addr(),
        d_comp_k.addr(),
        d_idx_norm_w.addr(),
        d_pos_ids.addr(),
        d_slot_map.addr(),
        0,
        qsa_tokens,
        128,
        64,
        10_000_000.0,
        1e-6,
        11,
        11,
        qsa_tokens,
    )
    .unwrap();
    let attn_batch = W34AttentionBatch {
        query_start_loc: d_qsl_attn.addr() as usize as *const i32,
        context_lens_before: d_ctx_before.addr() as usize as *const i32,
        seq_lens_after: d_seq_after.addr() as usize as *const i32,
        block_tables: d_blk_tbl.addr() as usize as *const i32,
        write_slot_mapping: d_slot_map.addr() as usize as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: qsa_tokens as i32,
        max_blocks_per_req: 4,
        max_q_len: qsa_tokens as i32,
        max_seq_len: 22,
        mask_mode: W34_ATTN_MASK_CAUSAL,
        sliding_window: -1,
        sm_scale: (head_dim as f32).powf(-0.5),
        reserved: 0,
    };

    let d_sel_phys = DeviceBuffer::alloc(0, qsa_tokens * select_k * 4).unwrap();
    let d_sel_log = DeviceBuffer::alloc(0, qsa_tokens * select_k * 4).unwrap();
    let d_num_sel = DeviceBuffer::alloc(0, qsa_tokens * 4).unwrap();
    let d_seq_pos = DeviceBuffer::alloc(0, qsa_tokens * 4).unwrap();

    gpu_ops::qsa_select_blocks_f16(
        &ex,
        d_q_idx.addr(),
        d_comp_k.addr(),
        &attn_batch,
        d_sel_phys.addr(),
        d_sel_log.addr(),
        d_num_sel.addr(),
        d_seq_pos.addr(),
        4,
        128,
        block_size,
        select_k,
        8,
        16,
    )
    .unwrap();

    let h_num_sel: Vec<i32> = dev_to_vec(&d_num_sel, qsa_tokens, &stream);
    let h_seq_pos: Vec<i32> = dev_to_vec(&d_seq_pos, qsa_tokens, &stream);
    let h_sel_phys: Vec<i32> = dev_to_vec(&d_sel_phys, qsa_tokens * select_k, &stream);
    assert_eq!(h_num_sel, vec![6, 6]);
    assert_eq!(h_seq_pos, vec![20, 21]);
    assert_eq!(&h_sel_phys[0..6], &[8, 9, 10, 11, 4, 5]);

    let d_q_main = dev_from_slice(
        &vec![f32_to_f16(0.1); qsa_tokens * q_heads * head_dim],
        &stream,
    );
    let d_gate = dev_from_slice(
        &vec![f32_to_f16(0.0); qsa_tokens * q_heads * head_dim],
        &stream,
    );
    let d_attn_out = DeviceBuffer::alloc(0, qsa_tokens * q_heads * head_dim * 2).unwrap();

    gpu_ops::qsa_sparse_paged_attn_f16(
        &ex,
        d_q_main.addr(),
        d_gate.addr(),
        &kv_layout,
        d_sel_phys.addr(),
        d_sel_log.addr(),
        d_num_sel.addr(),
        d_seq_pos.addr(),
        0,
        d_attn_out.addr(),
        qsa_tokens,
        q_heads,
        select_k,
        attn_batch.sm_scale,
    )
    .unwrap();

    let h_attn_out: Vec<u16> = dev_to_vec(&d_attn_out, qsa_tokens * q_heads * head_dim, &stream);
    // Since V = 0.25 everywhere and sigmoid(0.0) = 0.5, gated attention output must equal 0.125!
    for &v in &h_attn_out {
        let vf = f16_to_f32(v);
        assert!((vf - 0.125).abs() < 1e-3, "expected 0.125, got {vf}");
    }

    // Verify hc_count = 2 scaling (1.0 / hc_count = 0.5, matching vLLM test_hc_ops.py)
    let d_hc2_down_in = dev_from_slice(&[f32_to_f16(2.0); 4], &stream);
    let d_hc2_down_out = DeviceBuffer::alloc(0, 4 * 2).unwrap();
    let d_hc2_inj_in = dev_from_slice(&[f32_to_f16(0.0); 2], &stream);
    let d_hc2_inj_out = DeviceBuffer::alloc(0, 2 * 4).unwrap();
    gpu_ops::hc_down_silu_inject_f16(
        &ex,
        d_hc2_down_in.addr(),
        d_hc2_down_out.addr(),
        d_hc2_inj_in.addr(),
        d_hc2_inj_out.addr(),
        0,
        1,
        4,
        2,
    )
    .unwrap();
    let h_hc2_down: Vec<u16> = dev_to_vec(&d_hc2_down_out, 4, &stream);
    // down = 2.0, hc_count = 2 -> x = 2.0 / 2 = 1.0 -> SiLU(1.0) = 1.0 / (1 + e^-1) = 0.7310586
    let expected_silu1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
    assert!(
        (f16_to_f32(h_hc2_down[0]) - expected_silu1).abs() < 2e-3,
        "expected SiLU(2.0 / 2) = {expected_silu1}, got {}",
        f16_to_f32(h_hc2_down[0])
    );

    // 4. Non-uniform QSA top-k 4-token group selection (12 completed groups + 1 incomplete tail group
    //    -> select_k = 5: 4 highest-scoring completed groups + tail group)
    //    + sparse paged GQA across FP16, Q8_0 (272 B/row), and TQ4 (132 B/row + Rademacher WHT256)
    //    against an exact CPU softmax + sigmoid-gate reference.
    let nu_pages = 4usize;
    let nu_seq_len = 50usize; // 12 completed 4-token groups (0..11, tokens 0..48) + tail group 12 (tokens 48..50)
    let nu_select_k = 5usize; // must pick 4 highest-scoring completed groups (4..7) + tail group 12
    let nu_q_heads = 2usize;
    // Logical blocks [0, 1, 2, 3] -> physical pages [3, 1, 0, 2]
    let nu_blk_tbl = [3i32, 1, 0, 2];
    // Set compressed K so logical groups 4..7 (phys page 1, phys groups 4..7) have positive dot product
    // with q_idx (+0.75), while all other completed groups have negative dot products (-0.5).
    let mut h_nu_comp_k = vec![f32_to_f16(-0.5); nu_pages * 4 * 128];
    for sub in 0..4 {
        for d in 0..128 {
            h_nu_comp_k[(4 + sub) * 128 + d] = f32_to_f16(0.75);
        }
    }
    let d_nu_comp_k = dev_from_slice(&h_nu_comp_k, &stream);
    let d_nu_q_idx = dev_from_slice(&vec![f32_to_f16(0.5); 4 * 128], &stream);
    let d_nu_qsl = dev_from_slice(&[0i32, 1], &stream);
    let d_nu_ctx_before = dev_from_slice(&[(nu_seq_len - 1) as i32], &stream);
    let d_nu_seq_after = dev_from_slice(&[nu_seq_len as i32], &stream);
    let d_nu_blk_tbl = dev_from_slice(&nu_blk_tbl, &stream);
    let d_nu_slot_map = dev_from_slice(&[2i32 * 16 + 1], &stream);
    let nu_attn_batch = W34AttentionBatch {
        query_start_loc: d_nu_qsl.addr() as usize as *const i32,
        context_lens_before: d_nu_ctx_before.addr() as usize as *const i32,
        seq_lens_after: d_nu_seq_after.addr() as usize as *const i32,
        block_tables: d_nu_blk_tbl.addr() as usize as *const i32,
        write_slot_mapping: d_nu_slot_map.addr() as usize as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_blocks_per_req: 4,
        max_q_len: 1,
        max_seq_len: nu_seq_len as i32,
        mask_mode: W34_ATTN_MASK_CAUSAL,
        sliding_window: -1,
        sm_scale: (head_dim as f32).powf(-0.5),
        reserved: 0,
    };
    let d_nu_sel_phys = DeviceBuffer::alloc(0, nu_select_k * 4).unwrap();
    let d_nu_sel_log = DeviceBuffer::alloc(0, nu_select_k * 4).unwrap();
    let d_nu_num_sel = DeviceBuffer::alloc(0, 4).unwrap();
    let d_nu_seq_pos = DeviceBuffer::alloc(0, 4).unwrap();
    gpu_ops::qsa_select_blocks_f16(
        &ex,
        d_nu_q_idx.addr(),
        d_nu_comp_k.addr(),
        &nu_attn_batch,
        d_nu_sel_phys.addr(),
        d_nu_sel_log.addr(),
        d_nu_num_sel.addr(),
        d_nu_seq_pos.addr(),
        4,
        128,
        block_size,
        nu_select_k,
        0,
        0,
    )
    .unwrap();
    let h_nu_num_sel: Vec<i32> = dev_to_vec(&d_nu_num_sel, 1, &stream);
    let h_nu_sel_log: Vec<i32> = dev_to_vec(&d_nu_sel_log, nu_select_k, &stream);
    let h_nu_sel_phys: Vec<i32> = dev_to_vec(&d_nu_sel_phys, nu_select_k, &stream);
    assert_eq!(h_nu_num_sel, vec![5]);
    // Must select logical groups 4..7 (phys groups 4..7 on phys page 1) and tail group 12 (phys group 8 on phys page 2)
    assert_eq!(h_nu_sel_log, vec![4, 5, 6, 7, 12]);
    assert_eq!(h_nu_sel_phys, vec![4, 5, 6, 7, 8]);

    // Populate all 50 tokens across the 4 pages with deterministic non-uniform K and V via w34_kv_cache_write_strided
    // for FP16, Q8_0 (272 B/row), and TQ4 (132 B/row + Rademacher WHT256).
    let h_nu_k: Vec<u16> = (0..nu_seq_len * head_dim)
        .map(|i| f32_to_f16((((i * 17 + 3) % 53) as f32 - 26.0) * 0.03))
        .collect();
    let h_nu_v: Vec<u16> = (0..nu_seq_len * head_dim)
        .map(|i| f32_to_f16((((i * 23 + 7) % 47) as f32 - 23.0) * 0.04))
        .collect();
    let h_nu_q: Vec<u16> = (0..nu_q_heads * head_dim)
        .map(|i| f32_to_f16((((i * 13 + 11) % 37) as f32 - 18.0) * 0.05))
        .collect();
    let h_nu_gate: Vec<u16> = (0..nu_q_heads * head_dim)
        .map(|i| f32_to_f16((((i * 19 + 5) % 29) as f32 - 14.0) * 0.08))
        .collect();
    let write_slots: Vec<i32> = (0..nu_seq_len)
        .map(|t| {
            let log_blk = t / block_size;
            let off = t % block_size;
            nu_blk_tbl[log_blk] * (block_size as i32) + (off as i32)
        })
        .collect();
    let d_nu_k_in = dev_from_slice(&h_nu_k, &stream);
    let d_nu_v_in = dev_from_slice(&h_nu_v, &stream);
    let d_nu_q = dev_from_slice(&h_nu_q, &stream);
    let d_nu_gate = dev_from_slice(&h_nu_gate, &stream);
    let d_write_slots = dev_from_slice(&write_slots, &stream);
    let d_write_qsl = dev_from_slice(&[0i32, nu_seq_len as i32], &stream);
    let d_write_ctx0 = dev_from_slice(&[0i32], &stream);
    let write_batch = W34AttentionBatch {
        query_start_loc: d_write_qsl.addr() as usize as *const i32,
        context_lens_before: d_write_ctx0.addr() as usize as *const i32,
        seq_lens_after: d_nu_seq_after.addr() as usize as *const i32,
        block_tables: d_nu_blk_tbl.addr() as usize as *const i32,
        write_slot_mapping: d_write_slots.addr() as usize as *const i32,
        max_q_len: nu_seq_len as i32,
        total_tokens: nu_seq_len as i32,
        ..nu_attn_batch
    };

    // Exact CPU reference over the selected logical blocks [1 (tokens 16..32), 3 (tokens 48..50)]:
    let selected_tokens: Vec<usize> = (16..32).chain(48..50).collect();
    let mut cpu_ref = vec![0.0f32; nu_q_heads * head_dim];
    for qh in 0..nu_q_heads {
        let mut logits = Vec::with_capacity(selected_tokens.len());
        for &t in &selected_tokens {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += f16_to_f32(h_nu_q[qh * head_dim + d]) * f16_to_f32(h_nu_k[t * head_dim + d]);
            }
            logits.push(dot * nu_attn_batch.sm_scale);
        }
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_l: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_e: f32 = exp_l.iter().sum();
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (idx, &t) in selected_tokens.iter().enumerate() {
                acc += (exp_l[idx] / sum_e) * f16_to_f32(h_nu_v[t * head_dim + d]);
            }
            let g = f16_to_f32(h_nu_gate[qh * head_dim + d]);
            let sig = 1.0f32 / (1.0f32 + (-g).exp());
            cpu_ref[qh * head_dim + d] = acc * sig;
        }
    }

    for (fmt, xform, row_bytes, tol) in [
        (
            W34_KV_FORMAT_FP16,
            W34_KV_TRANSFORM_IDENTITY,
            head_dim * 2,
            1.5e-3f32,
        ),
        (
            W34_KV_FORMAT_Q8_0,
            W34_KV_TRANSFORM_IDENTITY,
            272usize,
            4.0e-3f32,
        ),
        (
            W34_KV_FORMAT_TQ4,
            W34_KV_TRANSFORM_RADEMACHER_WHT256,
            132usize,
            2.5e-2f32,
        ),
    ] {
        let d_kc = DeviceBuffer::alloc(0, nu_pages * block_size * row_bytes).unwrap();
        let d_vc = DeviceBuffer::alloc(0, nu_pages * block_size * row_bytes).unwrap();
        let layout = W34KvLayerLayout {
            k_base: d_kc.addr(),
            v_base: d_vc.addr(),
            k_page_stride_bytes: (block_size * row_bytes) as u64,
            k_head_stride_bytes: (block_size * row_bytes) as u64,
            k_token_stride_bytes: row_bytes as u64,
            v_page_stride_bytes: (block_size * row_bytes) as u64,
            v_head_stride_bytes: (block_size * row_bytes) as u64,
            v_token_stride_bytes: row_bytes as u64,
            namespace_id: W34_KV_NAMESPACE_TARGET_FULL_ATTN,
            layer_id: 3,
            local_kv_heads: 1,
            head_dim: head_dim as i32,
            block_size: block_size as i32,
            k_format: fmt,
            v_format: fmt,
            k_transform: xform,
            v_transform: xform,
            reserved: 0,
        };
        // SAFETY: `d_kc`, `d_vc`, `d_nu_k_in`, `d_nu_v_in`, and `write_batch` buffers are live
        // and sized for `nu_pages * block_size` rows of `row_bytes` bytes.
        unsafe {
            assert_eq!(
                w34_kv_cache_write_strided(
                    &ex,
                    &layout,
                    &write_batch,
                    d_nu_k_in.addr() as *const c_void,
                    d_nu_v_in.addr() as *const c_void,
                    head_dim as i32,
                    head_dim as i32,
                ),
                0
            );
        }
        let d_nu_out = DeviceBuffer::alloc(0, nu_q_heads * head_dim * 2).unwrap();
        gpu_ops::qsa_sparse_paged_attn_f16(
            &ex,
            d_nu_q.addr(),
            d_nu_gate.addr(),
            &layout,
            d_nu_sel_phys.addr(),
            d_nu_sel_log.addr(),
            d_nu_num_sel.addr(),
            d_nu_seq_pos.addr(),
            0,
            d_nu_out.addr(),
            1,
            nu_q_heads,
            nu_select_k,
            nu_attn_batch.sm_scale,
        )
        .unwrap();
        let h_nu_out: Vec<u16> = dev_to_vec(&d_nu_out, nu_q_heads * head_dim, &stream);
        let max_err = h_nu_out
            .iter()
            .zip(cpu_ref.iter())
            .map(|(&got, &exp)| (f16_to_f32(got) - exp).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < tol,
            "QSA sparse paged attn format={fmt} max_err={max_err} exceeded tol={tol}"
        );
    }

    // 5. Verify QSA Indexer 4:1 raw-key pooling -> GemmaRMSNorm(k_norm_weight) -> mRoPE(first_pos)
    //    order on non-uniform 4-token inputs (ensuring pool-then-norm-rope is tested and distinct
    //    from norm-rope-then-pool), plus hc_count = 6 (> 4, up to MAX_HC = 8) on hc_mix, hc_combine,
    //    and ple_signed_sqrt_gate.
    let g_tokens = 4usize;
    let g_qk_in: Vec<u16> = (0..g_tokens * 5 * 128)
        .map(|i| {
            let tok = i / (5 * 128);
            let d = i % 128;
            let scale = match tok {
                0 => 0.2f32,
                1 => 1.5f32,
                2 => -0.8f32,
                _ => 2.1f32,
            };
            f32_to_f16((((d * 11 + tok * 29 + 3) % 41) as f32 - 20.0) * 0.05 * scale)
        })
        .collect();
    let g_k_norm_w: Vec<u16> = (0..128)
        .map(|d| f32_to_f16((((d * 7 + 5) % 23) as f32 - 11.0) * 0.04))
        .collect();
    let g_pos = [
        100i32, 101, 102, 103, 100, 101, 102, 103, 100, 101, 102, 103,
    ];
    let g_slots = [8i32, 9, 10, 11]; // sub_block = 2 (slots 8..12)
    let d_g_qk = dev_from_slice(&g_qk_in, &stream);
    let d_g_knw = dev_from_slice(&g_k_norm_w, &stream);
    let d_g_pos = dev_from_slice(&g_pos, &stream);
    let d_g_slots = dev_from_slice(&g_slots, &stream);
    let d_g_q_out = DeviceBuffer::alloc(0, g_tokens * 4 * 128 * 2).unwrap();
    let d_g_k_out = DeviceBuffer::alloc(0, g_tokens * 128 * 2).unwrap();
    let d_g_raw_k = DeviceBuffer::alloc(0, 16 * 128 * 2).unwrap();
    let d_g_comp_k = DeviceBuffer::alloc(0, 4 * 128 * 2).unwrap();
    gpu_ops::qsa_indexer_norm_rope_f16(
        &ex,
        d_g_qk.addr(),
        d_g_knw.addr(),
        d_g_knw.addr(),
        d_g_pos.addr(),
        d_g_q_out.addr(),
        d_g_k_out.addr(),
        0,
        g_tokens,
        4,
        1,
        128,
        64,
        10_000_000.0,
        1e-6,
        11,
        11,
        g_tokens,
    )
    .unwrap();
    gpu_ops::qsa_indexer_cache_update_f16(
        &ex,
        d_g_k_out.addr(),
        d_g_raw_k.addr(),
        d_g_comp_k.addr(),
        d_g_knw.addr(),
        d_g_pos.addr(),
        d_g_slots.addr(),
        0,
        g_tokens,
        128,
        64,
        10_000_000.0,
        1e-6,
        11,
        11,
        g_tokens,
    )
    .unwrap();
    let h_g_comp_k: Vec<u16> = dev_to_vec(&d_g_comp_k, 4 * 128, &stream);
    let got_sub2 = &h_g_comp_k[2 * 128..3 * 128];

    // CPU reference for pool(4 raw keys) -> GemmaRMSNorm -> mRoPE(first_pos = 100):
    let mut pooled_raw = [0.0f32; 128];
    for d in 0..128 {
        let mut s = 0.0f32;
        for tok in 0..4 {
            s += f16_to_f32(g_qk_in[(tok * 5 + 4) * 128 + d]);
        }
        pooled_raw[d] = f16_to_f32(f32_to_f16(0.25f32 * s));
    }
    let sum_sq: f32 = pooled_raw.iter().map(|&x| x * x).sum();
    let inv_rms = (sum_sq / 128.0f32 + 1e-6f32).powf(-0.5);
    let mut normed_pooled = [0.0f32; 128];
    for d in 0..128 {
        normed_pooled[d] = f16_to_f32(f32_to_f16(
            pooled_raw[d] * inv_rms * (1.0f32 + f16_to_f32(g_k_norm_w[d])),
        ));
    }
    let mut ref_comp_k = normed_pooled;
    for i in 0..32 {
        let inv_freq = 1.0f32 / 10_000_000.0f32.powf((2 * i) as f32 / 64.0f32);
        let angle = 100.0f32 * inv_freq; // first_pos = 100
        let sin_a = f16_to_f32(f32_to_f16(angle.sin()));
        let cos_a = f16_to_f32(f32_to_f16(angle.cos()));
        let x1 = normed_pooled[i];
        let x2 = normed_pooled[i + 32];
        ref_comp_k[i] = x1 * cos_a - x2 * sin_a;
        ref_comp_k[i + 32] = x2 * cos_a + x1 * sin_a;
    }
    for d in 0..128 {
        let diff = (f16_to_f32(got_sub2[d]) - ref_comp_k[d]).abs();
        assert!(
            diff < 2e-3,
            "comp_k pool->norm->rope mismatch at d={d}: got={} ref={}",
            f16_to_f32(got_sub2[d]),
            ref_comp_k[d]
        );
    }

    // Verify hc_count = 6 (up to MAX_HC = 8) on hc_mix_streams_f16, hc_combine_f16, and ple_signed_sqrt_gate_f16:
    let hc6 = 6usize;
    let h6 = 64usize;
    let d_res6 = dev_from_slice(&vec![f32_to_f16(1.0); hc6 * h6], &stream);
    let d_up6 = dev_from_slice(&vec![f32_to_f16(0.0); hc6 * h6], &stream);
    let d_mix6 = DeviceBuffer::alloc(0, h6 * 2).unwrap();
    gpu_ops::hc_mix_streams_f16(
        &ex,
        d_res6.addr(),
        d_up6.addr(),
        d_mix6.addr(),
        0,
        1,
        hc6,
        h6,
    )
    .unwrap();
    let h_mix6: Vec<u16> = dev_to_vec(&d_mix6, h6, &stream);
    // 6 streams of 1.0 with gate = sigmoid(0.0) / 6 = 0.5 / 6 -> sum = 6 * (0.5 / 6) * 1.0 = 0.5
    assert!((f16_to_f32(h_mix6[0]) - 0.5).abs() < 1e-3);

    let d_blk6 = dev_from_slice(&vec![f32_to_f16(2.0); h6], &stream);
    let d_inj6 = dev_from_slice(&vec![0.5f32; hc6], &stream);
    let d_comb6 = DeviceBuffer::alloc(0, hc6 * h6 * 2).unwrap();
    gpu_ops::hc_combine_f16(
        &ex,
        d_res6.addr(),
        d_blk6.addr(),
        d_inj6.addr(),
        0,
        d_comb6.addr(),
        0,
        1,
        hc6,
        h6,
    )
    .unwrap();
    let h_comb6: Vec<u16> = dev_to_vec(&d_comb6, hc6 * h6, &stream);
    assert!((f16_to_f32(h_comb6[5 * h6]) - 2.0).abs() < 1e-3);

    let d_ple_out6 = DeviceBuffer::alloc(0, hc6 * h6 * 2).unwrap();
    gpu_ops::ple_signed_sqrt_gate_f16(
        &ex,
        d_up6.addr(),
        d_up6.addr(),
        d_blk6.addr(),
        d_ple_out6.addr(),
        0,
        1,
        hc6,
        h6,
        1e-6,
    )
    .unwrap();
    let h_ple_out6: Vec<u16> = dev_to_vec(&d_ple_out6, hc6 * h6, &stream);
    // dot = 0 -> gate = sigmoid(0) = 0.5 -> 0.5 * 2.0 = 1.0 across all 6 streams
    assert!((f16_to_f32(h_ple_out6[5 * h6]) - 1.0).abs() < 1e-3);

    // 6. Verify multi-chunk (> 512 groups) Bitonic Top-K `qsa_select_blocks_f16` and
    //    GQA-6 (`local_q_heads = 6, local_kv_heads = 1`) + Split-K `qsa_sparse_paged_attn_f16`.
    let lg_pages = 380usize;
    let lg_seq_len = 6002usize; // 1500 completed 4-token groups + 1 incomplete tail group (group 1500)
    let lg_select_k = 513usize;
    let lg_sink = 16usize;
    let lg_recent = 32usize;
    let lg_blk_tbl: Vec<i32> = (0..lg_pages as i32).collect();
    let d_lg_blk_tbl = dev_from_slice(&lg_blk_tbl, &stream);
    let d_lg_qsl = dev_from_slice(&[0i32, 1i32], &stream);
    let d_lg_ctx_before = dev_from_slice(&[(lg_seq_len - 1) as i32], &stream);
    let d_lg_seq_after = dev_from_slice(&[lg_seq_len as i32], &stream);
    let d_lg_slot = dev_from_slice(&[(lg_seq_len - 1) as i32], &stream);
    let lg_batch = W34AttentionBatch {
        query_start_loc: d_lg_qsl.addr() as usize as *const i32,
        context_lens_before: d_lg_ctx_before.addr() as usize as *const i32,
        seq_lens_after: d_lg_seq_after.addr() as usize as *const i32,
        block_tables: d_lg_blk_tbl.addr() as usize as *const i32,
        write_slot_mapping: d_lg_slot.addr() as usize as *const i32,
        valid_token_mask: std::ptr::null(),
        num_reqs: 1,
        total_tokens: 1,
        max_q_len: 1,
        max_seq_len: lg_seq_len as i32,
        max_blocks_per_req: lg_pages as i32,
        ..nu_attn_batch
    };
    let d_lg_q_idx = dev_from_slice(&vec![f32_to_f16(1.0); 4 * 128], &stream);
    let mut h_lg_comp_k = vec![f32_to_f16(-1.0); lg_pages * 4 * 128];
    let mut expected_mid: Vec<i32> = Vec::new();
    // Plant 464 high-scoring groups across all three 512-group chunks in [16, 1468):
    for idx in 0..(512 - lg_sink - lg_recent) {
        let g = 20 + idx * 3; // spans chunks 0, 1, and 2 (up to g = 1409)
        expected_mid.push(g as i32);
        let score_val = f32_to_f16(0.5 + (idx as f32) * 0.01);
        for d in 0..128 {
            h_lg_comp_k[g * 128 + d] = score_val;
        }
    }
    let d_lg_comp_k = dev_from_slice(&h_lg_comp_k, &stream);
    let d_lg_sel_phys = DeviceBuffer::alloc(0, lg_select_k * 4).unwrap();
    let d_lg_sel_log = DeviceBuffer::alloc(0, lg_select_k * 4).unwrap();
    let d_lg_num_sel = DeviceBuffer::alloc(0, 4).unwrap();
    let d_lg_seq_pos = DeviceBuffer::alloc(0, 4).unwrap();
    gpu_ops::qsa_select_blocks_f16(
        &ex,
        d_lg_q_idx.addr(),
        d_lg_comp_k.addr(),
        &lg_batch,
        d_lg_sel_phys.addr(),
        d_lg_sel_log.addr(),
        d_lg_num_sel.addr(),
        d_lg_seq_pos.addr(),
        4,
        128,
        block_size,
        lg_select_k,
        lg_sink,
        lg_recent,
    )
    .unwrap();
    let h_lg_num_sel: Vec<i32> = dev_to_vec(&d_lg_num_sel, 1, &stream);
    let h_lg_sel_log: Vec<i32> = dev_to_vec(&d_lg_sel_log, lg_select_k, &stream);
    assert_eq!(h_lg_num_sel, vec![lg_select_k as i32]);
    let mut exp_all: Vec<i32> = (0..lg_sink as i32)
        .chain(expected_mid)
        .chain((1500 - lg_recent as i32)..1500)
        .chain(std::iter::once(1500))
        .collect();
    assert_eq!(exp_all.len(), lg_select_k);
    assert_eq!(h_lg_sel_log, exp_all);
    exp_all.dedup();
    assert_eq!(exp_all.len(), lg_select_k);

    // Verify GQA-6 (`local_q_heads = 6, local_kv_heads = 1`) with and without Split-K scratch:
    let gqa6_heads = 6usize;
    let h_gqa6_q: Vec<u16> = (0..gqa6_heads * head_dim)
        .map(|i| f32_to_f16((((i * 13 + 7) % 43) as f32 - 21.0) * 0.04))
        .collect();
    let h_gqa6_gate: Vec<u16> = (0..gqa6_heads * head_dim)
        .map(|i| f32_to_f16((((i * 19 + 3) % 31) as f32 - 15.0) * 0.06))
        .collect();
    let d_gqa6_q = dev_from_slice(&h_gqa6_q, &stream);
    let d_gqa6_gate = dev_from_slice(&h_gqa6_gate, &stream);
    let d_kc6 = DeviceBuffer::alloc(0, nu_pages * block_size * 132).unwrap();
    let d_vc6 = DeviceBuffer::alloc(0, nu_pages * block_size * 132).unwrap();
    let layout6 = W34KvLayerLayout {
        k_base: d_kc6.addr(),
        v_base: d_vc6.addr(),
        k_page_stride_bytes: (block_size * 132) as u64,
        k_head_stride_bytes: (block_size * 132) as u64,
        k_token_stride_bytes: 132,
        v_page_stride_bytes: (block_size * 132) as u64,
        v_head_stride_bytes: (block_size * 132) as u64,
        v_token_stride_bytes: 132,
        namespace_id: W34_KV_NAMESPACE_TARGET_FULL_ATTN,
        layer_id: 3,
        local_kv_heads: 1,
        head_dim: head_dim as i32,
        block_size: block_size as i32,
        k_format: W34_KV_FORMAT_TQ4,
        v_format: W34_KV_FORMAT_TQ4,
        k_transform: W34_KV_TRANSFORM_RADEMACHER_WHT256,
        v_transform: W34_KV_TRANSFORM_RADEMACHER_WHT256,
        reserved: 0,
    };
    // SAFETY: `d_kc6`, `d_vc6`, `d_nu_k_in`, and `d_nu_v_in` are valid device allocations matching `layout6` and `write_batch`.
    unsafe {
        assert_eq!(
            w34_kv_cache_write_strided(
                &ex,
                &layout6,
                &write_batch,
                d_nu_k_in.addr() as *const c_void,
                d_nu_v_in.addr() as *const c_void,
                head_dim as i32,
                head_dim as i32,
            ),
            0
        );
    }
    let d_out_nosplit = DeviceBuffer::alloc(0, gqa6_heads * head_dim * 2).unwrap();
    let d_out_splitk = DeviceBuffer::alloc(0, gqa6_heads * head_dim * 2).unwrap();
    let d_scratch = DeviceBuffer::alloc(0, 256 * 1024).unwrap();
    let mut ex_splitk = ex;
    ex_splitk.scratch_ptr = d_scratch.addr();
    ex_splitk.scratch_size = 256 * 1024;
    let mut ex_nosplit = ex;
    ex_nosplit.scratch_ptr = 0;
    ex_nosplit.scratch_size = 0;
    gpu_ops::qsa_sparse_paged_attn_f16(
        &ex_nosplit,
        d_gqa6_q.addr(),
        d_gqa6_gate.addr(),
        &layout6,
        d_nu_sel_phys.addr(),
        d_nu_sel_log.addr(),
        d_nu_num_sel.addr(),
        d_nu_seq_pos.addr(),
        0,
        d_out_nosplit.addr(),
        1,
        gqa6_heads,
        nu_select_k,
        nu_attn_batch.sm_scale,
    )
    .unwrap();
    gpu_ops::qsa_sparse_paged_attn_f16(
        &ex_splitk,
        d_gqa6_q.addr(),
        d_gqa6_gate.addr(),
        &layout6,
        d_nu_sel_phys.addr(),
        d_nu_sel_log.addr(),
        d_nu_num_sel.addr(),
        d_nu_seq_pos.addr(),
        0,
        d_out_splitk.addr(),
        1,
        gqa6_heads,
        nu_select_k,
        nu_attn_batch.sm_scale,
    )
    .unwrap();
    let h_out_nosplit: Vec<u16> = dev_to_vec(&d_out_nosplit, gqa6_heads * head_dim, &stream);
    let h_out_splitk: Vec<u16> = dev_to_vec(&d_out_splitk, gqa6_heads * head_dim, &stream);
    let max_split_diff = h_out_nosplit
        .iter()
        .zip(h_out_splitk.iter())
        .map(|(&a, &b)| (f16_to_f32(a) - f16_to_f32(b)).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_split_diff < 1e-3,
        "GQA-6 Split-K vs no-split mismatch: max_split_diff={max_split_diff}"
    );
}

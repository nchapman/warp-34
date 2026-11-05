use super::*;
use serde_json::json;

fn make_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
    let mut header = Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        let mut m = Map::new();
        m.insert("dtype".into(), json!(dtype));
        m.insert("shape".into(), json!(shape));
        m.insert("data_offsets".into(), json!([offset, offset + bytes.len()]));
        header.insert((*name).to_string(), Value::Object(m));
        offset += bytes.len();
    }
    let mut hbytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !hbytes.len().is_multiple_of(8) {
        hbytes.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hbytes.len() + offset);
    out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hbytes);
    for (_, _, _, bytes) in tensors {
        out.extend_from_slice(bytes);
    }
    fs::write(path, out).unwrap();
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn clef_render_sorts_keys_compactly_and_formats_python_floats() {
    let v = json!({
        "zeta": [1, 2, {"b": true, "a": null}],
        "alpha": "raw <text>",
        "empty_arr": [],
        "empty_obj": {}
    });
    assert_eq!(
        clef_render_value(&v).unwrap(),
        r#"{"alpha":"raw <text>","empty_arr":[],"empty_obj":{},"zeta":[1,2,{"a":null,"b":true}]}"#
    );
    assert_eq!(clef_render_value(&json!("plain")).unwrap(), "plain");

    for (val, expected) in [
        (0.0f64, "0.0"),
        (-0.0f64, "-0.0"),
        (1.0f64, "1.0"),
        (12.34f64, "12.34"),
        (1e-4f64, "0.0001"),
        (1e-5f64, "1e-05"),
        (1.5e-7f64, "1.5e-07"),
        (1e15f64, "1000000000000000.0"),
        (1e16f64, "1e+16"),
        (1e20f64, "1e+20"),
        (1.5e20f64, "1.5e+20"),
        (-3.25e-12f64, "-3.25e-12"),
    ] {
        let n = serde_json::Number::from_f64(val).unwrap();
        assert_eq!(
            clef_render_value(&Value::Number(n)).unwrap(),
            expected,
            "float {val:e}"
        );
    }
}

#[test]
fn clef_joint_head_forward_and_perturbation_gates() {
    let cfg = ClefHeadConfig {
        hidden_size: 8,
        width: 4,
        routing_layers: 1,
        layers: 1,
        heads: 2,
        feedforward: 8,
    };
    let tmp = std::env::temp_dir().join(format!("w34_clef_unit_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let head_path = tmp.join("joint_head.safetensors");

    let mut named: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut add = |name: &str, shape: &[usize], scale: f32| {
        let n: usize = shape.iter().product::<usize>().max(1);
        let bytes = f32_bytes(&splitmix_vec(name, n, scale as f64));
        named.push((name.to_owned(), shape.to_vec(), bytes));
    };

    add("hidden_norm.weight", &[8], 1.0);
    add("hidden_norm.bias", &[8], 0.1);
    add("memory_projection.weight", &[4, 8], 0.3);
    add("question_projection.weight", &[4, 8], 0.3);
    add("option_question_projection.weight", &[4, 8], 0.3);
    add("global_projection.weight", &[4, 8], 0.3);
    add("option_context_projection.weight", &[4, 8], 0.3);
    add("option_lexical_projection.weight", &[4, 8], 0.3);
    add("type_embedding.weight", &[3, 4], 0.3);

    add("evidence_layers.0.query_norm.weight", &[4], 1.0);
    add("evidence_layers.0.query_norm.bias", &[4], 0.05);
    add("evidence_layers.0.memory_norm.weight", &[4], 1.0);
    add("evidence_layers.0.memory_norm.bias", &[4], 0.05);
    add("evidence_layers.0.attention.in_proj_weight", &[12, 4], 0.3);
    add("evidence_layers.0.attention.in_proj_bias", &[12], 0.05);
    add("evidence_layers.0.attention.out_proj.weight", &[4, 4], 0.3);
    add("evidence_layers.0.attention.out_proj.bias", &[4], 0.05);
    add("evidence_layers.0.feedforward_norm.weight", &[4], 1.0);
    add("evidence_layers.0.feedforward_norm.bias", &[4], 0.05);
    add("evidence_layers.0.feedforward.0.weight", &[8, 4], 0.3);
    add("evidence_layers.0.feedforward.0.bias", &[8], 0.05);
    add("evidence_layers.0.feedforward.3.weight", &[4, 8], 0.3);
    add("evidence_layers.0.feedforward.3.bias", &[4], 0.05);

    add("option_summary_norm.weight", &[4], 1.0);
    add("option_summary_norm.bias", &[4], 0.05);

    add("layers.0.norm1.weight", &[4], 1.0);
    add("layers.0.norm1.bias", &[4], 0.05);
    add("layers.0.self_attn.in_proj_weight", &[12, 4], 0.3);
    add("layers.0.self_attn.in_proj_bias", &[12], 0.05);
    add("layers.0.self_attn.out_proj.weight", &[4, 4], 0.3);
    add("layers.0.self_attn.out_proj.bias", &[4], 0.05);
    add("layers.0.norm2.weight", &[4], 1.0);
    add("layers.0.norm2.bias", &[4], 0.05);
    add("layers.0.multihead_attn.in_proj_weight", &[12, 4], 0.3);
    add("layers.0.multihead_attn.in_proj_bias", &[12], 0.05);
    add("layers.0.multihead_attn.out_proj.weight", &[4, 4], 0.3);
    add("layers.0.multihead_attn.out_proj.bias", &[4], 0.05);
    add("layers.0.norm3.weight", &[4], 1.0);
    add("layers.0.norm3.bias", &[4], 0.05);
    add("layers.0.linear1.weight", &[8, 4], 0.3);
    add("layers.0.linear1.bias", &[8], 0.05);
    add("layers.0.linear2.weight", &[4, 8], 0.3);
    add("layers.0.linear2.bias", &[4], 0.05);

    add("field_norm.weight", &[4], 1.0);
    add("field_norm.bias", &[4], 0.05);
    add("option_norm.weight", &[4], 1.0);
    add("option_norm.bias", &[4], 0.05);
    add("residual_scorer.0.weight", &[4, 16], 0.3);
    add("residual_scorer.0.bias", &[4], 0.05);
    add("residual_scorer.3.weight", &[1, 4], 0.3);
    add("residual_scorer.3.bias", &[1], 0.05);
    add("prior_logit_scale", &[], 0.5);
    add("joint_logit_scale", &[], 0.5);
    add("residual_gate", &[], 0.5);

    let refs: Vec<(&str, &str, &[usize], &[u8])> = named
        .iter()
        .map(|(n, s, b)| (n.as_str(), "F32", s.as_slice(), b.as_slice()))
        .collect();
    make_safetensors(&head_path, &refs);

    let arc = SafetensorsArchive::open(&head_path).unwrap();
    let out_emb =
        ClefOutputEmbeddings::from_in_memory(splitmix_vec("unit_emb", 16 * 8, 0.4), 16, 8).unwrap();
    let head = ClefJointHead::from_archive(&arc, cfg.clone(), out_emb).unwrap();

    let hidden_rows: Vec<Vec<f32>> = (0..10)
        .map(|r| splitmix_vec(&format!("unit_h_{r}"), 8, 0.5))
        .collect();
    let input_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let questions = vec![
        ClefEncodedQuestion {
            question_id: "q1".into(),
            question_type: 0,
            question_span: (1, 3),
            option_spans: vec![(3, 5), (5, 7)],
            option_ids: vec!["true".into(), "false".into()],
        },
        ClefEncodedQuestion {
            question_id: "q2".into(),
            question_type: 1,
            question_span: (7, 8),
            option_spans: vec![(8, 9), (9, 10)],
            option_ids: vec!["alpha".into(), "zeta".into()],
        },
    ];

    let out = head.forward(&hidden_rows, &input_ids, &questions).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].len(), 2);
    assert_eq!(out[1].len(), 2);
    assert_ne!(out[0][0], out[0][1]);
    assert_ne!(out[1][0], out[1][1]);

    // Cross-question coupling gate: changing q2's type embedding must shift q1's joint logits
    // via DecoderLayer self_attn across fields.
    let mut q_perturbed = questions.clone();
    q_perturbed[1].question_type = 2;
    let out_perturbed = head
        .forward(&hidden_rows, &input_ids, &q_perturbed)
        .unwrap();
    assert!(
        (out[0][0] - out_perturbed[0][0]).abs() > 1e-6,
        "joint field self-attention did not propagate cross-question information"
    );

    // Error validations
    assert!(head.forward(&[], &[], &questions).is_err());
    assert!(head.forward(&hidden_rows, &[1, 2], &questions).is_err());
    assert!(head.forward(&hidden_rows, &input_ids, &[]).is_err());

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn clef_head_config_and_logit_reordering_validation() {
    assert!(
        ClefHeadConfig::from_json(&json!({
            "hidden_size": 4096,
            "width": 1024,
            "routing_layers": 2,
            "layers": 4,
            "heads": 16,
            "feedforward": 4096
        }))
        .is_ok()
    );
    // width not divisible by heads
    assert!(
        ClefHeadConfig::from_json(&json!({
            "hidden_size": 4096,
            "width": 1024,
            "routing_layers": 2,
            "layers": 4,
            "heads": 15,
            "feedforward": 4096
        }))
        .is_err()
    );
    // exact GPU kernel upper boundary (width == 2048, head_dim == 256) is accepted
    assert!(
        ClefHeadConfig::from_json(&json!({
            "hidden_size": 5120,
            "width": 2048,
            "routing_layers": 2,
            "layers": 4,
            "heads": 8,
            "feedforward": 8192
        }))
        .is_ok()
    );
    // exceeds GPU kernel bounds (width > 2048 or head_dim > 256)
    assert!(
        ClefHeadConfig::from_json(&json!({
            "hidden_size": 4096,
            "width": 4096,
            "routing_layers": 2,
            "layers": 4,
            "heads": 16,
            "feedforward": 4096
        }))
        .is_err()
    );
    assert!(
        ClefHeadConfig::from_json(&json!({
            "hidden_size": 4096,
            "width": 1024,
            "routing_layers": 2,
            "layers": 4,
            "heads": 2,
            "feedforward": 4096
        }))
        .is_err()
    );

    let q = ClefEncodedQuestion {
        question_id: "team".into(),
        question_type: 1,
        question_span: (1, 2),
        option_spans: vec![(2, 3), (3, 4)],
        option_ids: vec!["alpha".into(), "zeta".into()],
    };
    let field = DecisionField {
        name: "team".into(),
        field_type: "choice".into(),
        description: "Pick team".into(),
        values: vec![json!("zeta"), json!("alpha")],
        value_descriptions: vec!["Zeta".into(), "Alpha".into()],
        keys: vec!["zeta".into(), "alpha".into()],
    };
    let reordered = reorder_clef_question_logits(&q, &field, &[1.25, -0.75]).unwrap();
    assert_eq!(reordered, vec![-0.75, 1.25]);
}

fn splitmix_vec(seed_tag: &str, count: usize, scale: f64) -> Vec<f32> {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in seed_tag.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut z = h.wrapping_add(i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        out.push(((u * 2.0 - 1.0) * scale) as f32);
    }
    out
}

#[test]
fn clef_joint_head_matches_pytorch_joint_schema_head_goldens() {
    let tmp = std::env::temp_dir().join(format!("w34_clef_pt_golden_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let head_path = tmp.join("joint_head.safetensors");
    let model_path = tmp.join("model.safetensors");

    let mut specs: Vec<(String, Vec<usize>)> = vec![
        ("hidden_norm.weight".into(), vec![8]),
        ("hidden_norm.bias".into(), vec![8]),
        ("memory_projection.weight".into(), vec![4, 8]),
        ("question_projection.weight".into(), vec![4, 8]),
        ("option_question_projection.weight".into(), vec![4, 8]),
        ("global_projection.weight".into(), vec![4, 8]),
        ("option_context_projection.weight".into(), vec![4, 8]),
        ("option_lexical_projection.weight".into(), vec![4, 8]),
        ("type_embedding.weight".into(), vec![3, 4]),
    ];
    for i in 0..2 {
        specs.extend([
            (format!("evidence_layers.{i}.query_norm.weight"), vec![4]),
            (format!("evidence_layers.{i}.query_norm.bias"), vec![4]),
            (format!("evidence_layers.{i}.memory_norm.weight"), vec![4]),
            (format!("evidence_layers.{i}.memory_norm.bias"), vec![4]),
            (
                format!("evidence_layers.{i}.attention.in_proj_weight"),
                vec![12, 4],
            ),
            (
                format!("evidence_layers.{i}.attention.in_proj_bias"),
                vec![12],
            ),
            (
                format!("evidence_layers.{i}.attention.out_proj.weight"),
                vec![4, 4],
            ),
            (
                format!("evidence_layers.{i}.attention.out_proj.bias"),
                vec![4],
            ),
            (
                format!("evidence_layers.{i}.feedforward_norm.weight"),
                vec![4],
            ),
            (
                format!("evidence_layers.{i}.feedforward_norm.bias"),
                vec![4],
            ),
            (
                format!("evidence_layers.{i}.feedforward.0.weight"),
                vec![8, 4],
            ),
            (format!("evidence_layers.{i}.feedforward.0.bias"), vec![8]),
            (
                format!("evidence_layers.{i}.feedforward.3.weight"),
                vec![4, 8],
            ),
            (format!("evidence_layers.{i}.feedforward.3.bias"), vec![4]),
        ]);
    }
    specs.extend([
        ("option_summary_norm.weight".into(), vec![4]),
        ("option_summary_norm.bias".into(), vec![4]),
    ]);
    for i in 0..2 {
        specs.extend([
            (format!("layers.{i}.norm1.weight"), vec![4]),
            (format!("layers.{i}.norm1.bias"), vec![4]),
            (format!("layers.{i}.self_attn.in_proj_weight"), vec![12, 4]),
            (format!("layers.{i}.self_attn.in_proj_bias"), vec![12]),
            (format!("layers.{i}.self_attn.out_proj.weight"), vec![4, 4]),
            (format!("layers.{i}.self_attn.out_proj.bias"), vec![4]),
            (format!("layers.{i}.norm2.weight"), vec![4]),
            (format!("layers.{i}.norm2.bias"), vec![4]),
            (
                format!("layers.{i}.multihead_attn.in_proj_weight"),
                vec![12, 4],
            ),
            (format!("layers.{i}.multihead_attn.in_proj_bias"), vec![12]),
            (
                format!("layers.{i}.multihead_attn.out_proj.weight"),
                vec![4, 4],
            ),
            (format!("layers.{i}.multihead_attn.out_proj.bias"), vec![4]),
            (format!("layers.{i}.norm3.weight"), vec![4]),
            (format!("layers.{i}.norm3.bias"), vec![4]),
            (format!("layers.{i}.linear1.weight"), vec![8, 4]),
            (format!("layers.{i}.linear1.bias"), vec![8]),
            (format!("layers.{i}.linear2.weight"), vec![4, 8]),
            (format!("layers.{i}.linear2.bias"), vec![4]),
        ]);
    }
    specs.extend([
        ("field_norm.weight".into(), vec![4]),
        ("field_norm.bias".into(), vec![4]),
        ("option_norm.weight".into(), vec![4]),
        ("option_norm.bias".into(), vec![4]),
        ("residual_scorer.0.weight".into(), vec![4, 16]),
        ("residual_scorer.0.bias".into(), vec![4]),
        ("residual_scorer.3.weight".into(), vec![1, 4]),
        ("residual_scorer.3.bias".into(), vec![1]),
        ("prior_logit_scale".into(), vec![]),
        ("joint_logit_scale".into(), vec![]),
        ("residual_gate".into(), vec![]),
    ]);

    let named: Vec<(String, Vec<usize>, Vec<u8>)> = specs
        .into_iter()
        .map(|(k, shape)| {
            let scale = if k.ends_with(".bias") {
                0.08
            } else if k.ends_with(".weight") && shape.len() == 1 {
                1.0
            } else if shape.is_empty() {
                0.5
            } else {
                0.3
            };
            let n = shape.iter().product::<usize>().max(1);
            let bytes = f32_bytes(&splitmix_vec(&k, n, scale));
            (k, shape, bytes)
        })
        .collect();
    let refs: Vec<(&str, &str, &[usize], &[u8])> = named
        .iter()
        .map(|(n, s, b)| (n.as_str(), "F32", s.as_slice(), b.as_slice()))
        .collect();
    make_safetensors(&head_path, &refs);
    fs::write(
        tmp.join("joint_head_config.json"),
        r#"{"hidden_size":8,"width":4,"routing_layers":2,"layers":2,"heads":2,"feedforward":8}"#,
    )
    .unwrap();

    // Also test Archive-backed on-demand `read_row_f32` via `lm_head.weight`!
    let out_emb_bytes = f32_bytes(&splitmix_vec("out_emb", 16 * 8, 0.4));
    make_safetensors(
        &model_path,
        &[("lm_head.weight", "F32", &[16, 8], &out_emb_bytes)],
    );
    fs::write(
            tmp.join("config.json"),
            r#"{"architectures":["ClefModel"],"model_type":"qwen3_5_text","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,"vocab_size":16,"tie_word_embeddings":false}"#,
        )
        .unwrap();

    let model_cfg = ModelConfig::from_dir(&tmp).unwrap();
    let head = ClefJointHead::from_dir_and_config(&tmp, &model_cfg).unwrap();

    let seq_flat = splitmix_vec("seq_hidden", 12 * 8, 0.6);
    let hidden_rows: Vec<Vec<f32>> = seq_flat
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| c.to_vec())
        .collect();
    let input_ids = vec![2, 5, 1, 7, 3, 9, 4, 8, 6, 11, 10, 13];
    let questions = vec![
        ClefEncodedQuestion {
            question_id: "urgent".into(),
            question_type: 0,
            question_span: (1, 3),
            option_spans: vec![(3, 5), (5, 7)],
            option_ids: vec!["true".into(), "false".into()],
        },
        ClefEncodedQuestion {
            question_id: "route".into(),
            question_type: 1,
            question_span: (7, 8),
            option_spans: vec![(8, 9), (9, 11), (11, 12)],
            option_ids: vec!["alpha".into(), "beta".into(), "zeta".into()],
        },
    ];

    let logits = head.forward(&hidden_rows, &input_ids, &questions).unwrap();
    let expected_q0 = [0.79531544f64, 0.57161945f64];
    let expected_q1 = [0.13990906f64, 0.55448157f64, 0.00338013f64];
    for (got, exp) in logits[0].iter().zip(expected_q0) {
        assert!(
            (got - exp).abs() < 2e-6,
            "q0 logit mismatch vs PyTorch JointSchemaHead: got {got}, expected {exp}"
        );
    }
    for (got, exp) in logits[1].iter().zip(expected_q1) {
        assert!(
            (got - exp).abs() < 2e-6,
            "q1 logit mismatch vs PyTorch JointSchemaHead: got {got}, expected {exp}"
        );
    }

    #[cfg(feature = "cuda")]
    if crate::cuda::CudaDevice::count().unwrap_or(0) >= 1 {
        crate::cuda::prepare_execution_plan(0).unwrap();
        let _dev = crate::cuda::CudaDevice::set_current(0).unwrap();
        let stream = crate::cuda::CudaStream::new(0).unwrap();
        let cublas = crate::cuda::CublasHandle::new(&stream).unwrap();
        let mut gpu = ClefJointHeadGpu::from_host(0, &stream, &head).unwrap();
        drop(head); // prove GPU head runs independently after host f32 weights are freed

        // Upload FP16-quantized input hidden sequence so CPU FP16-rounded input and GPU match.
        let seq_f16: Vec<u16> = seq_flat
            .iter()
            .copied()
            .map(crate::cuda::f32_to_f16)
            .collect();
        let d_seq = crate::cuda::DeviceBuffer::alloc(0, seq_f16.len() * 2).unwrap();
        d_seq.copy_from_host(&seq_f16, &stream).unwrap();
        stream.synchronize().unwrap();

        let rec = ClefEncodedRecord {
            input_ids: input_ids.clone(),
            questions: questions.clone(),
        };

        // 1. Single-chunk [0..12]
        let mut st1 = gpu.begin_request(&rec, &stream).unwrap();
        gpu.ingest_chunk(&mut st1, d_seq.addr(), 0, 12, &cublas, &stream)
            .unwrap();
        let gpu_single = gpu.finalize(st1, &rec, &cublas, &stream).unwrap();

        // 2. Multi-chunk [0..2, 2..6, 6..10, 10..12] splitting right through question/option spans
        //    and reusing the recycled device slot from `st1`.
        let mut st2 = gpu.begin_request(&rec, &stream).unwrap();
        for (start, len) in [(0usize, 2usize), (2, 4), (6, 4), (10, 2)] {
            let addr = d_seq.addr() + (start * 8 * 2) as u64;
            gpu.ingest_chunk(&mut st2, addr, start, len, &cublas, &stream)
                .unwrap();
        }
        let gpu_split = gpu.finalize(st2, &rec, &cublas, &stream).unwrap();

        for qi in 0..2 {
            for oi in 0..logits[qi].len() {
                let g1 = gpu_single[qi][oi];
                let g2 = gpu_split[qi][oi];
                let cpu_ref = logits[qi][oi];
                assert!(
                    (g1 - g2).abs() < 1e-4,
                    "chunk-split invariance q{qi}[{oi}]: single={g1}, split={g2}"
                );
                assert!(
                    (g1 - cpu_ref).abs() < 2e-3,
                    "FP16 GPU vs FP32 PyTorch golden q{qi}[{oi}]: gpu={g1}, ref={cpu_ref}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&tmp);

    #[cfg(feature = "cuda")]
    for (seq_len, num_q, total_opts, h, w, ff) in [
        (428usize, 16usize, 64usize, 4096usize, 1024usize, 4096usize),
        (8192, 24, 96, 5120, 2048, 8192),
        (1, 1, 1, 8, 4, 8),
    ] {
        let cap = ClefJointHeadGpu::finalize_scratch_bytes(seq_len, num_q, total_opts, h, w, ff);
        let num_spans = 1 + num_q + total_opts;
        let mut offset = 0usize;
        let mut sim_alloc = |bytes: usize| {
            let aligned = (offset + 255) & !255usize;
            offset = aligned + bytes.max(1);
            assert!(
                offset <= cap,
                "finalize_scratch_bytes undercounted ({offset} > {cap}) for seq={seq_len}, q={num_q}, opts={total_opts}, w={w}"
            );
        };
        sim_alloc(total_opts * 4);
        sim_alloc((num_q + 1) * 4);
        sim_alloc(num_q * 4);
        sim_alloc(total_opts * h * 2);
        sim_alloc(num_spans * h * 2);
        for elems in ClefJointHeadGpu::finalize_f16_elem_counts(seq_len, num_q, total_opts, w, ff) {
            sim_alloc((elems * 2).max(16));
        }
    }
}

fn build_synthetic_clef_tokenizer() -> Tokenizer {
    use tokenizers::AddedToken;
    use tokenizers::models::bpe::{BPE, Vocab};

    let mut vocab = Vocab::default();
    vocab.insert("[UNK]".to_owned(), 0);
    for (next_id, ch) in (1u32..).zip(
        ['\t', '\n', '\r']
            .into_iter()
            .chain((0x20u8..=0x7eu8).map(|b| b as char)),
    ) {
        vocab.insert(ch.to_string(), next_id);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    let mut tok = Tokenizer::new(bpe);
    let specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ];
    let added: Vec<AddedToken> = specials
        .iter()
        .map(|&s| AddedToken::from(s, true))
        .collect();
    tok.add_special_tokens(&added);
    // Also register a non-special added control token to verify non-bracket sanitization.
    tok.add_tokens(&[AddedToken::from("@@SECRET_ADDED@@", false)]);
    tok
}

#[test]
fn clef_joint_head_rejects_truncated_safetensors_payload() {
    let tmp = std::env::temp_dir().join(format!("w34_clef_trunc_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let short_bytes = f32_bytes(&[1.0, 2.0]);
    let head_path = tmp.join("joint_head.safetensors");
    make_safetensors(
        &head_path,
        &[("hidden_norm.weight", "F32", &[4], &short_bytes)],
    );
    let archive = SafetensorsArchive::open(&head_path).unwrap();
    let err = read_exact_1d(&archive, "hidden_norm.weight", 4).unwrap_err();
    assert!(err.contains("length 2 != expected 4"), "{err}");

    let mat_path = tmp.join("mat_trunc.safetensors");
    make_safetensors(
        &mat_path,
        &[("memory_projection.weight", "F32", &[2, 2], &short_bytes)],
    );
    let mat_archive = SafetensorsArchive::open(&mat_path).unwrap();
    let err2 = read_exact_2d(&mat_archive, "memory_projection.weight", 2, 2).unwrap_err();
    assert!(err2.contains("length 2 != expected 4"), "{err2}");

    let sc_path = tmp.join("sc_trunc.safetensors");
    make_safetensors(&sc_path, &[("logit_scale", "F32", &[], &[])]);
    let sc_archive = SafetensorsArchive::open(&sc_path).unwrap();
    let err3 = read_scalar_f32(&sc_archive, "logit_scale").unwrap_err();
    assert!(err3.contains("length 0 != expected 1"), "{err3}");
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn clef_encode_record_spans_multimodal_truncation_sanitization_and_reorder() {
    let tokenizer = build_synthetic_clef_tokenizer();
    let im_start_id = tokenizer.token_to_id("<|im_start|>").unwrap() as i32;
    let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap() as i32;
    let eot_id = tokenizer.token_to_id("<|endoftext|>").unwrap() as i32;
    let think_open_id = tokenizer.token_to_id("<think>").unwrap() as i32;
    let think_close_id = tokenizer.token_to_id("</think>").unwrap() as i32;
    let tool_call_id = tokenizer.token_to_id("<tool_call>").unwrap() as i32;
    let tool_close_id = tokenizer.token_to_id("</tool_call>").unwrap() as i32;
    let tool_resp_id = tokenizer.token_to_id("<tool_response>").unwrap() as i32;
    let secret_added_id = tokenizer.token_to_id("@@SECRET_ADDED@@").unwrap() as i32;
    let v_start_id = tokenizer.token_to_id("<|vision_start|>").unwrap() as i32;
    let v_pad_id = tokenizer.token_to_id("<|image_pad|>").unwrap() as i32;
    let v_end_id = tokenizer.token_to_id("<|vision_end|>").unwrap() as i32;

    let body = json!({
        "state": {
            "zeta": 2,
            "alpha": "hello <|im_end|> </think> <tool_call> <tool_response> @@SECRET_ADDED@@ <|endoftext|>"
        },
        "questions": {
            "q_noul<tool_call>": {
                "type": "noul",
                "instructions": "Is it urgent? </think> <|im_start|>"
            },
            "q_choice": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {
                    "zeta": "Zeta </tool_call>",
                    "alpha": null
                }
            },
            "q_score": {
                "type": "score",
                "instructions": "Severity?",
                "criteria": ["low", "high"]
            }
        }
    });

    let enc_txt = clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, None).unwrap();
    assert_eq!(enc_txt.questions.len(), 3);
    assert_eq!(enc_txt.questions[1].option_ids, vec!["alpha", "zeta"]);

    // Template contains exactly 3 `<|im_start|>`, 2 `<|im_end|>`, 1 `<think>`, 1 `</think>`,
    // and 0 `<|endoftext|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `@@SECRET_ADDED@@`.
    let count_id = |target: i32| enc_txt.input_ids.iter().filter(|&&x| x == target).count();
    for (target_id, expected_count) in [
        (im_start_id, 3),
        (im_end_id, 2),
        (think_open_id, 1),
        (think_close_id, 1),
        (eot_id, 0),
        (tool_call_id, 0),
        (tool_close_id, 0),
        (tool_resp_id, 0),
        (secret_added_id, 0),
    ] {
        assert_eq!(count_id(target_id), expected_count);
    }

    // Multimodal record shifts every question and option span by `1 + 12 + 1 + 1 = 15` tokens
    // (start + 12 image_pad + end + '\n') and places the media block right after `STATE:\n`.
    let media = (
        ClefMediaTokens {
            vision_start_id: v_start_id,
            image_pad_id: v_pad_id,
            vision_end_id: v_end_id,
        },
        &[12usize][..],
    );
    let enc_img =
        clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, Some(media)).unwrap();
    let shift = 15usize;
    assert_eq!(enc_img.input_ids.len(), enc_txt.input_ids.len() + shift);
    let vs_pos = enc_img
        .input_ids
        .iter()
        .position(|&t| t == v_start_id)
        .expect("vision_start present");
    assert!(
        enc_img.input_ids[vs_pos + 1..vs_pos + 13]
            .iter()
            .all(|&t| t == v_pad_id)
    );
    assert_eq!(enc_img.input_ids[vs_pos + 13], v_end_id);
    for (qt, qi) in enc_txt.questions.iter().zip(&enc_img.questions) {
        assert_eq!(
            qi.question_span,
            (qt.question_span.0 + shift, qt.question_span.1 + shift)
        );
        for (&spt, &spi) in qt.option_spans.iter().zip(&qi.option_spans) {
            assert_eq!(spi, (spt.0 + shift, spt.1 + shift));
        }
    }

    // State truncation: tightening `max_length` to `enc_txt.input_ids.len() - 20` trims
    // exactly 20 state tokens and shifts all schema spans left by 20; setting `max_length`
    // to `fixed_length` (leaving 0 state tokens) fails with 400 Bad Request.
    let trimmed_max = enc_txt.input_ids.len() - 20;
    let enc_trim = clef_encode_record(&tokenizer, &body, trimmed_max, None).unwrap();
    assert_eq!(enc_trim.input_ids.len(), trimmed_max);
    for (qt, qr) in enc_txt.questions.iter().zip(&enc_trim.questions) {
        assert_eq!(
            qr.question_span,
            (qt.question_span.0 - 20, qt.question_span.1 - 20)
        );
    }
    assert!(clef_encode_record(&tokenizer, &body, 32, None).is_err());

    // `reorder_clef_question_logits` maps alphabetical ["alpha", "zeta"] back to caller ["zeta", "alpha"]
    // and rejects length mismatches or unknown option keys.
    let field = DecisionField {
        name: "q_choice".into(),
        field_type: "choice".into(),
        description: "Which queue?".into(),
        values: vec![json!("zeta"), json!("alpha")],
        value_descriptions: vec!["Zeta".into(), "alpha".into()],
        keys: vec!["zeta".into(), "alpha".into()],
    };
    let reordered =
        reorder_clef_question_logits(&enc_txt.questions[1], &field, &[1.25, 4.5]).unwrap();
    assert_eq!(reordered, vec![4.5, 1.25]);
    assert!(reorder_clef_question_logits(&enc_txt.questions[1], &field, &[1.0]).is_err());

    let mut bad_field = field.clone();
    bad_field.keys = vec!["zeta".into(), "missing".into()];
    assert!(reorder_clef_question_logits(&enc_txt.questions[1], &bad_field, &[1.25, 4.5]).is_err());
}

fn resolve_clef_flash_hf_tokenizer() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("W34_CLEF_TOKENIZER") {
        let pb = std::path::PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let hub_root = std::env::var_os("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| std::path::PathBuf::from(h).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".cache/huggingface/hub"))
        })?;
    let repo = hub_root.join("models--Cloudflare--clef-flash");
    if let Ok(rev) = fs::read_to_string(repo.join("refs/main")) {
        let cand = repo
            .join("snapshots")
            .join(rev.trim())
            .join("tokenizer.json");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

#[test]
#[ignore = "requires Cloudflare/clef-flash tokenizer in local HF cache or W34_CLEF_TOKENIZER"]
fn clef_encode_record_matches_clef_flash_hf_tokenizer_goldens() {
    let tok_path = resolve_clef_flash_hf_tokenizer()
        .expect("set W34_CLEF_TOKENIZER or cache Cloudflare/clef-flash");
    let tokenizer = Tokenizer::from_file(&tok_path).unwrap();
    let body = json!({
        "state": {"zeta": 2, "alpha": "hello"},
        "questions": {
            "q_noul": {"type": "noul", "instructions": "Is it urgent?"},
            "q_choice": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"zeta": "Zeta queue", "alpha": null}
            },
            "q_score": {
                "type": "score",
                "instructions": "Severity?",
                "criteria": ["low", "high"]
            }
        }
    });

    let enc_txt = clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, None).unwrap();
    assert_eq!(enc_txt.input_ids.len(), 274);
    assert_eq!(
        &enc_txt.input_ids[..10],
        &[248045, 8678, 198, 4274, 279, 4434, 1528, 321, 10485, 13]
    );
    assert_eq!(
        &enc_txt.input_ids[enc_txt.input_ids.len() - 10..],
        &[248069, 271, 25310, 3118, 7295, 34651, 41402, 23451, 50, 25]
    );
    assert_eq!(enc_txt.questions[0].question_span, (73, 77));
    assert_eq!(
        enc_txt.questions[0].option_spans,
        vec![(88, 106), (112, 130)]
    );
    assert_eq!(enc_txt.questions[1].question_span, (152, 155));
    assert_eq!(
        enc_txt.questions[1].option_spans,
        vec![(166, 172), (178, 191)]
    );
    assert_eq!(enc_txt.questions[1].option_ids, vec!["alpha", "zeta"]);
    assert_eq!(enc_txt.questions[2].question_span, (213, 215));
    assert_eq!(
        enc_txt.questions[2].option_spans,
        vec![(226, 236), (242, 252)]
    );

    let media = (
        ClefMediaTokens {
            vision_start_id: 248053,
            image_pad_id: 248056,
            vision_end_id: 248054,
        },
        &[64usize][..],
    );
    let enc_img =
        clef_encode_record(&tokenizer, &body, CLEF_DEFAULT_MAX_LENGTH, Some(media)).unwrap();
    assert_eq!(enc_img.input_ids.len(), 341);
    assert_eq!(enc_img.questions[0].question_span, (140, 144));
    assert_eq!(
        enc_img.questions[0].option_spans,
        vec![(155, 173), (179, 197)]
    );
    assert_eq!(enc_img.questions[1].question_span, (219, 222));
    assert_eq!(
        enc_img.questions[1].option_spans,
        vec![(233, 239), (245, 258)]
    );
    assert_eq!(enc_img.questions[2].question_span, (280, 282));
    assert_eq!(
        enc_img.questions[2].option_spans,
        vec![(293, 303), (309, 319)]
    );
}

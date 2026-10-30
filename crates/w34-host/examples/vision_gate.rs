//! Replay a real-image vLLM capture, fail closed on input, embedding, or greedy mismatch.
//! cargo run --release --features cuda --example vision_gate -- ORACLE_DIR OUTPUT.json
#[cfg(not(feature = "cuda"))]
fn main() -> std::process::ExitCode {
    eprintln!("vision_gate requires --features cuda");
    std::process::ExitCode::FAILURE
}

#[cfg(feature = "cuda")]
fn main() -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    use w34_host::{
        cache::KvFormatPolicy, loader::SafetensorsArchive, model::Qwen3_5Engine, vision,
    };
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let trace_path = if let Some(index) = args.iter().position(|a| a == "--vision-trace") {
        if !(2..=3).contains(&index) || args.len() != index + 2 {
            return Err("--vision-trace PATH must follow 2-3 positional arguments".into());
        }
        let path = args[index + 1].clone();
        args.truncate(index);
        Some(path)
    } else {
        None
    };
    if !(2..=3).contains(&args.len()) {
        return Err("usage: vision_gate ORACLE_DIR OUTPUT.json [ACTUAL_VISUAL.json] [--vision-trace TRACE.json]".into());
    }
    let root = std::path::Path::new(&args[0]);
    let req: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("request.json"))?)?;
    let generation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("generation.json"))?)?;
    let model = req["model_resolved_snapshot"]
        .as_str()
        .ok_or("missing snapshot")?;
    let image_path = req["image"]["path"].as_str().ok_or("missing image path")?;
    let raw_prompt = req["raw_prompt"].as_str().ok_or("missing raw prompt")?;
    let expected_ids: Vec<i32> = serde_json::from_value(generation["token_ids"].clone())?;
    let expected_prompt: Vec<i32> = serde_json::from_value(req["prompt_token_ids"].clone())?;
    let hash = |p: &std::path::Path| -> Result<String, std::io::Error> {
        Ok(format!("{:x}", Sha256::digest(std::fs::read(p)?)))
    };
    if hash(std::path::Path::new(image_path))?
        != req["image"]["sha256"]
            .as_str()
            .ok_or("missing image digest")?
        || hash(&std::path::Path::new(model).join("config.json"))?
            != req["config_sha256"]
                .as_str()
                .ok_or("missing config digest")?
    {
        return Err("oracle input digest mismatch".into());
    }
    if req["vllm_version"] != "0.31.0"
        || req["engine_settings"]["dtype"] != "float16"
        || req["generation_settings"]["temperature"] != 0
        || req["generation_settings"]["ignore_eos"] != true
        || expected_ids.is_empty()
        || expected_ids.len() > 128
        || raw_prompt.matches("<|image_pad|>").count() != 1
    {
        return Err("unsupported oracle generation configuration".into());
    }
    let expected_processor = serde_json::json!({"min_pixels":1024,"max_pixels":1048576,"patch_size":16,"merge_size":2,"temporal_patch_size":2});
    if req["engine_settings"]["mm_processor_kwargs"] != expected_processor {
        return Err("oracle processor configuration differs from HTTP path".into());
    }
    let pages = (expected_prompt.len() + expected_ids.len() + 4).div_ceil(16);
    let mut engine = Qwen3_5Engine::load(model, 1, KvFormatPolicy::Fp16, Some(pages), true)?;
    let vc = engine
        .cfg
        .vision_config
        .as_ref()
        .ok_or("model has no vision configuration")?;
    if (vc.patch_size, vc.temporal_patch_size, vc.spatial_merge_size) != (16, 2, 2) {
        return Err("unsupported model vision geometry".into());
    }
    let (rgb, h, w) = vision::load_cli_image_with_policy(image_path, false)?;
    let image = vision::preprocess_rgb_image(&rgb, h, w, 16, 2, 2, 1024, 1048576);
    let expanded = raw_prompt.replace(
        "<|image_pad|>",
        &"<|image_pad|>".repeat(image.num_merged_tokens(2)),
    );
    let prompt = engine.encode(&expanded)?;
    if prompt != expected_prompt {
        return Err("processed prompt IDs differ from vLLM; refusing misaligned comparison".into());
    }
    let images = [image];
    let dump = engine.dump_multimodal_prompt(&prompt, &images)?;
    if let Some(path) = args.get(2) {
        let (_, visual) = dump
            .tensors_f32
            .get("visual_embeddings")
            .ok_or("missing visual dump")?;
        std::fs::write(
            path,
            serde_json::to_vec(
                &serde_json::json!({"visual_embeddings":visual,"patches_f32":images[0].patches_f32}),
            )?,
        )?;
    }
    let archive = SafetensorsArchive::open(&root.join("rank_0.safetensors"))?;
    if let Some(path) = trace_path {
        engine.ensure_vision_encoder()?;
        let mut trace = std::collections::BTreeMap::new();
        let rank = &engine.ranks[0];
        let _visual = engine
            .vision_encoder
            .as_ref()
            .ok_or("missing encoder")?
            .encode_images_with_trace(&images, &rank.cublas, &rank.stream, Some(&mut trace))?;
        let mut report = serde_json::Map::new();
        for (key, (shape, data)) in trace {
            let comparison = if archive.tensor_names().any(|name| name == key) {
                let reference = archive.read_f32(&key)?;
                if archive.tensor_info(&key)?.shape != shape || reference.len() != data.len() {
                    serde_json::json!({"status":"shape_mismatch"})
                } else if reference.iter().chain(&data).any(|v| !v.is_finite()) {
                    serde_json::json!({"status":"nonfinite"})
                } else {
                    let err: f64 = data
                        .iter()
                        .zip(&reference)
                        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                        .sum();
                    let denom: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
                    let max_abs = data
                        .iter()
                        .zip(&reference)
                        .map(|(&a, &b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    serde_json::json!({"status":"compared", "rel_l2":(err/denom.max(1e-30)).sqrt(), "max_abs":max_abs})
                }
            } else {
                serde_json::json!({"status":"native_key_missing"})
            };
            report.insert(
                key,
                serde_json::json!({"shape":shape,"data":data,"comparison":comparison}),
            );
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "diagnostic_only":true, "row_policy":"[0,L/2,L-1], duplicates retained", "tensors":report
            }))?,
        )?;
    }
    let is_sparse_moe = engine.cfg.num_experts > 1;
    let gate_mode = if is_sparse_moe {
        "sparse_moe_decomposed"
    } else {
        "dense_full_tensor"
    };
    let mut moe_frontend_limit = 0.05f64;
    let mut moe_layer0_limit = 0.05f64;
    let mut moe_envelope_limit = 0.05f64;
    let mut moe_repeat_range = serde_json::Value::Null;
    let moe_layer0 = if is_sparse_moe {
        let golden_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle/goldens");
        let meta_path = golden_dir.join("vllm_moe_layer0_m8.json");
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta_path)?)?;
        if meta["vllm_version"] != "0.31.0" {
            return Err("unsupported MoE layer-0 golden vLLM version".into());
        }
        let expected_cfg_sha = meta["config_sha256"]
            .as_str()
            .ok_or("missing MoE golden config_sha256")?;
        if hash(&std::path::Path::new(model).join("config.json"))? != expected_cfg_sha {
            return Err(
                "MoE checkpoint config.json does not match layer-0 golden provenance".into(),
            );
        }
        let st_name = meta["safetensors_file"]
            .as_str()
            .ok_or("missing MoE golden safetensors_file")?;
        let st_path = golden_dir.join(st_name);
        let expected_st_sha = meta["safetensors_sha256"]
            .as_str()
            .ok_or("missing MoE golden safetensors_sha256")?;
        if hash(&st_path)? != expected_st_sha {
            return Err("MoE layer-0 golden safetensors digest mismatch".into());
        }
        let parse_pos_limit = |v: &serde_json::Value, name: &str| -> Result<f64, String> {
            let x = v
                .as_f64()
                .ok_or_else(|| format!("missing MoE golden limit {name}"))?;
            if !(x.is_finite() && x > 0.0) {
                return Err(format!("non-positive MoE golden limit {name}"));
            }
            Ok(x)
        };
        moe_frontend_limit =
            parse_pos_limit(&meta["frontend_rel_l2_limit"], "frontend_rel_l2_limit")?;
        moe_layer0_limit = parse_pos_limit(&meta["layer0_rel_l2_limit"], "layer0_rel_l2_limit")?;
        let env_obj = &meta["native_self_repeat_final_norm_rel_l2"];
        moe_envelope_limit = parse_pos_limit(&env_obj["envelope_limit"], "envelope_limit")?;
        let lower_path = image_path.to_ascii_lowercase();
        moe_repeat_range = if lower_path.ends_with(".png") {
            env_obj["png_range"].clone()
        } else if lower_path.ends_with(".jpg") || lower_path.ends_with(".jpeg") {
            env_obj["jpg_range"].clone()
        } else {
            return Err("unsupported image extension for MoE repeatability range".into());
        };
        if !moe_repeat_range.is_array() {
            return Err("missing format repeatability range in MoE golden".into());
        }
        let moe_archive = SafetensorsArchive::open(&st_path)?;
        Some(engine.verify_moe_layer0_golden(&moe_archive, moe_layer0_limit)?)
    } else {
        None
    };

    let mut tensors = serde_json::Map::new();
    let mut tensors_passed = true;
    let mut keys = vec!["visual_embeddings", "embed", "final_norm"];
    keys.extend(
        archive
            .tensor_names()
            .filter(|name| name.starts_with("layer_") && name.ends_with("_hidden_after")),
    );
    for key in keys {
        let (shape, actual) = dump.tensors_f32.get(key).ok_or("missing model dump")?;
        let reference = archive.read_f32(key)?;
        if *shape != archive.tensor_info(key)?.shape
            || actual.len() != reference.len()
            || reference.is_empty()
            || actual.iter().chain(&reference).any(|x| !x.is_finite())
        {
            return Err(format!("invalid tensor dimensions/values: {key}").into());
        }
        let mut err_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&a, &b) in actual.iter().zip(&reference) {
            err_sq += (f64::from(a) - f64::from(b)).powi(2);
            ref_sq += f64::from(b).powi(2);
            max_abs = max_abs.max((a - b).abs());
        }
        let rel_l2 = (err_sq / ref_sq.max(1e-30)).sqrt();
        let limit = if is_sparse_moe {
            match key {
                "visual_embeddings" | "embed" => moe_frontend_limit,
                "layer_0_hidden_after" => moe_layer0_limit,
                "final_norm" => moe_envelope_limit,
                _ => 0.05,
            }
        } else {
            0.05
        };
        tensors_passed &= rel_l2 <= limit;
        let mut entry = serde_json::json!({
            "shape": shape,
            "max_abs": max_abs,
            "rel_l2": rel_l2,
            "limit": limit,
        });
        if is_sparse_moe && key == "final_norm" {
            entry["native_repeat_rel_l2_range"] = moe_repeat_range.clone();
        }
        tensors.insert(key.to_string(), entry);
    }
    let image_token_id = engine.cfg.vision_config.as_ref().unwrap().image_token_id;
    let (positions, _) =
        vision::compute_mrope_positions(&prompt, &[images[0].grid_thw], image_token_id, 2);
    if archive.tensor_info("mrope_positions")?.dtype != "I32"
        || archive.tensor_info("mrope_positions")?.shape != vec![3, prompt.len()]
    {
        return Err("invalid mrope tensor schema".into());
    }
    let mrope_exact = positions.concat() == archive.read_i32("mrope_positions")?;
    let step0_top1_exact = dump.greedy_token_id == expected_ids[0];
    let step0_logprob_diff = generation
        .get("steps")
        .and_then(|s| s.get(0))
        .and_then(|s0| s0.get("logprob"))
        .and_then(serde_json::Value::as_f64)
        .map(|exp_lp| (f64::from(dump.top5_logprobs[0]) - exp_lp).abs());
    let metrics = engine.generate_multimodal_with_radix_cache(
        &prompt,
        &images,
        expected_ids.len(),
        64,
        false,
    )?;
    let exact = metrics.generated_token_ids == expected_ids;
    let canonical_oracle = req
        .get("diagnostic_override_visual")
        .is_none_or(serde_json::Value::is_null);
    let moe_layer0_passed = moe_layer0.as_ref().is_none_or(|r| r.passed);
    let passed = exact
        && step0_top1_exact
        && tensors_passed
        && moe_layer0_passed
        && mrope_exact
        && canonical_oracle;
    let mut report = serde_json::json!({"passed":passed,"gate_mode":gate_mode,"model_snapshot":model,"image_sha256":req["image"]["sha256"],
        "prompt_ids_match":true,"prompt_tokens":prompt.len(),"tensors":tensors,"mrope_exact":mrope_exact,
        "step0_top1_exact":step0_top1_exact,"step0_top1_logprob_abs_diff":step0_logprob_diff,
        "greedy_exact":exact,"expected_ids":expected_ids,"actual_ids":metrics.generated_token_ids,
        "ttft_ms":metrics.ttft_ms,"decode_tok_per_s":metrics.decode_tok_per_s,
        "tp":1,"kv_policy":"fp16","prefill_chunk_size":64,"oracle_dir":root,"canonical_oracle":canonical_oracle});
    if let Some(moe_rep) = moe_layer0 {
        report["moe_layer0_same_input"] = serde_json::to_value(moe_rep)?;
    }
    std::fs::write(&args[1], serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(if passed {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    })
}

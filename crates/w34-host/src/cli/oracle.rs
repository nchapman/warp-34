use super::*;

#[cfg(any(test, feature = "cuda"))]
pub(crate) fn tensor_diff_metrics(a: &[f32], b: &[f32]) -> (f32, f64, f64) {
    let mut max_abs = 0.0f32;
    let mut err_sq = 0.0f64;
    let mut a_sq = 0.0f64;
    let mut b_sq = 0.0f64;
    let mut dot = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let xf = f64::from(x);
        let yf = f64::from(y);
        err_sq += (xf - yf) * (xf - yf);
        a_sq += xf * xf;
        b_sq += yf * yf;
        dot += xf * yf;
    }
    let rel_l2 = (err_sq / b_sq.max(1e-30)).sqrt();
    let denom = (a_sq * b_sq).sqrt();
    let cosine = if denom > 1e-30 {
        dot / denom
    } else if a_sq <= 1e-30 && b_sq <= 1e-30 {
        1.0
    } else {
        0.0
    };
    (max_abs, rel_l2, cosine)
}

#[cfg(any(test, feature = "cuda"))]
const GOLDEN_STRICT_MAX_REL_L2: f64 = 0.05;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_STRICT_MIN_LAST_COSINE: f64 = 0.998;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MM_MAX_FULL_REL_L2: f64 = 0.16;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MM_MIN_ROW_COSINE: f64 = 0.95;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_SUBLAYER_MAX_FULL_REL_L2: f64 = 0.55;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_SUBLAYER_MIN_FULL_COSINE: f64 = 0.85;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_RESIDUAL_MAX_FULL_REL_L2: f64 = 0.35;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_RESIDUAL_MIN_FULL_COSINE: f64 = 0.94;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_SUBLAYER_MAX_LAST_REL_L2: f64 = 0.30;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_SUBLAYER_MIN_LAST_COSINE: f64 = 0.95;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_MM_FINAL_MAX_LAST_REL_L2: f64 = 0.22;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_MM_FINAL_MIN_LAST_COSINE: f64 = 0.975;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_MM_LAYER_MAX_LAST_REL_L2: f64 = 0.30;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_MM_LAYER_MIN_LAST_COSINE: f64 = 0.955;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_TEXT_RESIDUAL_MAX_LAST_REL_L2: f64 = 0.06;
#[cfg(any(test, feature = "cuda"))]
const GOLDEN_MOE_TEXT_RESIDUAL_MIN_LAST_COSINE: f64 = 0.998;

#[cfg(any(test, feature = "cuda"))]
pub(crate) fn evaluate_golden_tensor_diff(
    name: &str,
    shape: &[usize],
    vals: &[f32],
    ref_vals: &[f32],
    is_multimodal: bool,
) -> Result<(f32, f64, f64), String> {
    if vals.len() != ref_vals.len() {
        return Err(format!(
            "dump-diff golden length mismatch on {name}: got {}, expected {}",
            vals.len(),
            ref_vals.len()
        ));
    }
    let expected_len = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("dump-diff golden shape overflow on {name}: {shape:?}"))?;
    if !shape.is_empty() && expected_len != vals.len() {
        return Err(format!(
            "dump-diff golden shape mismatch on {name}: shape {shape:?} implies {expected_len}, got {}",
            vals.len()
        ));
    }
    let (d, full_rel, full_cos) = tensor_diff_metrics(vals, ref_vals);
    let is_deep_mm_layer = is_multimodal
        && shape.len() == 2
        && shape[0] > 1
        && shape[1] > 0
        && (name == "final_norm" || (name.starts_with("layer_") && name != "layer_0_hidden_after"));
    if is_deep_mm_layer {
        if full_rel > GOLDEN_MM_MAX_FULL_REL_L2 {
            return Err(format!(
                "dump-diff multimodal full-sequence envelope failed on {name}: full_rel_l2={full_rel:.6}"
            ));
        }
        let cols = shape[1];
        for r in 0..shape[0] {
            let row_start = r * cols;
            let row_end = row_start + cols;
            let (_, row_rel, row_cos) =
                tensor_diff_metrics(&vals[row_start..row_end], &ref_vals[row_start..row_end]);
            if row_cos < GOLDEN_MM_MIN_ROW_COSINE {
                return Err(format!(
                    "dump-diff multimodal row {r} cosine failed on {name}: row_rel_l2={row_rel:.6} row_cosine={row_cos:.6}"
                ));
            }
        }
        let last_start = (shape[0] - 1) * cols;
        let (last_d, last_rel, last_cos) =
            tensor_diff_metrics(&vals[last_start..], &ref_vals[last_start..]);
        if last_rel > GOLDEN_STRICT_MAX_REL_L2 || last_cos < GOLDEN_STRICT_MIN_LAST_COSINE {
            return Err(format!(
                "dump-diff multimodal prompt-end check failed on {name}: last_rel_l2={last_rel:.6} last_cosine={last_cos:.6}"
            ));
        }
        Ok((d.max(last_d), last_rel, full_cos.min(last_cos)))
    } else {
        if full_rel > GOLDEN_STRICT_MAX_REL_L2 {
            return Err(format!(
                "dump-diff golden check failed on {name}: rel_l2={full_rel:.6} cosine={full_cos:.6}"
            ));
        }
        Ok((d, full_rel, full_cos))
    }
}

#[cfg(any(test, feature = "cuda"))]
pub(crate) fn evaluate_moe_or_golden_tensor_diff(
    name: &str,
    shape: &[usize],
    vals: &[f32],
    ref_vals: &[f32],
    is_multimodal: bool,
    is_moe: bool,
) -> Result<(f32, f64, f64), String> {
    let is_deep_post_router = is_moe
        && shape.len() == 2
        && shape[0] > 1
        && shape[1] > 0
        && (name == "final_norm" || (name.starts_with("layer_") && !name.starts_with("layer_0_")));
    if is_deep_post_router {
        if vals.len() != ref_vals.len() {
            return Err(format!(
                "dump-diff golden length mismatch on {name}: got {}, expected {}",
                vals.len(),
                ref_vals.len()
            ));
        }
        let expected_len = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("dump-diff golden shape overflow on {name}: {shape:?}"))?;
        if expected_len != vals.len() {
            return Err(format!(
                "dump-diff golden shape mismatch on {name}: shape {shape:?} implies {expected_len}, got {}",
                vals.len()
            ));
        }
        let (d, full_rel, full_cos) = tensor_diff_metrics(vals, ref_vals);
        let is_sublayer_delta = name.ends_with("_attn_or_gdn_out") || name.ends_with("_mlp_out");
        let (max_full_rel, min_full_cos) = if is_sublayer_delta {
            (
                GOLDEN_MOE_SUBLAYER_MAX_FULL_REL_L2,
                GOLDEN_MOE_SUBLAYER_MIN_FULL_COSINE,
            )
        } else {
            (
                GOLDEN_MOE_RESIDUAL_MAX_FULL_REL_L2,
                GOLDEN_MOE_RESIDUAL_MIN_FULL_COSINE,
            )
        };
        if full_rel > max_full_rel || full_cos < min_full_cos {
            return Err(format!(
                "dump-diff MoE full-sequence envelope failed on {name}: full_rel_l2={full_rel:.6} full_cosine={full_cos:.6}"
            ));
        }
        let cols = shape[1];
        let last_start = (shape[0] - 1) * cols;
        let (last_d, last_rel, last_cos) =
            tensor_diff_metrics(&vals[last_start..], &ref_vals[last_start..]);
        let (max_last_rel, min_last_cos) = if is_sublayer_delta {
            (
                GOLDEN_MOE_SUBLAYER_MAX_LAST_REL_L2,
                GOLDEN_MOE_SUBLAYER_MIN_LAST_COSINE,
            )
        } else if !is_multimodal {
            (
                GOLDEN_MOE_TEXT_RESIDUAL_MAX_LAST_REL_L2,
                GOLDEN_MOE_TEXT_RESIDUAL_MIN_LAST_COSINE,
            )
        } else if name == "final_norm" {
            (
                GOLDEN_MOE_MM_FINAL_MAX_LAST_REL_L2,
                GOLDEN_MOE_MM_FINAL_MIN_LAST_COSINE,
            )
        } else {
            (
                GOLDEN_MOE_MM_LAYER_MAX_LAST_REL_L2,
                GOLDEN_MOE_MM_LAYER_MIN_LAST_COSINE,
            )
        };
        if last_rel > max_last_rel || last_cos < min_last_cos {
            return Err(format!(
                "dump-diff MoE prompt-end check failed on {name}: last_rel_l2={last_rel:.6} last_cosine={last_cos:.6}"
            ));
        }
        Ok((d.max(last_d), last_rel, full_cos.min(last_cos)))
    } else {
        evaluate_golden_tensor_diff(name, shape, vals, ref_vals, is_multimodal)
    }
}

#[cfg(any(test, feature = "cuda"))]
pub(crate) struct Tp1VsTp2Summary {
    pub max_layer_diff: f32,
    pub target_max_rel_l2: f64,
    pub min_cosine: f64,
    pub drafter_rel_l2: f64,
    pub drafter_cosine: f64,
    pub compared_drafter_hidden: bool,
}

#[cfg(any(test, feature = "cuda"))]
pub(crate) fn compare_tp1_vs_tp2_tensors(
    tp1_tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    tp2_tensors: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    is_multimodal: bool,
    is_moe: bool,
) -> Result<Tp1VsTp2Summary, String> {
    let mut max_layer_diff = 0.0f32;
    let mut target_max_rel_l2 = 0.0f64;
    let mut min_cosine = 1.0f64;
    let mut drafter_rel_l2 = 0.0f64;
    let mut drafter_cosine = 1.0f64;
    let mut compared_drafter_hidden = false;
    let mut names: Vec<&String> = tp1_tensors.keys().collect();
    names.sort();
    for name in names {
        let (shape1, vals1) = &tp1_tensors[name];
        if matches!(name.as_str(), "logits" | "logits_last") {
            continue; // logits_buf on rank 0 is vocab-sharded under TP=2
        }
        let (shape2, vals2) = tp2_tensors
            .get(name)
            .ok_or_else(|| format!("dump-diff tp1-vs-tp2 missing tensor on TP=2: {name}"))?;
        if shape1 != shape2 || vals1.len() != vals2.len() {
            return Err(format!(
                "dump-diff tp1-vs-tp2 shape/length mismatch on {name}: tp1={shape1:?} (len={}), tp2={shape2:?} (len={})",
                vals1.len(),
                vals2.len()
            ));
        }
        if name == "drafter_hidden" {
            let (_, rel, cos) = tensor_diff_metrics(vals1, vals2);
            drafter_rel_l2 = rel;
            drafter_cosine = cos;
            compared_drafter_hidden = true;
        } else if is_multimodal
            && (name.ends_with("_attn_or_gdn_out") || name.ends_with("_mlp_out"))
        {
            continue;
        } else {
            let (mdiff, rel, cos) = evaluate_moe_or_golden_tensor_diff(
                name,
                shape1,
                vals1,
                vals2,
                is_multimodal,
                is_moe,
            )?;
            max_layer_diff = max_layer_diff.max(mdiff);
            target_max_rel_l2 = target_max_rel_l2.max(rel);
            min_cosine = min_cosine.min(cos);
        }
    }
    let mut tp2_names: Vec<&String> = tp2_tensors.keys().collect();
    tp2_names.sort();
    for name in tp2_names {
        if !matches!(name.as_str(), "logits" | "logits_last") && !tp1_tensors.contains_key(name) {
            return Err(format!(
                "dump-diff tp1-vs-tp2 missing tensor on TP=1: {name}"
            ));
        }
    }
    Ok(Tp1VsTp2Summary {
        max_layer_diff,
        target_max_rel_l2,
        min_cosine,
        drafter_rel_l2,
        drafter_cosine,
        compared_drafter_hidden,
    })
}

#[cfg(feature = "cuda")]
pub(crate) type GreedyCase = (String, Vec<i32>, Vec<i32>, Option<Vec<Vec<(i32, f64)>>>);

#[cfg(feature = "cuda")]
pub(crate) fn load_greedy_cases(
    engine: &Qwen3_5Engine,
    golden_arg: &str,
) -> Result<Vec<GreedyCase>, String> {
    let golden_p = Path::new(golden_arg);
    let mut cases = Vec::new();
    if golden_p.is_dir() {
        let prompts_path = golden_p
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("prompts.json"))
            .unwrap_or_else(|| Path::new("tools/oracle/prompts.json").to_path_buf());
        let prompts_str = fs::read_to_string(&prompts_path)
            .map_err(|e| format!("failed to read {}: {e}", prompts_path.display()))?;
        let prompts_json: serde_json::Value =
            serde_json::from_str(&prompts_str).map_err(|e| e.to_string())?;
        for group in ["chat", "agentic"] {
            if let Some(arr) = prompts_json.get(group).and_then(|v| v.as_array()) {
                for (idx, p_val) in arr.iter().enumerate() {
                    let tag = format!("{group}{idx:02}");
                    let ids_file = golden_p.join(format!("{tag}.ids"));
                    if !ids_file.exists() {
                        continue;
                    }
                    let prompt_text = p_val.as_str().unwrap_or("");
                    let prompt_ids = engine.encode(prompt_text).map_err(|e| e.to_string())?;
                    let ids_str = fs::read_to_string(&ids_file).map_err(|e| e.to_string())?;
                    let ref_gen_ids: Vec<i32> = ids_str
                        .split_whitespace()
                        .filter_map(|s| s.parse::<i32>().ok())
                        .collect();
                    let top5_file = golden_p.join(format!("{tag}.top5.json"));
                    let top5_opt = if top5_file.exists() {
                        let t_str = fs::read_to_string(&top5_file).map_err(|e| e.to_string())?;
                        let t_json: serde_json::Value =
                            serde_json::from_str(&t_str).map_err(|e| e.to_string())?;
                        t_json.as_array().map(|steps| {
                            steps
                                .iter()
                                .map(|step_arr| {
                                    step_arr
                                        .as_array()
                                        .map(|entries| {
                                            entries
                                                .iter()
                                                .filter_map(|pair| {
                                                    let p = pair.as_array()?;
                                                    Some((
                                                        p.first()?.as_i64()? as i32,
                                                        p.get(1)?.as_f64()?,
                                                    ))
                                                })
                                                .collect::<Vec<_>>()
                                        })
                                        .unwrap_or_default()
                                })
                                .collect::<Vec<_>>()
                        })
                    } else {
                        None
                    };
                    cases.push((tag, prompt_ids, ref_gen_ids, top5_opt));
                }
            }
        }
    } else {
        let golden_str = fs::read_to_string(golden_p)
            .map_err(|e| format!("failed to read {golden_arg}: {e}"))?;
        let golden_json: serde_json::Value =
            serde_json::from_str(&golden_str).map_err(|e| e.to_string())?;
        let prompts = golden_json["prompts"]
            .as_array()
            .ok_or("missing prompts array in golden json")?;
        for p in prompts {
            let id = p["id"].as_str().unwrap_or("unknown").to_string();
            let prompt_ids: Vec<i32> = p["prompt_token_ids"]
                .as_array()
                .ok_or("missing prompt_token_ids")?
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            let ref_gen_ids: Vec<i32> = p["generated_token_ids"]
                .as_array()
                .ok_or("missing generated_token_ids")?
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            let top5_opt = p.get("top5").and_then(|v| v.as_array()).map(|steps| {
                steps
                    .iter()
                    .map(|step_arr| {
                        step_arr
                            .as_array()
                            .map(|entries| {
                                entries
                                    .iter()
                                    .filter_map(|pair| {
                                        let p = pair.as_array()?;
                                        Some((p.first()?.as_i64()? as i32, p.get(1)?.as_f64()?))
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>()
            });
            cases.push((id, prompt_ids, ref_gen_ids, top5_opt));
        }
    }

    Ok(cases)
}

#[cfg(feature = "cuda")]
fn generate_radix_or_multimodal(
    engine: &mut Qwen3_5Engine,
    prompt_ids: &[i32],
    mm_images: &[crate::vision::PreprocessedImage],
    cmp_len: usize,
) -> Result<crate::model::RadixBenchMetrics, String> {
    if !mm_images.is_empty() {
        engine
            .generate_multimodal_with_radix_cache(prompt_ids, mm_images, cmp_len, 64, false)
            .map_err(|e| e.to_string())
    } else {
        engine
            .generate_with_radix_cache(prompt_ids, cmp_len, 64, false)
            .map_err(|e| e.to_string())
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn run_greedy_diff(
    options: CudaCliOptions,
    policy: KvFormatPolicy,
    spec_config: Option<crate::proposer::SpeculativeConfig>,
) -> Result<ExitCode, String> {
    let CudaCliOptions {
        model,
        resolved_model,
        tp,
        image_urls,
        allow_remote_images,
        golden_path,
        max_prompts,
        max_tokens,
        verify_lossless,
        verify_radix_warm,
        kv_format,
        kv_pages,
        use_cuda_graph,
        lora_modules,
        active_lora,
        ..
    } = options;
    let golden_arg = golden_path.ok_or("--golden <dir-or-json> is required")?;
    if !image_urls.is_empty() && (spec_config.is_some() || verify_lossless) {
        return Err(
            "greedy-diff with --image does not support --speculative or --verify-lossless".into(),
        );
    }
    let mut engine = load_cli_engine_with_lora(
        &resolved_model,
        tp,
        policy,
        kv_pages,
        use_cuda_graph,
        &spec_config,
        &lora_modules,
        active_lora.as_deref(),
    )
    .map_err(|e| e.to_string())?;
    let (_, mm_images) = if image_urls.is_empty() {
        (String::new(), Vec::new())
    } else {
        prepare_multimodal_prompt_and_images(&engine.cfg, "", &image_urls, allow_remote_images)?
    };
    let cases = load_greedy_cases(&engine, &golden_arg)?;

    let mut total_tokens_compared = 0usize;
    let mut total_matches = 0usize;
    let mut first_step_matches = 0usize;
    let mut adjudicated_passes = 0usize;
    let mut total_spec_rounds = 0usize;
    let mut total_spec_accepted = 0usize;
    let mut lossless_passes = 0usize;
    let mut radix_warm_passes = 0usize;
    let num_cases = cases.len().min(max_prompts);
    if num_cases == 0 {
        return Err("no golden cases found to compare".into());
    }

    for (id, prompt_ids, ref_gen_ids, top5_opt) in cases.iter().take(num_cases) {
        let cmp_len = ref_gen_ids.len().min(max_tokens);
        let plain_ref_ids = if verify_lossless && engine.speculative.is_some() {
            let saved_spec = engine.speculative.take();
            engine.reset_all_states().map_err(|e| e.to_string())?;
            let plain_ids = engine
                .generate_tokens(prompt_ids, cmp_len, false)
                .map_err(|e| e.to_string())?
                .generated_token_ids;
            engine.speculative = saved_spec;
            Some(plain_ids)
        } else {
            None
        };

        engine.reset_all_states().map_err(|e| e.to_string())?;
        let metrics = if !mm_images.is_empty() || verify_radix_warm {
            generate_radix_or_multimodal(&mut engine, prompt_ids, &mm_images, cmp_len).map(|m| {
                crate::model::BenchMetrics {
                    prompt_len: m.prompt_len,
                    gen_len: m.gen_len,
                    ttft_ms: m.ttft_ms,
                    prefill_tok_per_s: m.prefill_tok_per_s,
                    decode_ms: m.decode_ms,
                    decode_tok_per_s: m.decode_tok_per_s,
                    itl_ms: m.itl_ms,
                    generated_token_ids: m.generated_token_ids,
                }
            })?
        } else {
            engine
                .generate_tokens(prompt_ids, cmp_len, false)
                .map_err(|e| e.to_string())?
        };

        let mut spec_info = String::new();
        if let Some(proposer) = &engine.speculative {
            let r = proposer.last_accepted.len();
            let a: usize = proposer.last_accepted.iter().copied().sum();
            total_spec_rounds += r;
            total_spec_accepted += a;
            let tau = if r > 0 {
                1.0 + (a as f64 / r as f64)
            } else {
                1.0
            };
            spec_info = format!(" [spec: r={r} acc={a} tau={tau:.2}x]");
        }

        if let Some(ref plain_ids) = plain_ref_ids {
            if &metrics.generated_token_ids == plain_ids {
                lossless_passes += 1;
            } else {
                return Err(format!(
                    "greedy-diff --verify-lossless failed on case {id}: speculative output diverged from plain target greedy"
                ));
            }
        } else if verify_lossless {
            lossless_passes += 1;
        }

        if verify_radix_warm {
            let warm = generate_radix_or_multimodal(&mut engine, prompt_ids, &mm_images, cmp_len)?;
            if warm.generated_token_ids == metrics.generated_token_ids {
                radix_warm_passes += 1;
                spec_info.push_str(&format!(
                    " [radix: cached={} ttft={:.1}->{:.1}ms]",
                    warm.cached_prefix_tokens, metrics.ttft_ms, warm.ttft_ms
                ));
            } else {
                return Err(format!(
                    "greedy-diff --verify-radix-warm failed on case {id}: warm prefix-hit output diverged from cold output"
                ));
            }
        }

        let mut prefix_match = 0usize;
        let mut exact_matches = 0usize;
        for (k, (&got, &want)) in metrics
            .generated_token_ids
            .iter()
            .zip(ref_gen_ids.iter())
            .enumerate()
            .take(cmp_len)
        {
            if got == want {
                exact_matches += 1;
                if prefix_match == k {
                    prefix_match += 1;
                }
            }
        }
        if cmp_len > 0 && metrics.generated_token_ids[0] == ref_gen_ids[0] {
            first_step_matches += 1;
        }
        let mut flip_info = String::from("exact");
        let mut is_benign = prefix_match == cmp_len;
        if prefix_match < cmp_len {
            let got_tok = metrics.generated_token_ids[prefix_match];
            let want_tok = ref_gen_ids[prefix_match];
            if let Some(top5) = top5_opt
                && let Some(step_top5) = top5.get(prefix_match)
            {
                let rank_pos = step_top5.iter().position(|&(tid, _)| tid == got_tok);
                if let Some(r_idx) = rank_pos {
                    let gap = step_top5[0].1 - step_top5[r_idx].1;
                    let max_runner_up_gap = match parse_kv_policy(&kv_format) {
                        Ok(KvFormatPolicy::Fp16 | KvFormatPolicy::Bf16) => 0.60,
                        _ => 0.75,
                    };
                    is_benign = (r_idx == 1 && gap < max_runner_up_gap)
                        || (prefix_match >= 64 && r_idx <= 2);
                    flip_info = format!(
                        "seat={prefix_match} got={got_tok}(rank#{}) want={want_tok} gap={gap:.4}",
                        r_idx + 1
                    );
                } else {
                    flip_info = format!("seat={prefix_match} got={got_tok}(>top5) want={want_tok}");
                }
            } else {
                flip_info = format!("seat={prefix_match} got={got_tok} want={want_tok}");
            }
        }
        if is_benign {
            adjudicated_passes += 1;
        }
        total_tokens_compared += cmp_len;
        total_matches += exact_matches;
        println!(
            "[greedy-diff] case={id:<10} first_tok={} prefix={prefix_match:>3}/{cmp_len} exact={exact_matches:>3}/{cmp_len} decode={:>5.1}tok/s [{flip_info}]{spec_info}",
            metrics.generated_token_ids.first() == ref_gen_ids.first(),
            metrics.decode_tok_per_s
        );
    }
    let spec_summary = if total_spec_rounds > 0 {
        let tau = 1.0 + (total_spec_accepted as f64 / total_spec_rounds as f64);
        format!(
            " spec_rounds={total_spec_rounds} spec_accepted={total_spec_accepted} mean_tau={tau:.2}x"
        )
    } else {
        String::new()
    };
    let lossless_summary = if verify_lossless {
        format!(" lossless={lossless_passes}/{num_cases}")
    } else {
        String::new()
    };
    let radix_summary = if verify_radix_warm {
        format!(" radix_warm={radix_warm_passes}/{num_cases}")
    } else {
        String::new()
    };
    println!(
        "[greedy-diff summary] model={model} tp={tp} kv={kv_format} first_step={first_step_matches}/{num_cases} adjudicated={adjudicated_passes}/{num_cases} total_token_match={total_matches}/{total_tokens_compared} ({:.1}%){spec_summary}{lossless_summary}{radix_summary}",
        if total_tokens_compared > 0 {
            100.0 * (total_matches as f64) / (total_tokens_compared as f64)
        } else {
            100.0
        }
    );
    if adjudicated_passes < num_cases {
        return Err(format!(
            "greedy-diff failed: only {adjudicated_passes}/{num_cases} prompts passed adjudication"
        ));
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_dump_diff(
    options: CudaCliOptions,
    policy: KvFormatPolicy,
    spec_config: Option<crate::proposer::SpeculativeConfig>,
) -> Result<ExitCode, String> {
    let CudaCliOptions {
        model,
        resolved_model,
        tp,
        prompt,
        image_urls,
        allow_remote_images,
        kv_format,
        compare_tp2,
        golden_path,
        kv_pages,
        use_cuda_graph,
        lora_modules,
        active_lora,
        ..
    } = options;
    let mut engine = load_cli_engine_with_lora(
        &resolved_model,
        tp,
        policy,
        kv_pages,
        use_cuda_graph,
        &spec_config,
        &lora_modules,
        active_lora.as_deref(),
    )
    .map_err(|e| e.to_string())?;
    let (expanded_prompt, mm_images) = prepare_multimodal_prompt_and_images(
        &engine.cfg,
        &prompt,
        &image_urls,
        allow_remote_images,
    )?;
    let prompt_ids = engine.encode(&expanded_prompt).map_err(|e| e.to_string())?;
    let dump = if mm_images.is_empty() {
        engine.dump_prompt(&prompt_ids).map_err(|e| e.to_string())?
    } else {
        engine
            .dump_multimodal_prompt(&prompt_ids, &mm_images)
            .map_err(|e| e.to_string())?
    };
    let decoded_top1 = engine
        .decode(&[dump.greedy_token_id])
        .unwrap_or_else(|_| "<err>".to_string());
    let draft_str = if !dump.draft_proposal_ids.is_empty() {
        format!(" draft_proposals={:?}", dump.draft_proposal_ids)
    } else {
        String::new()
    };
    let img_str = if mm_images.is_empty() {
        String::new()
    } else {
        format!(" images={}", mm_images.len())
    };
    println!(
        "[dump-diff] model={model} tp={tp} kv={kv_format}{img_str} prompt_len={} tensors_captured={} greedy_id={} ({:?}) top5_ids={:?} top5_logprobs={:?}{draft_str}",
        prompt_ids.len(),
        dump.tensors_f32.len(),
        dump.greedy_token_id,
        decoded_top1,
        dump.top5_ids,
        dump.top5_logprobs
    );
    let is_multimodal = !mm_images.is_empty();
    let is_moe = engine.cfg.num_experts > 0;
    let max_allowed_rel_l2 = if is_moe {
        GOLDEN_MOE_SUBLAYER_MAX_LAST_REL_L2
    } else {
        GOLDEN_STRICT_MAX_REL_L2
    };
    if compare_tp2 {
        drop(engine);
        let pages = kv_pages.or_else(|| spec_config.as_ref().map(|_| 256));
        let mut engine_tp2 = Qwen3_5Engine::load(&resolved_model, 2, policy, pages, false)
            .map_err(|e| e.to_string())?;
        attach_cli_lora(&mut engine_tp2, &lora_modules, active_lora.as_deref(), true)
            .map_err(|e| e.to_string())?;
        if let Some(config) = &spec_config {
            engine_tp2
                .enable_speculative(config)
                .map_err(|e| e.to_string())?;
        }
        let dump_tp2 = if mm_images.is_empty() {
            engine_tp2
                .dump_prompt(&prompt_ids)
                .map_err(|e| e.to_string())?
        } else {
            engine_tp2
                .dump_multimodal_prompt(&prompt_ids, &mm_images)
                .map_err(|e| e.to_string())?
        };
        let Tp1VsTp2Summary {
            max_layer_diff,
            target_max_rel_l2,
            min_cosine,
            drafter_rel_l2,
            drafter_cosine,
            compared_drafter_hidden,
        } = compare_tp1_vs_tp2_tensors(
            &dump.tensors_f32,
            &dump_tp2.tensors_f32,
            is_multimodal,
            is_moe,
        )?;
        let top1_match = dump.greedy_token_id == dump_tp2.greedy_token_id;
        let draft_prefix_match = dump
            .draft_proposal_ids
            .iter()
            .zip(dump_tp2.draft_proposal_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let draft_exact_match = dump
            .draft_proposal_ids
            .iter()
            .zip(dump_tp2.draft_proposal_ids.iter())
            .filter(|(a, b)| a == b)
            .count();
        let draft_len = dump.draft_proposal_ids.len();
        let draft_ok = if spec_config.is_some() {
            compared_drafter_hidden
                && draft_len > 0
                && dump_tp2.draft_proposal_ids.len() == draft_len
                && draft_prefix_match >= 1
                && drafter_cosine >= 0.95
        } else {
            true
        };
        let spec_cmp = if spec_config.is_some() {
            format!(
                " drafter_rel_l2={drafter_rel_l2:.6} drafter_cosine={drafter_cosine:.6} draft_prefix={draft_prefix_match}/{draft_len} draft_exact={draft_exact_match}/{draft_len} tp1_drafts={:?} tp2_drafts={:?}",
                dump.draft_proposal_ids, dump_tp2.draft_proposal_ids
            )
        } else {
            String::new()
        };
        println!(
            "[dump-diff tp1-vs-tp2] model={model} max_hidden_diff={max_layer_diff:.6} target_max_rel_l2={target_max_rel_l2:.6} min_cosine={min_cosine:.6} tp1_top1={} tp2_top1={} match={top1_match}{spec_cmp}",
            dump.greedy_token_id, dump_tp2.greedy_token_id,
        );
        if !top1_match || !draft_ok || target_max_rel_l2 > max_allowed_rel_l2 {
            return Err(format!(
                "dump-diff tp1-vs-tp2 parity check failed: top1_match={top1_match} draft_ok={draft_ok} target_max_rel_l2={target_max_rel_l2:.6}"
            ));
        }
    }
    if let Some(gpath) = golden_path {
        let archive = crate::loader::SafetensorsArchive::open(Path::new(&gpath))
            .map_err(|e| e.to_string())?;
        let mut compared = 0usize;
        let mut skipped = 0usize;
        let mut max_diff = 0.0f32;
        let mut max_rel_l2 = 0.0f64;
        let mut min_cosine = 1.0f64;
        let mut names: Vec<&String> = dump.tensors_f32.keys().collect();
        names.sort();
        for name in names {
            let (shape, vals) = &dump.tensors_f32[name];
            if let Some(resolved) = archive.resolve_name(name) {
                let ref_vals = archive.read_f32(&resolved).map_err(|e| e.to_string())?;
                let (d, rel, cos) = evaluate_moe_or_golden_tensor_diff(
                    name,
                    shape,
                    vals,
                    &ref_vals,
                    is_multimodal,
                    is_moe,
                )?;
                max_diff = max_diff.max(d);
                max_rel_l2 = max_rel_l2.max(rel);
                min_cosine = min_cosine.min(cos);
                compared += 1;
            } else {
                skipped += 1;
            }
        }
        println!(
            "[dump-diff golden] golden={gpath} compared_tensors={compared} skipped_tensors={skipped} max_abs_diff={max_diff:.6} max_rel_l2={max_rel_l2:.6} min_cosine={min_cosine:.6}"
        );
        if compared == 0 || max_rel_l2 > max_allowed_rel_l2 {
            return Err(format!(
                "dump-diff golden check failed: compared_tensors={compared} max_rel_l2={max_rel_l2:.6}"
            ));
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "cuda")]
pub(crate) fn profile_bench_args(options: &CudaCliOptions, exe: &Path) -> Vec<String> {
    let mut args = vec![
        exe.display().to_string(),
        "bench".to_string(),
        "--model".to_string(),
        options.model.clone(),
        "--tp".to_string(),
        options.tp.to_string(),
        "--prompt-len".to_string(),
        options.prompt_len.to_string(),
        "--gen-len".to_string(),
        options.gen_len.to_string(),
        "--kv-format".to_string(),
        options.kv_format.clone(),
    ];
    if let Some(kind) = options.speculative_kind {
        args.push("--speculative".to_string());
        match kind {
            crate::proposer::SpeculativeKind::Mtp => {
                args.push("mtp".to_string());
                args.push("--mtp-k".to_string());
                args.push(options.mtp_k.to_string());
            }
            crate::proposer::SpeculativeKind::DFlash2 => {
                args.push("dflash2".to_string());
            }
        }
    }
    for (flag, value) in [
        ("--draft-model", &options.draft_model),
        ("--revision", &options.revision),
        ("--draft-revision", &options.draft_revision),
    ] {
        if let Some(value) = value {
            args.push(flag.to_string());
            args.push(value.clone());
        }
    }
    if options.offline {
        args.push("--offline".to_string());
    }
    if let Some(pages) = options.kv_pages {
        args.push("--kv-pages".to_string());
        args.push(pages.to_string());
    }
    if !options.use_cuda_graph {
        args.push("--no-cuda-graph".to_string());
    }
    if options.bench_stage != "all" {
        args.push("--stage".to_string());
        args.push(options.bench_stage.clone());
    }
    if !options.lora_modules.is_empty() {
        args.push("--lora-modules".to_string());
        for spec in &options.lora_modules {
            let rev_suffix = spec
                .revision
                .as_deref()
                .map(|r| format!("@{r}"))
                .unwrap_or_default();
            args.push(format!("{}={}{rev_suffix}", spec.name, spec.path));
        }
    }
    if let Some(ref active) = options.active_lora {
        args.push("--active-lora".to_string());
        args.push(active.clone());
    }
    args
}

#[cfg(feature = "cuda")]
pub(crate) fn run_profile_diff(options: CudaCliOptions) -> Result<ExitCode, String> {
    let CudaCliOptions {
        ref oracle_path,
        ref preset_key,
        ref w34_prof_path,
        prof_rounds,
        live_profile,
        ref out_json,
        ..
    } = options;
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let resolve_repo_file = |rel: &str| -> String {
        let p = Path::new(rel);
        if p.exists() {
            rel.to_string()
        } else {
            repo_root.join(rel).display().to_string()
        }
    };
    let oracle = oracle_path
        .clone()
        .unwrap_or_else(|| resolve_repo_file("tools/oracle/goldens/oracle_baselines.json"));
    let script = resolve_repo_file("tools/oracle/profile_diff.py");
    let mut cmd = std::process::Command::new("python3");
    cmd.arg("-B")
        .arg(&script)
        .arg("--oracle")
        .arg(&oracle)
        .arg("--rounds")
        .arg(prof_rounds.to_string());
    if let Some(p) = preset_key {
        cmd.arg("--preset").arg(p);
    }
    if let Some(w) = w34_prof_path {
        cmd.arg("--w34").arg(w);
    } else if live_profile {
        let exe = env::current_exe().map_err(|e| e.to_string())?;
        let bench_args = profile_bench_args(&options, &exe);
        cmd.arg("--live-cmd")
            .arg(serde_json::to_string(&bench_args).unwrap());
    }
    if let Some(o) = out_json {
        cmd.arg("--json-out").arg(o);
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to run profile_diff.py: {e}"))?;
    if status.success() {
        Ok(ExitCode::SUCCESS)
    } else {
        Err("profile_diff.py exited with non-zero status".to_string())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::compare_tp1_vs_tp2_tensors;
    use std::collections::HashMap;

    #[test]
    fn compare_tp1_vs_tp2_skips_vocab_sharded_logits_and_logits_last() {
        let mut tp1 = HashMap::new();
        tp1.insert(
            "final_norm".to_string(),
            (vec![1usize, 4usize], vec![1.0f32, 2.0, 3.0, 4.0]),
        );
        tp1.insert("logits".to_string(), (vec![1, 8], vec![0.1; 8]));
        tp1.insert("logits_last".to_string(), (vec![8], vec![0.1; 8]));

        let mut tp2 = HashMap::new();
        tp2.insert(
            "final_norm".to_string(),
            (vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]),
        );
        tp2.insert("logits".to_string(), (vec![1, 4], vec![0.1; 4]));
        tp2.insert("logits_last".to_string(), (vec![4], vec![0.1; 4]));

        let summary = compare_tp1_vs_tp2_tensors(&tp1, &tp2, false, false).unwrap();
        assert_eq!(summary.max_layer_diff, 0.0);
        assert_eq!(summary.target_max_rel_l2, 0.0);
    }
}

use super::*;

#[cfg(feature = "cuda")]
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
            cases.push((id, prompt_ids, ref_gen_ids, None));
        }
    }

    Ok(cases)
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
        golden_path,
        max_prompts,
        max_tokens,
        verify_lossless,
        verify_radix_warm,
        kv_format,
        kv_pages,
        use_cuda_graph,
        ..
    } = options;
    let golden_arg = golden_path.ok_or("--golden <dir-or-json> is required")?;
    let mut engine = load_cli_engine(
        &resolved_model,
        tp,
        policy,
        kv_pages,
        use_cuda_graph,
        &spec_config,
    )
    .map_err(|e| e.to_string())?;
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
            let plain_m = engine
                .generate_tokens(prompt_ids, cmp_len, false)
                .map_err(|e| e.to_string())?;
            engine.speculative = saved_spec;
            Some(plain_m.generated_token_ids)
        } else {
            None
        };

        engine.reset_all_states().map_err(|e| e.to_string())?;
        let metrics = if verify_radix_warm {
            engine
                .generate_with_radix_cache(prompt_ids, cmp_len, 64, false)
                .map(|m| crate::model::BenchMetrics {
                    prompt_len: m.prompt_len,
                    gen_len: m.gen_len,
                    ttft_ms: m.ttft_ms,
                    prefill_tok_per_s: m.prefill_tok_per_s,
                    decode_ms: m.decode_ms,
                    decode_tok_per_s: m.decode_tok_per_s,
                    itl_ms: m.itl_ms,
                    generated_token_ids: m.generated_token_ids,
                })
                .map_err(|e| e.to_string())?
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
            let warm = engine
                .generate_with_radix_cache(prompt_ids, cmp_len, 64, false)
                .map_err(|e| e.to_string())?;
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
        ..
    } = options;
    let mut engine = load_cli_engine(
        &resolved_model,
        tp,
        policy,
        kv_pages,
        use_cuda_graph,
        &spec_config,
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
    if compare_tp2 {
        drop(engine);
        let pages = kv_pages.or_else(|| spec_config.as_ref().map(|_| 256));
        let mut engine_tp2 = Qwen3_5Engine::load(&resolved_model, 2, policy, pages, false)
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
        let mut max_layer_diff = 0.0f32;
        let mut target_max_rel_l2 = 0.0f64;
        let mut min_cosine = 1.0f64;
        let mut drafter_rel_l2 = 0.0f64;
        let mut drafter_cosine = 1.0f64;
        let mut compared_drafter_hidden = false;
        for (name, (shape1, vals1)) in &dump.tensors_f32 {
            if name == "logits" {
                continue; // logits_buf on rank 0 is vocab-sharded under TP=2
            }
            if let Some((shape2, vals2)) = dump_tp2.tensors_f32.get(name)
                && shape1 == shape2
            {
                let (mdiff, rel, cos) = tensor_diff_metrics(vals1, vals2);
                if name == "drafter_hidden" {
                    drafter_rel_l2 = rel;
                    drafter_cosine = cos;
                    compared_drafter_hidden = true;
                } else {
                    max_layer_diff = max_layer_diff.max(mdiff);
                    target_max_rel_l2 = target_max_rel_l2.max(rel);
                    min_cosine = min_cosine.min(cos);
                }
            }
        }
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
        if !top1_match || !draft_ok || target_max_rel_l2 > 0.05 {
            return Err(format!(
                "dump-diff tp1-vs-tp2 parity check failed: top1_match={top1_match} draft_ok={draft_ok} target_max_rel_l2={target_max_rel_l2:.6}"
            ));
        }
    }
    if let Some(gpath) = golden_path {
        let archive = crate::loader::SafetensorsArchive::open(Path::new(&gpath))
            .map_err(|e| e.to_string())?;
        let mut compared = 0usize;
        let mut max_diff = 0.0f32;
        let mut max_rel_l2 = 0.0f64;
        let mut min_cosine = 1.0f64;
        for (name, (_shape, vals)) in &dump.tensors_f32 {
            if let Some(resolved) = archive.resolve_name(name) {
                let ref_vals = archive.read_f32(&resolved).map_err(|e| e.to_string())?;
                if ref_vals.len() == vals.len() {
                    let (d, rel, cos) = tensor_diff_metrics(vals, &ref_vals);
                    max_diff = max_diff.max(d);
                    max_rel_l2 = max_rel_l2.max(rel);
                    min_cosine = min_cosine.min(cos);
                    compared += 1;
                }
            }
        }
        println!(
            "[dump-diff golden] golden={gpath} compared_tensors={compared} max_abs_diff={max_diff:.6} max_rel_l2={max_rel_l2:.6} min_cosine={min_cosine:.6}"
        );
        if compared == 0 || max_rel_l2 > 0.05 {
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

use super::*;

#[cfg(feature = "cuda")]
pub(crate) fn prepare_multimodal_prompt_and_images(
    cfg: &w34_host::loader::ModelConfig,
    raw_prompt: &str,
    image_urls: &[String],
    allow_remote_images: bool,
) -> Result<(String, Vec<w34_host::vision::PreprocessedImage>), String> {
    if image_urls.is_empty() {
        return Ok((raw_prompt.to_string(), Vec::new()));
    }
    let (patch_size, temporal_patch_size, merge_size) = cfg
        .vision_config
        .as_ref()
        .map(|v| (v.patch_size, v.temporal_patch_size, v.spatial_merge_size))
        .unwrap_or((16, 2, 2));
    let mut images = Vec::with_capacity(image_urls.len());
    let mut prefix = String::new();
    for url in image_urls {
        let (rgb, h, w) = w34_host::vision::load_cli_image_with_policy(url, allow_remote_images)
            .map_err(|e| e.to_string())?;
        w34_host::vision::validate_image_aspect_ratio(h, w)?;
        let pre = w34_host::vision::preprocess_rgb_image(
            &rgb,
            h,
            w,
            patch_size,
            temporal_patch_size,
            merge_size,
            64 * 32 * 32,
            2048 * 32 * 32,
        );
        if !raw_prompt.contains("<|image_pad|>") {
            let n_pads = pre.num_merged_tokens(merge_size);
            prefix.push_str("<|vision_start|>");
            for _ in 0..n_pads {
                prefix.push_str("<|image_pad|>");
            }
            prefix.push_str("<|vision_end|>");
        }
        images.push(pre);
    }
    let expanded = if prefix.is_empty() {
        raw_prompt.to_string()
    } else {
        format!("{prefix}{raw_prompt}")
    };
    Ok((expanded, images))
}

#[cfg(feature = "cuda")]
pub(crate) fn run_cuda_cli(args: &[String]) -> Result<ExitCode, String> {
    let Some(PreparedCudaCli {
        options,
        policy,
        spec_config,
    }) = prepare_cuda_cli(args, w34_host::hub::resolve_model)?
    else {
        return Ok(ExitCode::SUCCESS);
    };
    match args[0].as_str() {
        "generate" => run_generate(options, policy, spec_config),
        "bench" => run_bench(options, policy, spec_config),
        "greedy-diff" => run_greedy_diff(options, policy, spec_config),
        "dump-diff" => run_dump_diff(options, policy, spec_config),
        "profile-diff" => run_profile_diff(options),
        "serve" => run_serve(options, policy, spec_config),
        "chat" => run_chat(options, policy, spec_config),
        other => Err(format!("unknown subcommand: {other}")),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn run_generate(
    options: CudaCliOptions,
    policy: KvFormatPolicy,
    spec_config: Option<w34_host::proposer::SpeculativeConfig>,
) -> Result<ExitCode, String> {
    let CudaCliOptions {
        model,
        resolved_model,
        tp,
        prompt,
        image_urls,
        allow_remote_images,
        max_tokens,
        kv_format,
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
    let (gen_ids, prompt_len, gen_len, ttft_ms, decode_tok_per_s, itl_ms) = if mm_images.is_empty()
    {
        let m = engine
            .generate_tokens(&prompt_ids, max_tokens, true)
            .map_err(|e| e.to_string())?;
        (
            m.generated_token_ids,
            m.prompt_len,
            m.gen_len,
            m.ttft_ms,
            m.decode_tok_per_s,
            m.itl_ms,
        )
    } else {
        let m = engine
            .generate_multimodal_with_radix_cache(&prompt_ids, &mm_images, max_tokens, 64, true)
            .map_err(|e| e.to_string())?;
        (
            m.generated_token_ids,
            m.prompt_len,
            m.gen_len,
            m.ttft_ms,
            m.decode_tok_per_s,
            m.itl_ms,
        )
    };
    let text = engine.decode(&gen_ids).map_err(|e| e.to_string())?;
    println!("{text}");
    let img_tag = if mm_images.is_empty() {
        String::new()
    } else {
        format!(" images={}", mm_images.len())
    };
    eprintln!(
        "[w34 generate] model={model} tp={tp} kv={kv_format}{img_tag} prompt_toks={} gen_toks={} ttft={:.2}ms decode={:.2}tok/s ({:.2}ms/tok)",
        prompt_len, gen_len, ttft_ms, decode_tok_per_s, itl_ms
    );
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_bench(
    options: CudaCliOptions,
    policy: KvFormatPolicy,
    spec_config: Option<w34_host::proposer::SpeculativeConfig>,
) -> Result<ExitCode, String> {
    let CudaCliOptions {
        model,
        resolved_model,
        tp,
        prompt_len,
        gen_len,
        kv_format,
        speculative_kind,
        bench_stage,
        batch_size,
        image_urls,
        allow_remote_images,
        out_json,
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
    let base_ids = engine
        .encode("The quick brown fox jumps over the lazy dog. ")
        .map_err(|e| e.to_string())?;
    let make_prompt = |len: usize| -> Vec<i32> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            for &id in &base_ids {
                if out.len() < len {
                    out.push(id);
                }
            }
        }
        out
    };
    let synth_prompt = make_prompt(prompt_len);
    // Warmup pass (captures CUDA Graph on decode step 1 and ramps GPU boost clock out of P8)
    let _ = engine
        .generate_tokens(&synth_prompt[..16.min(prompt_len)], 32, false)
        .map_err(|e| e.to_string())?;

    let spec_label = speculative_kind.map(|k| match k {
        w34_host::proposer::SpeculativeKind::Mtp => "mtp",
        w34_host::proposer::SpeculativeKind::DFlash2 => "dflash2",
    });

    let mut decode_summary = serde_json::Value::Null;
    if matches!(bench_stage.as_str(), "decode" | "all") {
        engine.reset_all_states().map_err(|e| e.to_string())?;
        let metrics = {
            let _measured = w34_host::profiling::Range::new(c"w34.bench.measured");
            engine
                .generate_tokens(&synth_prompt, gen_len, false)
                .map_err(|e| e.to_string())?
        };
        let accepted = engine.speculative.as_ref().map(|p| p.last_accepted.clone());
        let verify_rounds = accepted.as_ref().map(Vec::len);
        let total_accepted: usize = accepted
            .as_ref()
            .map(|v| v.iter().copied().sum())
            .unwrap_or(0);
        let mean_accepted_drafts = verify_rounds.and_then(|r| {
            if r > 0 {
                Some(total_accepted as f64 / r as f64)
            } else {
                None
            }
        });
        let mean_tokens_per_round = mean_accepted_drafts.map(|a| 1.0 + a);
        let ms_per_verify_round = verify_rounds.and_then(|r| {
            if r > 0 {
                Some(metrics.decode_ms / r as f64)
            } else {
                None
            }
        });
        let num_fa_layers = engine.ranks[0].kv_pool.layers.len();
        let kv_bytes_per_token = policy.bytes_per_token(
            num_fa_layers,
            engine.cfg.num_key_value_heads,
            engine.cfg.head_dim,
        );
        decode_summary = serde_json::json!({
            "model": model,
            "tp": tp,
            "kv_format": kv_format,
            "kv_bytes_per_token": kv_bytes_per_token,
            "num_experts": engine.cfg.num_experts,
            "num_experts_per_tok": engine.cfg.num_experts_per_tok,
            "cuda_graph": engine.use_cuda_graph,
            "speculative": spec_label,
            "accepted_drafts_per_round": accepted,
            "verify_rounds": verify_rounds,
            "measured_rounds": verify_rounds.unwrap_or(metrics.gen_len.saturating_sub(1)),
            "prompt_token_ids": synth_prompt,
            "generated_token_ids": metrics.generated_token_ids,
            "mean_accepted_drafts": mean_accepted_drafts,
            "mean_tokens_per_round": mean_tokens_per_round,
            "ms_per_verify_round": ms_per_verify_round,
            "prompt_len": metrics.prompt_len,
            "gen_len": metrics.gen_len,
            "ttft_ms": metrics.ttft_ms,
            "prefill_tok_per_s": metrics.prefill_tok_per_s,
            "decode_ms": metrics.decode_ms,
            "decode_tok_per_s": metrics.decode_tok_per_s,
            "itl_ms": metrics.itl_ms,
        });
    }

    let mut radix_summary = serde_json::Value::Null;
    if matches!(bench_stage.as_str(), "radix" | "all") {
        engine.reset_all_states().map_err(|e| e.to_string())?;
        let r_prompt_len = (prompt_len.max(128) / 64) * 64 + 8;
        let r_prompt = make_prompt(r_prompt_len);
        let r_gen = gen_len.clamp(4, 32);
        let cold = engine
            .generate_with_radix_cache(&r_prompt, r_gen, 64, false)
            .map_err(|e| e.to_string())?;
        let warm = engine
            .generate_with_radix_cache(&r_prompt, r_gen, 64, false)
            .map_err(|e| e.to_string())?;
        let lossless_parity = cold.generated_token_ids == warm.generated_token_ids;
        if !lossless_parity {
            return Err(
                "radix benchmark: warm prefix-hit output diverged from cold reference".into(),
            );
        }
        if warm.cached_prefix_tokens < 64 {
            return Err(format!(
                "radix benchmark: expected >= 64 cached prefix tokens on warm hit, got {}",
                warm.cached_prefix_tokens
            ));
        }
        let restored_proposer = engine.speculative.as_ref().map(|p| p.last_restored_tokens);
        let ttft_reduction_pct = if cold.ttft_ms > 0.0 {
            100.0 * (1.0 - warm.ttft_ms / cold.ttft_ms)
        } else {
            0.0
        };
        let ttft_speedup_x = if warm.ttft_ms > 0.0 {
            cold.ttft_ms / warm.ttft_ms
        } else {
            0.0
        };
        radix_summary = serde_json::json!({
            "prompt_len": r_prompt.len(),
            "gen_len": r_gen,
            "cold_ttft_ms": cold.ttft_ms,
            "warm_ttft_ms": warm.ttft_ms,
            "ttft_speedup_x": ttft_speedup_x,
            "ttft_reduction_pct": ttft_reduction_pct,
            "cached_prefix_tokens": warm.cached_prefix_tokens,
            "donated_seq_len": cold.donated_seq_len,
            "proposer_restored_tokens": restored_proposer,
            "warm_decode_tok_per_s": warm.decode_tok_per_s,
            "lossless_parity": lossless_parity,
        });
    }

    let mut scheduler_summary = serde_json::Value::Null;
    if matches!(bench_stage.as_str(), "scheduler" | "all") {
        engine.reset_all_states().map_err(|e| e.to_string())?;
        let shared_prefix = make_prompt(64);
        let s_gen = gen_len.clamp(4, 32);
        let mut batch_prompts = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut p = shared_prefix.clone();
            for t in 0..16 {
                p.push(base_ids[(b * 7 + t) % base_ids.len()]);
            }
            batch_prompts.push(p);
        }
        // Seed shared 64-token prefix into MambaRadixCache so batch exercises prefix sharing
        let _ = engine
            .generate_with_radix_cache(&batch_prompts[0], 2, 64, false)
            .map_err(|e| e.to_string())?;
        let sched_cfg = w34_host::scheduler::SchedulerConfig {
            max_num_seqs: batch_size.min(8),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let batch_res = engine
            .generate_batch_scheduled(&batch_prompts, s_gen, sched_cfg)
            .map_err(|e| e.to_string())?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let total_prompt_tokens: usize = batch_prompts.iter().map(Vec::len).sum();
        let total_cached_tokens: usize = batch_res.iter().map(|(_, c)| *c).sum();
        let total_gen_tokens: usize = batch_res.iter().map(|(g, _)| g.len()).sum();
        let aggregate_decode_tok_per_s = if wall_ms > 0.0 {
            total_gen_tokens as f64 * 1000.0 / wall_ms
        } else {
            0.0
        };
        let total_tok_per_s = if wall_ms > 0.0 {
            (total_prompt_tokens + total_gen_tokens) as f64 * 1000.0 / wall_ms
        } else {
            0.0
        };
        scheduler_summary = serde_json::json!({
            "batch_size": batch_size,
            "max_active_seqs": sched_cfg.max_num_seqs,
            "total_prompt_tokens": total_prompt_tokens,
            "total_cached_prefix_tokens": total_cached_tokens,
            "radix_hit_rate": engine.radix.hit_rate(),
            "total_gen_tokens": total_gen_tokens,
            "wall_ms": wall_ms,
            "aggregate_decode_tok_per_s": aggregate_decode_tok_per_s,
            "total_tok_per_s": total_tok_per_s,
        });
    }

    let mut vision_summary = serde_json::Value::Null;
    if bench_stage == "vision" || (bench_stage == "all" && engine.cfg.vision_config.is_some()) {
        let vis_ready = match engine.ensure_vision_encoder() {
            Ok(()) => true,
            Err(e) if bench_stage == "all" => {
                vision_summary = serde_json::json!({
                    "skipped": true,
                    "reason": e.to_string(),
                });
                false
            }
            Err(e) => return Err(e.to_string()),
        };
        if vis_ready {
            engine.reset_all_states().map_err(|e| e.to_string())?;
            let urls = if image_urls.is_empty() {
                vec!["rgb://https://w34.test/bench_img_a.ppm".to_string()]
            } else {
                image_urls.clone()
            };
            let (v_prompt_str, mm_imgs_a) = prepare_multimodal_prompt_and_images(
                &engine.cfg,
                "Describe the colors, geometry, and layout of this image in detail across several sentences.",
                &urls,
                allow_remote_images,
            )?;
            let (_, mm_imgs_b) = prepare_multimodal_prompt_and_images(
                &engine.cfg,
                "Describe the colors, geometry, and layout of this image in detail across several sentences.",
                &["rgb://https://w34.test/bench_img_b_distinct.ppm".to_string()],
                allow_remote_images,
            )?;
            let mut v_prompt_ids = engine.encode(&v_prompt_str).map_err(|e| e.to_string())?;
            while v_prompt_ids.len() < 72 {
                v_prompt_ids.extend_from_slice(&base_ids);
            }
            let v_gen = gen_len.clamp(4, 32);
            let cold = engine
                .generate_multimodal_with_radix_cache(&v_prompt_ids, &mm_imgs_a, v_gen, 64, false)
                .map_err(|e| e.to_string())?;
            let warm = engine
                .generate_multimodal_with_radix_cache(&v_prompt_ids, &mm_imgs_a, v_gen, 64, false)
                .map_err(|e| e.to_string())?;
            let distinct = engine
                .generate_multimodal_with_radix_cache(&v_prompt_ids, &mm_imgs_b, 4, 64, false)
                .map_err(|e| e.to_string())?;
            let lossless_parity = cold.generated_token_ids == warm.generated_token_ids;
            if !lossless_parity {
                return Err(
                    "vision benchmark: warm multimodal radix hit diverged from cold reference"
                        .into(),
                );
            }
            if warm.cached_prefix_tokens < 64 {
                return Err(format!(
                    "vision benchmark: expected >= 64 cached prefix tokens on warm image hit, got {}",
                    warm.cached_prefix_tokens
                ));
            }
            let content_hash_isolated = distinct.cached_prefix_tokens < 64;
            if !content_hash_isolated {
                return Err(format!(
                    "vision benchmark: distinct image falsely matched {} cached prefix tokens",
                    distinct.cached_prefix_tokens
                ));
            }
            let merge_sz = engine
                .cfg
                .vision_config
                .as_ref()
                .map(|v| v.spatial_merge_size)
                .unwrap_or(2);
            let visual_tokens: usize = mm_imgs_a
                .iter()
                .map(|img| img.num_merged_tokens(merge_sz))
                .sum();
            let ttft_reduction_pct = if cold.ttft_ms > 0.0 {
                100.0 * (1.0 - warm.ttft_ms / cold.ttft_ms)
            } else {
                0.0
            };
            let ttft_speedup_x = if warm.ttft_ms > 0.0 {
                cold.ttft_ms / warm.ttft_ms
            } else {
                0.0
            };
            vision_summary = serde_json::json!({
                "image_grid_thw": mm_imgs_a.first().map(|img| img.grid_thw),
                "visual_tokens": visual_tokens,
                "prompt_len": v_prompt_ids.len(),
                "gen_len": v_gen,
                "cold_ttft_ms": cold.ttft_ms,
                "warm_ttft_ms": warm.ttft_ms,
                "ttft_speedup_x": ttft_speedup_x,
                "ttft_reduction_pct": ttft_reduction_pct,
                "cached_prefix_tokens": warm.cached_prefix_tokens,
                "distinct_image_cached_tokens": distinct.cached_prefix_tokens,
                "content_hash_isolated": content_hash_isolated,
                "warm_decode_tok_per_s": warm.decode_tok_per_s,
                "lossless_parity": lossless_parity,
            });
        }
    }

    let summary = match bench_stage.as_str() {
        "decode" => decode_summary,
        "radix" => radix_summary,
        "scheduler" => scheduler_summary,
        "vision" => vision_summary,
        "all" => serde_json::json!({
            "model": model,
            "tp": tp,
            "kv_format": kv_format,
            "speculative": spec_label,
            "decode": decode_summary,
            "radix": radix_summary,
            "scheduler": scheduler_summary,
            "vision": vision_summary,
        }),
        _ => unreachable!(),
    };
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    if let Some(out_p) = out_json {
        fs::write(out_p, serde_json::to_string_pretty(&summary).unwrap())
            .map_err(|e| e.to_string())?;
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_serve(
    options: CudaCliOptions,
    policy: KvFormatPolicy,
    spec_config: Option<w34_host::proposer::SpeculativeConfig>,
) -> Result<ExitCode, String> {
    let CudaCliOptions {
        model,
        serve,
        resolved_model,
        tp,
        host,
        port,
        allow_remote_images,
        kv_pages,
        use_cuda_graph,
        ..
    } = options;
    let model = serve
        .model_name
        .map(|name| name.trim().to_string())
        .unwrap_or(model);
    let max_num_seqs = serve.options.max_num_seqs;
    let auth_enabled = serve.options.api_key.is_some();
    let bind_addr = format!("{host}:{port}");
    println!("[w34 serve] loading model={model} tp={tp} for http://{bind_addr}");
    let worker_model = resolved_model;
    let state = w34_host::server::OpenAiServerState::from_loader_with_options(
        model,
        allow_remote_images,
        serve.options,
        move || {
            let pages = kv_pages.or_else(|| spec_config.as_ref().map(|_| 256));
            let mut engine = Qwen3_5Engine::load_for_serving(
                &worker_model,
                tp,
                policy,
                pages,
                use_cuda_graph,
                max_num_seqs,
            )
            .map_err(|e| e.to_string())?;
            if let Some(config) = &spec_config {
                engine
                    .enable_speculative(config)
                    .map_err(|e| e.to_string())?;
            }
            Ok(engine)
        },
    )?;
    println!(
        "[w34 serve] api_auth={} max_num_seqs={max_num_seqs} context_limit={} tokens (dashboard/metrics/health are public)",
        if auth_enabled { "enabled" } else { "disabled" },
        state.worker.max_model_len(),
    );
    w34_host::server::run_blocking_server(&bind_addr, state)
        .map_err(|e| format!("server error: {e}"))?;
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "cuda")]
pub(crate) fn run_chat(
    options: CudaCliOptions,
    policy: KvFormatPolicy,
    spec_config: Option<w34_host::proposer::SpeculativeConfig>,
) -> Result<ExitCode, String> {
    let CudaCliOptions {
        model,
        resolved_model,
        tp,
        prompt,
        image_urls,
        allow_remote_images,
        enable_thinking,
        max_tokens,
        kv_format,
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
    let msgs = vec![w34_host::server::ChatMessage {
        tool_calls: Vec::new(),
        role: "user".to_string(),
        content: Some(expanded_prompt),
        reasoning_content: None,
    }];
    let formatted = w34_host::server::format_chatml_prompt(&msgs, None, enable_thinking);
    let prompt_ids = engine.encode(&formatted).map_err(|e| e.to_string())?;
    let metrics = if mm_images.is_empty() {
        engine
            .generate_with_radix_cache(&prompt_ids, max_tokens, 64, true)
            .map_err(|e| e.to_string())?
    } else {
        engine
            .generate_multimodal_with_radix_cache(&prompt_ids, &mm_images, max_tokens, 64, true)
            .map_err(|e| e.to_string())?
    };
    let raw_text = engine
        .decode(&metrics.generated_token_ids)
        .map_err(|e| e.to_string())?;
    let parsed = w34_host::parser::Qwen3Parser::parse_complete(&raw_text, enable_thinking);
    let img_tag = if mm_images.is_empty() {
        String::new()
    } else {
        format!(" images={}", mm_images.len())
    };
    println!(
        "[w34 chat] model={model} tp={tp} kv={kv_format}{img_tag} prompt_tokens={} cached_tokens={} gen_tokens={} ttft={:.2}ms decode={:.2}tok/s",
        metrics.prompt_len,
        metrics.cached_prefix_tokens,
        metrics.gen_len,
        metrics.ttft_ms,
        metrics.decode_tok_per_s
    );
    if let Some(ref r) = parsed.reasoning_content {
        println!("--- reasoning ---\n{r}");
    }
    if let Some(ref c) = parsed.content {
        println!("--- content ---\n{c}");
    }
    if !parsed.tool_calls.is_empty() {
        println!(
            "--- tool_calls ---\n{}",
            serde_json::to_string_pretty(&parsed.tool_calls).unwrap_or_default()
        );
    }
    Ok(ExitCode::SUCCESS)
}

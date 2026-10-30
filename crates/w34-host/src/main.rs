use std::env;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::path::Path;
use std::process::ExitCode;

#[cfg(feature = "cuda")]
use w34_host::cache::KvFormatPolicy;
#[cfg(feature = "cuda")]
use w34_host::model::Qwen3_5Engine;

#[cfg(feature = "cuda")]
fn parse_kv_policy(s: &str) -> Result<KvFormatPolicy, String> {
    match s.to_ascii_lowercase().as_str() {
        "fp16" | "f16" => Ok(KvFormatPolicy::Fp16),
        "bf16" => Ok(KvFormatPolicy::Bf16),
        "q8_0" | "q8" => Ok(KvFormatPolicy::Q8_0),
        "tq4" => Ok(KvFormatPolicy::Tq4),
        "tq4_all" | "tq4-all" => Ok(KvFormatPolicy::Tq4All),
        other => Err(format!(
            "unknown --kv-format '{other}' (expected tq4, tq4_all, q8_0, fp16, or bf16)"
        )),
    }
}

fn print_usage() {
    eprintln!(
        "Usage: w34 <subcommand> [options]\n\
         Subcommands:\n\
           generate     --model <id|path> [--tp 1|2] [--prompt <str>] [--image <path|url>]... [--max-tokens <n>] [--kv-format tq4|tq4_all|q8_0|fp16]\n\
           bench        --model <id|path> [--tp 1|2] [--stage decode|radix|scheduler|vision|all] [--batch-size <n>] [--prompt-len <n>] [--gen-len <n>] [--kv-format tq4|tq4_all|q8_0|fp16] [--out <json>]\n\
           greedy-diff  --model <id|path> --golden <dir|json> [--tp 1|2] [--max-prompts <n>] [--max-tokens <n>] [--kv-format tq4|tq4_all|q8_0|fp16] [--verify-lossless] [--verify-radix-warm]\n\
           dump-diff    --model <id|path> [--tp 1|2] [--prompt <str>] [--image <path|url>]... [--compare-tp2] [--golden <safetensors>]\n\
           profile-diff [--oracle <sqlite|json>] [--preset <key>] [--w34 <sqlite|json>] [--live] [--rounds <n>]\n\
           serve        --model <id|path> [--tp 1|2] [--host <addr>] [--port <n>] [--kv-format tq4|tq4_all|q8_0|fp16]\n\
           chat         --model <id|path> [--tp 1|2] [--prompt <str>] [--image <path|url>]... [--max-tokens <n>] [--no-thinking]\n\
         Image options: --allow-remote-images enables public HTTP(S) fetch (16 MiB, 10 s, no redirects).\n\
         Speculative options (generate/bench/greedy-diff/dump-diff/profile-diff/serve/chat):\n\
           --speculative mtp|dflash2 [--draft-model <id|path>] [--mtp-k 1..3] [--kv-pages <n>]\n\
           Native MTP shares target weights; DFlash2 requires a draft model.\n\
           Speculative CLI mode is greedy (CUDA Graph verify by default; pass --no-cuda-graph for eager); serve also honors sampling parameters.\n\
           Default --kv-format is tq4 (rotated Q8_0 boundary layers + TQ4 interior). Native mode defaults to 256 target KV pages (4096 tokens)."
    );
}

#[cfg(feature = "cuda")]
fn prepare_multimodal_prompt_and_images(
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
fn tensor_diff_metrics(a: &[f32], b: &[f32]) -> (f32, f64, f64) {
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
    let cosine = if denom > 1e-30 { dot / denom } else { 1.0 };
    (max_abs, rel_l2, cosine)
}

#[cfg(feature = "cuda")]
fn run_cuda_cli(args: &[String]) -> Result<ExitCode, String> {
    if args.is_empty() {
        print_usage();
        return Ok(ExitCode::SUCCESS);
    }
    let sub = args[0].as_str();
    let mut model = "nchapman/Qwen3.5-4B-W4A16-GPTQ".to_string();
    let mut tp = 1usize;
    let mut prompt =
        "Explain the difference between a mutex and a semaphore in two sentences.".to_string();
    let mut max_tokens = 64usize;
    let mut prompt_len = 128usize;
    let mut gen_len = 128usize;
    let mut kv_format = "tq4".to_string();
    let mut golden_path: Option<String> = None;
    let mut out_json: Option<String> = None;
    let mut max_prompts = 12usize;
    let mut use_cuda_graph = true;
    let mut compare_tp2 = false;
    let mut oracle_path: Option<String> = None;
    let mut preset_key: Option<String> = None;
    let mut w34_prof_path: Option<String> = None;
    let mut prof_rounds = 128usize;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8000u16;
    let mut enable_thinking = true;
    let mut speculative_kind = None;
    let mut draft_model = None;
    let mut mtp_k = 3usize;
    let mut kv_pages = None;
    let mut bench_stage = "decode".to_string();
    let mut batch_size = 8usize;
    let mut verify_lossless = false;
    let mut verify_radix_warm = false;
    let mut live_profile = false;
    let mut image_urls: Vec<String> = Vec::new();
    let mut allow_remote_images = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--speculative" => {
                i += 1;
                speculative_kind = Some(match args.get(i).map(String::as_str) {
                    Some("mtp") => w34_host::proposer::SpeculativeKind::Mtp,
                    Some("dflash2") => w34_host::proposer::SpeculativeKind::DFlash2,
                    _ => return Err("--speculative expects mtp or dflash2".into()),
                });
            }
            "--draft-model" => {
                i += 1;
                draft_model = Some(args.get(i).cloned().ok_or("missing --draft-model")?);
            }
            "--mtp-k" => {
                i += 1;
                mtp_k = args
                    .get(i)
                    .ok_or("missing --mtp-k")?
                    .parse()
                    .map_err(|_| "invalid --mtp-k")?;
            }
            "--kv-pages" => {
                i += 1;
                kv_pages = Some(
                    args.get(i)
                        .ok_or("missing --kv-pages")?
                        .parse::<usize>()
                        .map_err(|_| "invalid --kv-pages")?,
                );
            }
            "--model" | "-m" => {
                i += 1;
                model = args.get(i).cloned().ok_or("missing --model value")?;
            }
            "--tp" => {
                i += 1;
                tp = args
                    .get(i)
                    .ok_or("missing --tp value")?
                    .parse()
                    .map_err(|_| "invalid --tp")?;
            }
            "--prompt" | "-p" => {
                i += 1;
                prompt = args.get(i).cloned().ok_or("missing --prompt value")?;
            }
            "--max-tokens" | "-n" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or("missing --max-tokens")?
                    .parse()
                    .map_err(|_| "invalid --max-tokens")?;
            }
            "--prompt-len" => {
                i += 1;
                prompt_len = args
                    .get(i)
                    .ok_or("missing --prompt-len")?
                    .parse()
                    .map_err(|_| "invalid --prompt-len")?;
            }
            "--gen-len" => {
                i += 1;
                gen_len = args
                    .get(i)
                    .ok_or("missing --gen-len")?
                    .parse()
                    .map_err(|_| "invalid --gen-len")?;
            }
            "--kv-format" => {
                i += 1;
                kv_format = args.get(i).cloned().ok_or("missing --kv-format")?;
            }
            "--golden" => {
                i += 1;
                golden_path = Some(args.get(i).cloned().ok_or("missing --golden")?);
            }
            "--out" => {
                i += 1;
                out_json = Some(args.get(i).cloned().ok_or("missing --out")?);
            }
            "--max-prompts" => {
                i += 1;
                max_prompts = args
                    .get(i)
                    .ok_or("missing --max-prompts")?
                    .parse()
                    .map_err(|_| "invalid --max-prompts")?;
            }
            "--no-cuda-graph" => {
                use_cuda_graph = false;
            }
            "--compare-tp2" => {
                compare_tp2 = true;
            }
            "--oracle" => {
                i += 1;
                oracle_path = Some(args.get(i).cloned().ok_or("missing --oracle")?);
            }
            "--preset" => {
                i += 1;
                preset_key = Some(args.get(i).cloned().ok_or("missing --preset")?);
            }
            "--w34" => {
                i += 1;
                w34_prof_path = Some(args.get(i).cloned().ok_or("missing --w34")?);
            }
            "--rounds" => {
                i += 1;
                prof_rounds = args
                    .get(i)
                    .ok_or("missing --rounds")?
                    .parse()
                    .map_err(|_| "invalid --rounds")?;
            }
            "--host" => {
                i += 1;
                host = args.get(i).cloned().ok_or("missing --host")?;
            }
            "--port" => {
                i += 1;
                port = args
                    .get(i)
                    .ok_or("missing --port")?
                    .parse()
                    .map_err(|_| "invalid --port")?;
            }
            "--no-thinking" => {
                enable_thinking = false;
            }
            "--stage" => {
                i += 1;
                bench_stage = args.get(i).cloned().ok_or("missing --stage value")?;
                if !matches!(
                    bench_stage.as_str(),
                    "decode" | "radix" | "scheduler" | "vision" | "all"
                ) {
                    return Err("--stage expects decode, radix, scheduler, vision, or all".into());
                }
            }
            "--allow-remote-images" => allow_remote_images = true,
            "--image" => {
                i += 1;
                image_urls.push(args.get(i).cloned().ok_or("missing --image value")?);
            }
            "--batch-size" => {
                i += 1;
                batch_size = args
                    .get(i)
                    .ok_or("missing --batch-size")?
                    .parse()
                    .map_err(|_| "invalid --batch-size")?;
                if !(1..=1024).contains(&batch_size) {
                    return Err("--batch-size must be 1..1024".into());
                }
            }
            "--verify-lossless" => {
                verify_lossless = true;
            }
            "--verify-radix-warm" => {
                verify_radix_warm = true;
            }
            "--live" => {
                live_profile = true;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(ExitCode::SUCCESS);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let policy = parse_kv_policy(&kv_format)?;
    if kv_pages == Some(0) {
        return Err("--kv-pages must be positive".into());
    }
    if !matches!(mtp_k, 1..=3) {
        return Err("--mtp-k must be 1..3".into());
    }
    if draft_model.is_some()
        && speculative_kind != Some(w34_host::proposer::SpeculativeKind::DFlash2)
    {
        return Err(
            "--draft-model requires --speculative dflash2; native MTP uses target weights".into(),
        );
    }
    if speculative_kind == Some(w34_host::proposer::SpeculativeKind::DFlash2)
        && draft_model.is_none()
    {
        return Err("--speculative dflash2 requires --draft-model".into());
    }
    if verify_lossless && speculative_kind.is_none() {
        return Err("--verify-lossless requires --speculative".into());
    }
    if speculative_kind.is_some()
        && !matches!(
            sub,
            "generate" | "bench" | "greedy-diff" | "dump-diff" | "profile-diff" | "serve" | "chat"
        )
    {
        return Err(
            "speculation is supported by generate, bench, greedy-diff, dump-diff, profile-diff, serve and chat".into(),
        );
    }
    let spec_config = speculative_kind.map(|kind| w34_host::proposer::SpeculativeConfig {
        kind,
        draft_model: draft_model.clone(),
        mtp_k,
    });
    // Leave VRAM for the resident drafter instead of filling it with target KV.
    // Speculative taps/logits are configured in enable_speculative before lazy graph capture.
    let load_engine = || -> w34_host::cuda::Result<Qwen3_5Engine> {
        let pages = kv_pages.or_else(|| spec_config.as_ref().map(|_| 256));
        let mut engine = Qwen3_5Engine::load(&model, tp, policy, pages, use_cuda_graph)?;
        if let Some(config) = &spec_config {
            engine.enable_speculative(config)?;
        }
        Ok(engine)
    };

    match sub {
        "generate" => {
            let mut engine = load_engine().map_err(|e| e.to_string())?;
            let (expanded_prompt, mm_images) = prepare_multimodal_prompt_and_images(
                &engine.cfg,
                &prompt,
                &image_urls,
                allow_remote_images,
            )?;
            let prompt_ids = engine.encode(&expanded_prompt).map_err(|e| e.to_string())?;
            let (gen_ids, prompt_len, gen_len, ttft_ms, decode_tok_per_s, itl_ms) =
                if mm_images.is_empty() {
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
                        .generate_multimodal_with_radix_cache(
                            &prompt_ids,
                            &mm_images,
                            max_tokens,
                            64,
                            true,
                        )
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
        "bench" => {
            let mut engine = load_engine().map_err(|e| e.to_string())?;
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
                        "radix benchmark: warm prefix-hit output diverged from cold reference"
                            .into(),
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
            if bench_stage == "vision"
                || (bench_stage == "all" && engine.cfg.vision_config.is_some())
            {
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
                    let mut v_prompt_ids =
                        engine.encode(&v_prompt_str).map_err(|e| e.to_string())?;
                    while v_prompt_ids.len() < 72 {
                        v_prompt_ids.extend_from_slice(&base_ids);
                    }
                    let v_gen = gen_len.clamp(4, 32);
                    let cold = engine
                        .generate_multimodal_with_radix_cache(
                            &v_prompt_ids,
                            &mm_imgs_a,
                            v_gen,
                            64,
                            false,
                        )
                        .map_err(|e| e.to_string())?;
                    let warm = engine
                        .generate_multimodal_with_radix_cache(
                            &v_prompt_ids,
                            &mm_imgs_a,
                            v_gen,
                            64,
                            false,
                        )
                        .map_err(|e| e.to_string())?;
                    let distinct = engine
                        .generate_multimodal_with_radix_cache(
                            &v_prompt_ids,
                            &mm_imgs_b,
                            4,
                            64,
                            false,
                        )
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
        "greedy-diff" => {
            let golden_arg = golden_path.ok_or("--golden <dir-or-json> is required")?;
            let golden_p = Path::new(&golden_arg);
            let mut engine = load_engine().map_err(|e| e.to_string())?;

            #[allow(clippy::type_complexity)]
            let mut cases: Vec<(
                String,
                Vec<i32>,
                Vec<i32>,
                Option<Vec<Vec<(i32, f64)>>>,
            )> = Vec::new();
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
                            let prompt_ids =
                                engine.encode(prompt_text).map_err(|e| e.to_string())?;
                            let ids_str =
                                fs::read_to_string(&ids_file).map_err(|e| e.to_string())?;
                            let ref_gen_ids: Vec<i32> = ids_str
                                .split_whitespace()
                                .filter_map(|s| s.parse::<i32>().ok())
                                .collect();
                            let top5_file = golden_p.join(format!("{tag}.top5.json"));
                            let top5_opt = if top5_file.exists() {
                                let t_str =
                                    fs::read_to_string(&top5_file).map_err(|e| e.to_string())?;
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

            let mut total_tokens_compared = 0usize;
            let mut total_matches = 0usize;
            let mut first_step_matches = 0usize;
            let mut adjudicated_passes = 0usize;
            let mut total_spec_rounds = 0usize;
            let mut total_spec_accepted = 0usize;
            let mut lossless_passes = 0usize;
            let mut radix_warm_passes = 0usize;
            let num_cases = cases.len().min(max_prompts);

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
                        .map(|m| w34_host::model::BenchMetrics {
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
                            flip_info =
                                format!("seat={prefix_match} got={got_tok}(>top5) want={want_tok}");
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
        "dump-diff" => {
            let mut engine = load_engine().map_err(|e| e.to_string())?;
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
                let mut engine_tp2 = Qwen3_5Engine::load(&model, 2, policy, pages, false)
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
                let archive = w34_host::loader::SafetensorsArchive::open(Path::new(&gpath))
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
        "profile-diff" => {
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
                .unwrap_or_else(|| resolve_repo_file("tools/oracle/goldens/oracle_baselines.json"));
            let script = resolve_repo_file("tools/oracle/profile_diff.py");
            let mut cmd = std::process::Command::new("python3");
            cmd.arg("-B")
                .arg(&script)
                .arg("--oracle")
                .arg(&oracle)
                .arg("--rounds")
                .arg(prof_rounds.to_string());
            if let Some(p) = &preset_key {
                cmd.arg("--preset").arg(p);
            }
            if let Some(w) = &w34_prof_path {
                cmd.arg("--w34").arg(w);
            } else if live_profile {
                let exe = env::current_exe().map_err(|e| e.to_string())?;
                let mut bench_args = vec![
                    exe.display().to_string(),
                    "bench".to_string(),
                    "--model".to_string(),
                    model.clone(),
                    "--tp".to_string(),
                    tp.to_string(),
                    "--prompt-len".to_string(),
                    prompt_len.to_string(),
                    "--gen-len".to_string(),
                    gen_len.to_string(),
                    "--kv-format".to_string(),
                    kv_format.clone(),
                ];
                if let Some(kind) = speculative_kind {
                    bench_args.push("--speculative".to_string());
                    match kind {
                        w34_host::proposer::SpeculativeKind::Mtp => {
                            bench_args.push("mtp".to_string());
                            bench_args.push("--mtp-k".to_string());
                            bench_args.push(mtp_k.to_string());
                        }
                        w34_host::proposer::SpeculativeKind::DFlash2 => {
                            bench_args.push("dflash2".to_string());
                        }
                    }
                }
                if let Some(dm) = &draft_model {
                    bench_args.push("--draft-model".to_string());
                    bench_args.push(dm.clone());
                }
                if let Some(kp) = kv_pages {
                    bench_args.push("--kv-pages".to_string());
                    bench_args.push(kp.to_string());
                }
                cmd.arg("--live-cmd")
                    .arg(serde_json::to_string(&bench_args).unwrap());
            }
            if let Some(o) = &out_json {
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
        "serve" => {
            let bind_addr = format!("{host}:{port}");
            println!("[w34 serve] loading model={model} tp={tp} for http://{bind_addr}");
            let worker_model = model.clone();
            let state = w34_host::server::OpenAiServerState::from_loader(
                model,
                allow_remote_images,
                move || {
                    let pages = kv_pages.or_else(|| spec_config.as_ref().map(|_| 256));
                    let mut engine = Qwen3_5Engine::load_for_serving(
                        &worker_model,
                        tp,
                        policy,
                        pages,
                        use_cuda_graph,
                        w34_host::scheduler::SchedulerConfig::default().max_num_seqs,
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
            w34_host::server::run_blocking_server(&bind_addr, state)
                .map_err(|e| format!("server error: {e}"))?;
            Ok(ExitCode::SUCCESS)
        }
        "chat" => {
            let mut engine = load_engine().map_err(|e| e.to_string())?;
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
                    .generate_multimodal_with_radix_cache(
                        &prompt_ids,
                        &mm_images,
                        max_tokens,
                        64,
                        true,
                    )
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
        other => Err(format!("unknown subcommand: {other}")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    #[cfg(feature = "cuda")]
    {
        match run_cuda_cli(&args) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("w34 error: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        let _ = print_usage;
        println!("w34 v{} (cuda=false)", env!("CARGO_PKG_VERSION"));
        ExitCode::SUCCESS
    }
}

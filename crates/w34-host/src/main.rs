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
        "tq4" => Ok(KvFormatPolicy::Tq4),
        "k8v4" => Ok(KvFormatPolicy::K8V4),
        "firstlast2" | "first-last2" => Ok(KvFormatPolicy::FirstLast2),
        other => Err(format!(
            "unknown --kv-format '{other}' (expected fp16, bf16, tq4, k8v4, or firstlast2)"
        )),
    }
}

fn print_usage() {
    eprintln!(
        "Usage: w34 <subcommand> [options]\n\
         Subcommands:\n\
           generate     --model <id|path> [--tp 1|2] [--prompt <str>] [--max-tokens <n>] [--kv-format fp16|tq4|k8v4|firstlast2]\n\
           bench        --model <id|path> [--tp 1|2] [--prompt-len <n>] [--gen-len <n>] [--kv-format fp16|tq4|k8v4|firstlast2] [--out <json>]\n\
           greedy-diff  --model <id|path> --golden <dir|json> [--tp 1|2] [--max-prompts <n>] [--max-tokens <n>]\n\
           dump-diff    --model <id|path> [--tp 1|2] [--prompt <str>] [--compare-tp2] [--golden <safetensors>]\n\
           profile-diff --oracle <sqlite|json> [--preset <key>] [--w34 <sqlite|json>] [--rounds <n>]\n\
           serve        --model <id|path> [--tp 1|2] [--host <addr>] [--port <n>] [--kv-format fp16|tq4|k8v4|firstlast2]\n\
           chat         --model <id|path> [--tp 1|2] [--prompt <str>] [--max-tokens <n>] [--no-thinking]"
    );
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
    let mut kv_format = "fp16".to_string();
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

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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
            "--help" | "-h" => {
                print_usage();
                return Ok(ExitCode::SUCCESS);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let policy = parse_kv_policy(&kv_format)?;

    match sub {
        "generate" => {
            let mut engine = Qwen3_5Engine::load(&model, tp, policy, None, use_cuda_graph)
                .map_err(|e| e.to_string())?;
            let prompt_ids = engine.encode(&prompt).map_err(|e| e.to_string())?;
            let metrics = engine
                .generate_tokens(&prompt_ids, max_tokens, true)
                .map_err(|e| e.to_string())?;
            let text = engine
                .decode(&metrics.generated_token_ids)
                .map_err(|e| e.to_string())?;
            println!("{text}");
            eprintln!(
                "[w34 generate] model={model} tp={tp} kv={kv_format} prompt_toks={} gen_toks={} ttft={:.2}ms decode={:.2}tok/s ({:.2}ms/tok)",
                metrics.prompt_len,
                metrics.gen_len,
                metrics.ttft_ms,
                metrics.decode_tok_per_s,
                metrics.itl_ms
            );
            Ok(ExitCode::SUCCESS)
        }
        "bench" => {
            let mut engine = Qwen3_5Engine::load(&model, tp, policy, None, use_cuda_graph)
                .map_err(|e| e.to_string())?;
            let base_ids = engine
                .encode("The quick brown fox jumps over the lazy dog. ")
                .map_err(|e| e.to_string())?;
            let mut synth_prompt = Vec::with_capacity(prompt_len);
            while synth_prompt.len() < prompt_len {
                for &id in &base_ids {
                    if synth_prompt.len() < prompt_len {
                        synth_prompt.push(id);
                    }
                }
            }
            // Warmup pass (captures CUDA Graph on decode step 1 and ramps GPU boost clock out of P8)
            let _ = engine
                .generate_tokens(&synth_prompt[..16.min(prompt_len)], 32, false)
                .map_err(|e| e.to_string())?;

            let metrics = engine
                .generate_tokens(&synth_prompt, gen_len, false)
                .map_err(|e| e.to_string())?;
            let summary = serde_json::json!({
                "model": model,
                "tp": tp,
                "kv_format": kv_format,
                "cuda_graph": use_cuda_graph,
                "prompt_len": metrics.prompt_len,
                "gen_len": metrics.gen_len,
                "ttft_ms": metrics.ttft_ms,
                "prefill_tok_per_s": metrics.prefill_tok_per_s,
                "decode_ms": metrics.decode_ms,
                "decode_tok_per_s": metrics.decode_tok_per_s,
                "itl_ms": metrics.itl_ms,
            });
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
            let mut engine = Qwen3_5Engine::load(&model, tp, policy, None, use_cuda_graph)
                .map_err(|e| e.to_string())?;

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
            let num_cases = cases.len().min(max_prompts);

            for (id, prompt_ids, ref_gen_ids, top5_opt) in cases.iter().take(num_cases) {
                let cmp_len = ref_gen_ids.len().min(max_tokens);
                let metrics = engine
                    .generate_tokens(prompt_ids, cmp_len, false)
                    .map_err(|e| e.to_string())?;

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
                            is_benign =
                                (r_idx == 1 && gap < 0.60) || (prefix_match >= 64 && r_idx <= 2);
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
                    "[greedy-diff] case={id:<10} first_tok={} prefix={prefix_match:>3}/{cmp_len} exact={exact_matches:>3}/{cmp_len} decode={:>5.1}tok/s [{flip_info}]",
                    metrics.generated_token_ids.first() == ref_gen_ids.first(),
                    metrics.decode_tok_per_s
                );
            }
            println!(
                "[greedy-diff summary] model={model} tp={tp} first_step={first_step_matches}/{num_cases} adjudicated={adjudicated_passes}/{num_cases} total_token_match={total_matches}/{total_tokens_compared} ({:.1}%)",
                if total_tokens_compared > 0 {
                    100.0 * (total_matches as f64) / (total_tokens_compared as f64)
                } else {
                    100.0
                }
            );
            Ok(ExitCode::SUCCESS)
        }
        "dump-diff" => {
            let mut engine =
                Qwen3_5Engine::load(&model, tp, policy, None, false).map_err(|e| e.to_string())?;
            let prompt_ids = engine.encode(&prompt).map_err(|e| e.to_string())?;
            let dump = engine.dump_prompt(&prompt_ids).map_err(|e| e.to_string())?;
            let decoded_top1 = engine
                .decode(&[dump.greedy_token_id])
                .unwrap_or_else(|_| "<err>".to_string());
            println!(
                "[dump-diff] model={model} tp={tp} prompt_len={} tensors_captured={} greedy_id={} ({:?}) top5_ids={:?} top5_logprobs={:?}",
                prompt_ids.len(),
                dump.tensors_f32.len(),
                dump.greedy_token_id,
                decoded_top1,
                dump.top5_ids,
                dump.top5_logprobs
            );
            if compare_tp2 {
                drop(engine);
                let mut engine_tp2 = Qwen3_5Engine::load(&model, 2, policy, None, false)
                    .map_err(|e| e.to_string())?;
                let dump_tp2 = engine_tp2
                    .dump_prompt(&prompt_ids)
                    .map_err(|e| e.to_string())?;
                let mut max_layer_diff = 0.0f32;
                for (name, (shape1, vals1)) in &dump.tensors_f32 {
                    if name == "logits" {
                        continue; // logits_buf on rank 0 is vocab-sharded under TP=2
                    }
                    if let Some((shape2, vals2)) = dump_tp2.tensors_f32.get(name)
                        && shape1 == shape2
                    {
                        let mut mdiff = 0.0f32;
                        for (&a, &b) in vals1.iter().zip(vals2.iter()) {
                            mdiff = mdiff.max((a - b).abs());
                        }
                        max_layer_diff = max_layer_diff.max(mdiff);
                    }
                }
                println!(
                    "[dump-diff tp1-vs-tp2] model={model} max_hidden_diff={max_layer_diff:.6} tp1_top1={} tp2_top1={} match={}",
                    dump.greedy_token_id,
                    dump_tp2.greedy_token_id,
                    dump.greedy_token_id == dump_tp2.greedy_token_id
                );
            }
            if let Some(gpath) = golden_path {
                let archive = w34_host::loader::SafetensorsArchive::open(Path::new(&gpath))
                    .map_err(|e| e.to_string())?;
                let mut compared = 0usize;
                let mut max_diff = 0.0f32;
                for (name, (_shape, vals)) in &dump.tensors_f32 {
                    if let Some(resolved) = archive.resolve_name(name) {
                        let ref_vals = archive.read_f32(&resolved).map_err(|e| e.to_string())?;
                        if ref_vals.len() == vals.len() {
                            let mut d = 0.0f32;
                            for (&a, &b) in vals.iter().zip(ref_vals.iter()) {
                                d = d.max((a - b).abs());
                            }
                            max_diff = max_diff.max(d);
                            compared += 1;
                        }
                    }
                }
                println!(
                    "[dump-diff golden] golden={gpath} compared_tensors={compared} max_abs_diff={max_diff:.6}"
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        "profile-diff" => {
            let oracle = oracle_path
                .unwrap_or_else(|| "tools/oracle/goldens/oracle_baselines.json".to_string());
            let mut cmd = std::process::Command::new("python3");
            cmd.arg("tools/oracle/profile_diff.py")
                .arg("--oracle")
                .arg(&oracle)
                .arg("--rounds")
                .arg(prof_rounds.to_string());
            if let Some(p) = &preset_key {
                cmd.arg("--preset").arg(p);
            }
            if let Some(w) = &w34_prof_path {
                cmd.arg("--w34").arg(w);
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
            let engine = Qwen3_5Engine::load(&model, tp, policy, None, use_cuda_graph)
                .map_err(|e| e.to_string())?;
            let bind_addr = format!("{host}:{port}");
            println!("[w34 serve] model={model} tp={tp} listening on http://{bind_addr}");
            let state = w34_host::server::OpenAiServerState::new(model, engine);
            w34_host::server::run_blocking_server(&bind_addr, state)
                .map_err(|e| format!("server error: {e}"))?;
            Ok(ExitCode::SUCCESS)
        }
        "chat" => {
            let mut engine = Qwen3_5Engine::load(&model, tp, policy, None, use_cuda_graph)
                .map_err(|e| e.to_string())?;
            let msgs = vec![w34_host::server::ChatMessage {
                role: "user".to_string(),
                content: Some(prompt.clone()),
                reasoning_content: None,
            }];
            let formatted = w34_host::server::format_chatml_prompt(&msgs, None, enable_thinking);
            let prompt_ids = engine.encode(&formatted).map_err(|e| e.to_string())?;
            let metrics = engine
                .generate_with_radix_cache(&prompt_ids, max_tokens, 64, true)
                .map_err(|e| e.to_string())?;
            let raw_text = engine
                .decode(&metrics.generated_token_ids)
                .map_err(|e| e.to_string())?;
            let parsed = w34_host::parser::Qwen3Parser::parse_complete(&raw_text, enable_thinking);
            println!(
                "[w34 chat] model={model} tp={tp} prompt_tokens={} cached_tokens={} gen_tokens={} ttft={:.2}ms decode={:.2}tok/s",
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

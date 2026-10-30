//! Reproducible single-GPU, fixed-token-window needle retrieval gate.
//! Usage: cargo run --release --features cuda --example needle_gate -- MODEL POLICY WINDOW DEPTH_PERCENT OUTPUT.json
//! WINDOW includes prompt, 32 output tokens, and four reserved engine tokens.

#[cfg(not(feature = "cuda"))]
fn main() -> std::process::ExitCode {
    eprintln!("needle_gate requires --features cuda");
    std::process::ExitCode::FAILURE
}

#[cfg(feature = "cuda")]
fn main() -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
    use w34_host::{cache::KvFormatPolicy, model::Qwen3_5Engine};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: needle_gate MODEL tq4|tq4_all WINDOW DEPTH_PERCENT OUTPUT.json".into());
    }
    let policy = match args[1].as_str() {
        "tq4" => KvFormatPolicy::Tq4,
        "tq4_all" => KvFormatPolicy::Tq4All,
        _ => return Err("expected tq4 or tq4_all".into()),
    };
    let window: usize = args[2].parse()?;
    let depth: usize = args[3].parse()?;
    if !(512..=262144).contains(&window) || !window.is_multiple_of(16) || depth > 100 {
        return Err("window must be 512..262144, multiple of 16; depth must be 0..100".into());
    }
    let max_new = 32;
    let prompt_len = window - max_new - 4;
    let mut engine = Qwen3_5Engine::load(&args[0], 1, policy, Some(window / 16), true)?;
    let prefix = engine.encode("<|im_start|>system\nAnswer the user's question using the supplied archive. Return only the requested eight-digit access code, without explanation.<|im_end|>\n<|im_start|>user\nArchive begins:\n")?;
    let filler = engine.encode("The daily archive records mild weather, routine maintenance, and ordinary deliveries. No unusual activity was reported.\n")?;
    let needle = engine.encode("\nIMPORTANT RECORD: The secret access code is 73918462. Keep this exact eight-digit code for later retrieval.\n")?;
    let suffix = engine.encode("\nArchive ends. What is the secret access code? Return only the eight digits.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")?;
    let padding = prompt_len
        .checked_sub(prefix.len() + needle.len() + suffix.len())
        .ok_or("window too short for prompt")?;
    let before = padding * depth / 100;
    let needle_offset = prefix.len() + before;
    let mut prompt = prefix;
    prompt.extend(filler.iter().copied().cycle().take(before));
    prompt.extend(needle);
    prompt.extend(filler.iter().copied().cycle().take(padding - before));
    prompt.extend(suffix);
    assert_eq!(prompt.len(), prompt_len);
    eprintln!(
        "needle: model={} policy={} window={window} prompt={prompt_len} needle_offset={needle_offset} depth={depth}%",
        args[0], args[1]
    );
    let start = std::time::Instant::now();
    let metrics = engine.generate_with_radix_cache(&prompt, max_new, 256, true)?;
    let answer_ids: Vec<_> = metrics
        .generated_token_ids
        .iter()
        .copied()
        .take_while(|id| !engine.cfg.eos_token_ids.contains(id))
        .collect();
    let answer = engine.decode(&answer_ids)?;
    let passed = answer.trim() == "73918462";
    let report = serde_json::json!({
        "schema_version":1,"model":args[0],"model_snapshot":engine.model_dir,
        "kv_policy":args[1],"tp":1,"context_window":window,"prompt_tokens":prompt_len,
        "reserved_tokens":4,"max_new_tokens":max_new,"depth_percent":depth,"needle_token_offset":needle_offset,
        "expected":"73918462","answer":answer,"passed":passed,
        "generated_token_ids":metrics.generated_token_ids,"generated_tokens":metrics.gen_len,
        "cached_prefix_tokens":metrics.cached_prefix_tokens,"prefill_chunk_size":256,
        "ttft_ms":metrics.ttft_ms,"decode_tok_per_s":metrics.decode_tok_per_s,
        "elapsed_seconds":start.elapsed().as_secs_f64(),"prompt_token_ids":prompt
    });
    let out = std::path::Path::new(&args[4]);
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "needle window={window} depth={depth}% policy={} passed={passed} answer={answer:?} ttft_ms={:.2}",
        args[1], metrics.ttft_ms
    );
    Ok(if passed {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    })
}

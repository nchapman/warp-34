#![cfg(feature = "cuda")]

use std::path::PathBuf;

/// Runs the exact reference benchmark prompts (`benchmark_serving.py` `short-chat` `128:256` & `128:512`,
/// the 5-task `512`-token chat benchmark suite, and `scripts/bench.py` `4096:256` & `8192:1024`) on the real
/// 48-layer `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` / `nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ`
/// checkpoint (`TP=2 / EP=2` + `hot120` + `MTP-3`).
#[test]
#[ignore = "requires local Qwen3.8-Flash-Next checkpoint and 2x RTX 3090 Ti"]
fn test_qwen4_exp_reference_bench_suite() {
    use w34_host::cache::KvFormatPolicy;
    use w34_host::cuda::CudaDevice;
    use w34_host::model::Qwen3_5Engine;

    if CudaDevice::count().unwrap_or(0) < 2 {
        return;
    }
    let Ok(dir_buf) =
        w34_host::loader::resolve_hf_model_dir("nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ").or_else(
            |_| w34_host::loader::resolve_hf_model_dir("albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE"),
        )
    else {
        eprintln!("Skipping test_qwen4_exp_reference_bench_suite: checkpoint not cached");
        return;
    };
    let dir = dir_buf.as_path();

    let mut engine = Qwen3_5Engine::load_from_dir(dir, 2, KvFormatPolicy::Fp16, Some(640), true)
        .expect("load real Qwen3.8-Flash-Next TP=2 / EP=2");
    let spec_cfg = w34_host::proposer::SpeculativeConfig {
        kind: w34_host::proposer::SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    engine
        .enable_speculative(&spec_cfg)
        .expect("enable real Qwen3.8 MTP-3 on TP=2 / EP=2");
    for r in &engine.ranks {
        let (free_b, total_b) = r.device.mem_info().unwrap_or((0, 0));
        eprintln!(
            "[REF-BENCH VRAM] rank={} free={:.2} GiB / {:.2} GiB (kv_pages={})",
            r.rank,
            free_b as f64 / (1024.0 * 1024.0 * 1024.0),
            total_b as f64 / (1024.0 * 1024.0 * 1024.0),
            r.kv_pool.num_pages,
        );
    }

    // Collect `runtime/vllm-overlay` corpus exactly like `scripts/bench.py` / `scripts/benchmark_serving.py`
    // (falling back to the cached FP8PLE `runtime/vllm-overlay` when running against HGPTQ).
    let overlay_dir = {
        let primary = dir.join("runtime/vllm-overlay");
        let fallback_fp8 = PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(
            ".cache/huggingface/hub/models--albucino--Qwen3.8-Flash-Next-W4A16-FP8PLE/snapshots/ef554143369a706525336f6b42a09094835dc077/runtime/vllm-overlay",
        );
        if primary.is_dir() {
            primary
        } else if fallback_fp8.is_dir() {
            fallback_fp8
        } else {
            eprintln!(
                "Skipping test_qwen4_exp_reference_bench_suite: runtime/vllm-overlay corpus not found at {} or {}",
                primary.display(),
                fallback_fp8.display()
            );
            return;
        }
    };
    eprintln!("[REF-BENCH CORPUS] overlay_dir={}", overlay_dir.display());
    let mut overlay_files = Vec::new();
    fn visit_overlay(cur: &std::path::Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(cur) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    visit_overlay(&p, out);
                } else if p.is_file()
                    && p.extension()
                        .and_then(|s| s.to_str())
                        .is_some_and(|ext| matches!(ext, "py" | "cpp" | "cu"))
                {
                    out.push(p);
                }
            }
        }
    }
    visit_overlay(&overlay_dir, &mut overlay_files);
    overlay_files.sort();
    let mut corpus_text = String::new();
    for p in &overlay_files {
        let rel = p.strip_prefix(&overlay_dir).unwrap_or(p).display();
        let body = std::fs::read_to_string(p).unwrap_or_default();
        corpus_text.push_str(&format!("\n# ===== {rel} =====\n{body}"));
    }
    let corpus_ids = engine.encode(&corpus_text).expect("encode overlay corpus");
    if corpus_ids.is_empty() {
        eprintln!(
            "Skipping test_qwen4_exp_reference_bench_suite: runtime/vllm-overlay corpus is empty at {}",
            overlay_dir.display()
        );
        return;
    }

    // 1. `benchmark_serving.py` `short-chat` (`128` input tokens):
    //    Wrapper prefix/suffix from `<|im_start|>user\n{REPO_CHAT_PREFIX}{source}{REPO_CHAT_SUFFIX}<|im_end|>\n<|im_start|>assistant\n<think>\n`
    let repo_chat_prefix = "Write a detailed tutorial on multi-GPU inference: tensor parallelism, CPU offload, expert LRU caching, KV state, and speculative decoding. Use the source below where helpful. Include algorithms, numerical examples, tradeoffs, and tests. Continue with useful detail for the entire response budget.\n\n<source>\n";
    let repo_chat_suffix = "\n</source>";
    let empty_chat = format!(
        "<|im_start|>user\n{repo_chat_prefix}{repo_chat_suffix}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let prefix_char_end = corpus_text.floor_char_boundary(corpus_text.len().min(8192));
    let full_chat = format!(
        "<|im_start|>user\n{repo_chat_prefix}{}{repo_chat_suffix}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        &corpus_text[..prefix_char_end]
    );
    let empty_ids = engine.encode(&empty_chat).unwrap();
    let full_ids = engine.encode(&full_chat).unwrap();
    let mut pfx_len = 0;
    while pfx_len < empty_ids.len().min(full_ids.len()) && empty_ids[pfx_len] == full_ids[pfx_len] {
        pfx_len += 1;
    }
    let mut sfx_len = 0;
    let rem = empty_ids.len().min(full_ids.len()) - pfx_len;
    while sfx_len < rem
        && empty_ids[empty_ids.len() - 1 - sfx_len] == full_ids[full_ids.len() - 1 - sfx_len]
    {
        sfx_len += 1;
    }
    let interior = &full_ids[pfx_len..full_ids.len() - sfx_len];
    let src_budget = 128usize.saturating_sub(pfx_len + sfx_len);
    assert!(
        src_budget > 0 && interior.len() >= src_budget,
        "corpus excerpt too short for 128-token short-chat prompt"
    );
    let mut short_chat_128 = Vec::with_capacity(128);
    short_chat_128.extend_from_slice(&full_ids[..pfx_len]);
    short_chat_128.extend_from_slice(&interior[..src_budget]);
    short_chat_128.extend_from_slice(&full_ids[full_ids.len() - sfx_len..]);
    assert_eq!(short_chat_128.len(), 128);

    // 1a. Cold `short-chat 128:512` (first request after load, no prior warmup on this prompt)
    let m_cold_short = engine
        .generate_tokens(&short_chat_128, 512, false)
        .expect("cold short-chat 128:512");
    let acc_cold_short = engine
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_acc_cold = if acc_cold_short.is_empty() {
        0.0
    } else {
        acc_cold_short.iter().sum::<usize>() as f64 / acc_cold_short.len() as f64
    };
    eprintln!(
        "[REF-BENCH short-chat 128:512 COLD] ttft={:.2}ms prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
        m_cold_short.ttft_ms,
        m_cold_short.prefill_tok_per_s,
        m_cold_short.decode_tok_per_s,
        acc_cold_short.len(),
        mean_acc_cold,
        1.0 + mean_acc_cold,
    );

    assert_eq!(m_cold_short.generated_token_ids.len(), 512);
    assert!(
        mean_acc_cold > 1.0,
        "expected MTP acceptance > 1.0, got {mean_acc_cold}"
    );

    // 1b. Warm `short-chat 128:256` & `128:512` (matching `benchmark_serving.py` 1 warmup + measured protocol)
    for &out_len in &[256usize, 512usize] {
        let m = engine
            .generate_tokens(&short_chat_128, out_len, false)
            .expect("short-chat run");
        let accepted = engine
            .speculative
            .as_ref()
            .map(|p| p.last_accepted.clone())
            .unwrap_or_default();
        let mean_acc = if accepted.is_empty() {
            0.0
        } else {
            accepted.iter().sum::<usize>() as f64 / accepted.len() as f64
        };
        eprintln!(
            "[REF-BENCH short-chat 128:{} WARM] ttft={:.2}ms prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
            out_len,
            m.ttft_ms,
            m.prefill_tok_per_s,
            m.decode_tok_per_s,
            accepted.len(),
            mean_acc,
            1.0 + mean_acc,
        );
    }

    // 2. 5-prompt diverse chat suite (`512` output tokens each, no prefix reuse between prompts)
    let chat_suite: &[(&str, &str)] = &[
        (
            "creative_writing",
            "Write a vivid science-fiction short story about an archivist on a tidally locked planet who discovers an ancient signal buried inside glacial ice.",
        ),
        (
            "code_generation",
            "Implement a complete in-memory B+ tree in Python supporting insert, point lookup, and range scan, with unit tests.",
        ),
        (
            "reasoning_math",
            "A snail climbs a 30-meter well. Each day it climbs 7 meters and each night it slips back 4 meters, except on every 4th day when rain causes it to slip 6 meters that night. Show step by step on which day the snail reaches the top.",
        ),
        (
            "factual_essay",
            "Compare the economic, institutional, and cultural consequences of the European printing press with the rise of the modern internet.",
        ),
        (
            "technical_explain",
            "Explain how Pin and Unpin work in async Rust, why self-referential futures require pinning, and how projections are safely implemented.",
        ),
    ];
    let mut suite_tok_s = Vec::new();
    for &(name, user_msg) in chat_suite {
        let formatted =
            format!("<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n");
        let ids = engine.encode(&formatted).expect("encode chat prompt");
        let m = engine
            .generate_tokens(&ids, 512, false)
            .expect("chat suite 512 tokens");
        let accepted = engine
            .speculative
            .as_ref()
            .map(|p| p.last_accepted.clone())
            .unwrap_or_default();
        let mean_acc = if accepted.is_empty() {
            0.0
        } else {
            accepted.iter().sum::<usize>() as f64 / accepted.len() as f64
        };
        suite_tok_s.push(m.decode_tok_per_s);
        eprintln!(
            "[REF-BENCH chat-512 {name:<18}] prompt_len={:>3} ttft={:>6.2}ms decode={:>6.2} tok/s rounds={:>3} mean_accepted={:.2}/3 ({:.2} tok/round)",
            ids.len(),
            m.ttft_ms,
            m.decode_tok_per_s,
            accepted.len(),
            mean_acc,
            1.0 + mean_acc,
        );
    }
    let suite_mean = suite_tok_s.iter().sum::<f64>() / suite_tok_s.len() as f64;
    eprintln!(
        "[REF-BENCH chat-512 SUMMARY] mean_decode={:.2} tok/s min={:.2} tok/s max={:.2} tok/s",
        suite_mean,
        suite_tok_s.iter().cloned().fold(f64::INFINITY, f64::min),
        suite_tok_s
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );

    // 3. `scripts/bench.py` shape `4096:256` (`offset = 5000`, `n = 4096`, `TASKS[0]`)
    let window_4k: Vec<i32> = (0..4096)
        .map(|i| corpus_ids[(5000 + i) % corpus_ids.len()])
        .collect();
    let window_4k_text = engine.decode(&window_4k).unwrap();
    let task0 = "Review this code for correctness bugs. For each bug, explain the failure scenario and propose a fix.";
    let prompt_4k_str = format!(
        "<|im_start|>user\n{window_4k_text}\n\n{task0}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let prompt_4k_ids = engine.encode(&prompt_4k_str).unwrap();
    let m_4k = engine
        .generate_tokens(&prompt_4k_ids, 256, false)
        .expect("bench.py 4096:256");
    let accepted_4k = engine
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_acc_4k = if accepted_4k.is_empty() {
        0.0
    } else {
        accepted_4k.iter().sum::<usize>() as f64 / accepted_4k.len() as f64
    };
    eprintln!(
        "[REF-BENCH bench.py 4096:256] prompt_tokens={} completion_tokens={} ttft={:.2}s prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
        m_4k.prompt_len,
        m_4k.gen_len,
        m_4k.ttft_ms / 1000.0,
        m_4k.prefill_tok_per_s,
        m_4k.decode_tok_per_s,
        accepted_4k.len(),
        mean_acc_4k,
        1.0 + mean_acc_4k,
    );

    // 4. `scripts/bench.py` shape `8192:1024` (`offset = 5000 + 997 * 3 = 7991`, `n = 8192`, `TASKS[3]`)
    let window_8k: Vec<i32> = (0..8192)
        .map(|i| corpus_ids[(7991 + i) % corpus_ids.len()])
        .collect();
    let window_8k_text = engine.decode(&window_8k).unwrap();
    let task3 = "Write unit tests for the cache logic in this code, covering edge cases.";
    let prompt_8k_str = format!(
        "<|im_start|>user\n{window_8k_text}\n\n{task3}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let prompt_8k_ids = engine.encode(&prompt_8k_str).unwrap();
    let m_8k = engine
        .generate_tokens(&prompt_8k_ids, 1024, false)
        .expect("bench.py 8192:1024");
    let accepted_8k = engine
        .speculative
        .as_ref()
        .map(|p| p.last_accepted.clone())
        .unwrap_or_default();
    let mean_acc_8k = if accepted_8k.is_empty() {
        0.0
    } else {
        accepted_8k.iter().sum::<usize>() as f64 / accepted_8k.len() as f64
    };
    eprintln!(
        "[REF-BENCH bench.py 8192:1024] prompt_tokens={} completion_tokens={} ttft={:.2}s prefill={:.1} tok/s decode={:.2} tok/s rounds={} mean_accepted={:.2}/3 ({:.2} tok/round)",
        m_8k.prompt_len,
        m_8k.gen_len,
        m_8k.ttft_ms / 1000.0,
        m_8k.prefill_tok_per_s,
        m_8k.decode_tok_per_s,
        accepted_8k.len(),
        mean_acc_8k,
        1.0 + mean_acc_8k,
    );
}

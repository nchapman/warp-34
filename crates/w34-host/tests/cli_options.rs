//! CLI validation must fail before allocating any GPU/model resources.
#![cfg(feature = "cuda")]
use std::process::Command;
#[test]
fn speculative_cli_validation_without_a_visible_gpu() {
    for (args, error) in [
        (
            vec!["generate", "--speculative", "bad"],
            "--speculative expects mtp or dflash2",
        ),
        (
            vec!["generate", "--speculative", "dflash2"],
            "requires --draft-model",
        ),
        (
            vec![
                "generate",
                "--speculative",
                "mtp",
                "--draft-model",
                "unused",
            ],
            "native MTP uses target weights",
        ),
        (vec!["generate", "--mtp-k", "4"], "--mtp-k must be 1..3"),
        (
            vec!["generate", "--kv-pages", "0"],
            "--kv-pages must be positive",
        ),
        (
            vec!["bench", "--stage", "bad"],
            "--stage expects decode, radix, scheduler, vision, or all",
        ),
        (vec!["generate", "--image"], "missing --image value"),
        (
            vec!["generate", "--kv-format", "bad"],
            "unknown --kv-format",
        ),
        (
            vec!["generate", "--kv-format", "firstlast2"],
            "unknown --kv-format",
        ),
        (
            vec!["bench", "--batch-size", "0"],
            "--batch-size must be 1..1024",
        ),
        (
            vec!["bench", "--batch-size", "2048"],
            "--batch-size must be 1..1024",
        ),
        (
            vec!["greedy-diff", "--verify-lossless"],
            "--verify-lossless requires --speculative",
        ),
        (
            vec!["unknown-sub", "--speculative", "mtp"],
            "speculation is supported by",
        ),
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_w34"))
            .env("CUDA_VISIBLE_DEVICES", "")
            .args(args)
            .output()
            .unwrap();
        assert!(!out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(error), "{stderr}");
    }
    let out = Command::new(env!("CARGO_BIN_EXE_w34"))
        .env("CUDA_VISIBLE_DEVICES", "")
        .args(["generate", "--help"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--speculative mtp|dflash2"));
}

#[test]
fn cli_option_parsing_preserves_errors_and_help_short_circuit() {
    for (flag, error) in [
        ("--draft-model", "missing --draft-model"),
        ("--mtp-k", "missing --mtp-k"),
        ("--kv-pages", "missing --kv-pages"),
        ("--model", "missing --model value"),
        ("-m", "missing --model value"),
        ("--tp", "missing --tp value"),
        ("--prompt", "missing --prompt value"),
        ("-p", "missing --prompt value"),
        ("--max-tokens", "missing --max-tokens"),
        ("-n", "missing --max-tokens"),
        ("--prompt-len", "missing --prompt-len"),
        ("--gen-len", "missing --gen-len"),
        ("--kv-format", "missing --kv-format"),
        ("--golden", "missing --golden"),
        ("--out", "missing --out"),
        ("--max-prompts", "missing --max-prompts"),
        ("--oracle", "missing --oracle"),
        ("--preset", "missing --preset"),
        ("--w34", "missing --w34"),
        ("--rounds", "missing --rounds"),
        ("--host", "missing --host"),
        ("--port", "missing --port"),
        ("--stage", "missing --stage value"),
        ("--image", "missing --image value"),
        ("--batch-size", "missing --batch-size"),
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_w34"))
            .env("CUDA_VISIBLE_DEVICES", "")
            .args(["generate", flag])
            .output()
            .unwrap();
        assert!(!out.status.success(), "{flag}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(error),
            "{flag}: {out:?}"
        );
    }
    // Help is handled during parsing, before cross-option validation or loading.
    for args in [
        vec![],
        vec!["generate", "--mtp-k", "4", "--help"],
        vec!["generate", "--speculative", "dflash2", "-h"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_w34"))
            .env("CUDA_VISIBLE_DEVICES", "")
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("--speculative mtp|dflash2"));
    }
}

#[test]
fn profile_diff_presets_and_spec_classification() {
    for preset in [
        "vllm_4b_tp1_decode",
        "vllm_4b_tp1_mtp_k3",
        "sglang_4b_tp1_dflash2_m16",
        "vllm_27b_tp2_decode",
        "sglang_27b_tp2_dflash2_m8",
        "vllm_27b_tp1_tq4_decode",
        "vllm_35b_moe_tp1_decode",
        "vllm_35b_moe_tp2_decode",
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_w34"))
            .env("CUDA_VISIBLE_DEVICES", "")
            .args(["profile-diff", "--preset", preset])
            .output()
            .unwrap();
        assert!(out.status.success(), "{:?}", out);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("Kernel Scoreboard (ms/round)"), "{stdout}");
        assert!(stdout.contains("TOK/S (effective)"), "{stdout}");
    }

    let oracle_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/oracle");
    let py_check = Command::new("python3")
        .args([
            "-B",
            "-c",
            &format!(
                "import sys; sys.path.insert(0, {:?}); import profile_diff as p; \
                 assert p.classify_kernel('ampere_bf16_s16816gemm_bf16_128x64') == 'spec_decode'; \
                 assert p.classify_kernel('rms_norm_bf16_kernel') == 'spec_decode'; \
                 assert p.classify_kernel('tap_add_f16_to_bf16_kernel') == 'spec_decode'; \
                 assert p.classify_kernel('marlin_gemm_f16', in_spec_span=True) == 'spec_decode'; \
                 assert p.classify_kernel('marlin_gemm_f16', in_spec_span=False) == 'marlin_gemm'; \
                 assert p.classify_kernel('moe_w4a16_gate_up_silu_kernel') == 'marlin_gemm'; \
                 assert p.classify_kernel('moe_w4a16_down_accum_kernel') == 'marlin_gemm'; \
                 assert p.classify_kernel('moe_topk_router_f16_kernel') == 'norms_rope_act'; \
                 assert p.classify_kernel('vision_qkv_rope_attn_f16_kernel') == 'full_attn'; \
                 assert p.classify_kernel('embedding_splice_f16_kernel') == 'norms_rope_act'",
                oracle_dir.display().to_string()
            ),
        ])
        .output()
        .unwrap();
    assert!(py_check.status.success(), "{:?}", py_check);
}

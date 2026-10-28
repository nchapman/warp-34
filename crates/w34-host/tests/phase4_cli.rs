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
            "--stage expects decode, radix, scheduler, or all",
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
fn profile_diff_presets_and_spec_classification() {
    for preset in [
        "vllm_4b_tp1_decode",
        "vllm_4b_tp1_mtp_k3",
        "sglang_4b_tp1_dflash2_m16",
        "vllm_27b_tp2_decode",
        "sglang_27b_tp2_dflash2_m8",
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
                 assert p.classify_kernel('marlin_gemm_f16', in_spec_span=False) == 'marlin_gemm'",
                oracle_dir.display().to_string()
            ),
        ])
        .output()
        .unwrap();
    assert!(py_check.status.success(), "{:?}", py_check);
}

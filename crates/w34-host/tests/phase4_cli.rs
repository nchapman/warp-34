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
            vec!["dump-diff", "--speculative", "mtp"],
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

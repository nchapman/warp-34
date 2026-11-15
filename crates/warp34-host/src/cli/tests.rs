use super::*;

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn parse(values: &[&str]) -> CudaCliOptions {
    parse_cuda_options(&args(values)).unwrap().unwrap()
}

#[test]
fn serve_defaults_limits_and_alias() {
    let defaults = parse(&["serve"]);
    assert_eq!(defaults.serve.options.max_num_seqs, 16);
    assert!(defaults.serve.options.max_model_len.is_none());
    assert!(defaults.serve.options.api_key.is_none());
    for limit in ["1", "16"] {
        let prepared = prepare_cuda_cli_with_env(
            &args(&[
                "serve",
                "--model",
                "original",
                "--served-model-name",
                " alias ",
                "--max-num-seqs",
                limit,
                "--max-model-len",
                "123",
            ]),
            |model, _, _| {
                assert_eq!(model, "original");
                Ok("/cache/model".into())
            },
            |_| panic!("no ambient environment lookup"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            prepared.options.serve.options.max_num_seqs.to_string(),
            limit
        );
        assert_eq!(prepared.options.serve.options.max_model_len, Some(123));
    }
}

#[test]
fn serve_invalid_controls_never_resolve_or_read_environment() {
    for values in [
        vec!["--max-num-seqs", "0"],
        vec!["--max-num-seqs", "17"],
        vec!["--max-num-seqs", "x"],
        vec!["--max-model-len", "0"],
        vec!["--max-model-len", "-1"],
        vec!["--max-model-len", "x"],
        vec!["--served-model-name", "  "],
        vec!["--served-model-name", "a\nb"],
        vec!["--api-key", "secret", "--api-key-env", "KEY"],
    ] {
        let mut argv = args(&["serve"]);
        argv.extend(args(&values));
        assert!(
            prepare_cuda_cli_with_env(
                &argv,
                |_, _, _| panic!("download"),
                |_| panic!("environment")
            )
            .is_err()
        );
    }
    for flag in [
        "--api-key",
        "--api-key-env",
        "--served-model-name",
        "--max-num-seqs",
        "--max-model-len",
    ] {
        assert!(prepare_cuda_cli(&args(&["serve", flag]), |_, _, _| panic!("download")).is_err());
        let value = if flag.starts_with("--max-") {
            "1"
        } else {
            "value"
        };
        for sub in [
            "generate",
            "bench",
            "chat",
            "dump-diff",
            "profile-diff",
            "greedy-diff",
        ] {
            assert!(
                prepare_cuda_cli_with_env(
                    &args(&[sub, flag, value]),
                    |_, _, _| panic!("download"),
                    |_| panic!("environment")
                )
                .is_err()
            );
        }
    }
}

#[test]
fn serve_ignored_flags_are_rejected_before_download() {
    for (flag, value) in [
        ("--batch-size", Some("4")),
        ("--prompt", Some("text")),
        ("-p", Some("text")),
        ("--image", Some("image.png")),
        ("--max-tokens", Some("4")),
        ("-n", Some("4")),
        ("--no-thinking", None),
        ("--prompt-len", Some("4")),
        ("--gen-len", Some("4")),
        ("--stage", Some("decode")),
        ("--golden", Some("file")),
        ("--out", Some("file")),
        ("--max-prompts", Some("4")),
        ("--compare-tp2", None),
        ("--oracle", Some("file")),
        ("--preset", Some("key")),
        ("--warp34", Some("file")),
        ("--rounds", Some("4")),
        ("--verify-lossless", None),
        ("--verify-radix-warm", None),
        ("--live", None),
    ] {
        let mut argv = args(&["serve", flag]);
        argv.extend(value.map(str::to_string));
        let error = prepare_cuda_cli(&argv, |_, _, _| panic!("download for {flag}"))
            .err()
            .unwrap();
        assert!(error.contains(flag), "{error}");
    }
}

#[test]
fn serve_secret_lookup_is_explicit_once_and_after_validation() {
    let mut calls = 0;
    let prepared = prepare_cuda_cli_with_env(
        &args(&["serve", "--api-key-env", "KEY"]),
        |_, _, _| Ok("/cache/model".into()),
        |name| {
            calls += 1;
            assert_eq!(name, "KEY");
            Ok("secret-value".into())
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(calls, 1);
    assert!(prepared.options.serve.options.api_key.is_some());
    for value in [
        Ok(String::new()),
        Ok("bad\nsecret".into()),
        Err(env::VarError::NotPresent),
        Err(env::VarError::NotUnicode(std::ffi::OsString::from(
            "secret-value",
        ))),
    ] {
        let error = prepare_cuda_cli_with_env(
            &args(&["serve", "--api-key-env", "KEY"]),
            |_, _, _| panic!("download"),
            |_| value.clone(),
        )
        .err()
        .unwrap();
        assert!(!error.contains("secret"));
    }
    for argv in [
        vec!["serve", "--api-key-env", "KEY", "--help"],
        vec!["serve", "--api-key", "secret", "--api-key-env", "KEY", "-h"],
        vec!["serve", "--batch-size", "4", "--help"],
    ] {
        assert!(
            prepare_cuda_cli_with_env(
                &args(&argv),
                |_, _, _| panic!("download"),
                |_| panic!("environment")
            )
            .unwrap()
            .is_none()
        );
    }
    for value in ["", "bad\nsecret"] {
        let error = prepare_cuda_cli(&args(&["serve", "--api-key", value]), |_, _, _| {
            panic!("download")
        })
        .err()
        .unwrap();
        assert!(!error.contains("secret"));
    }
}

#[test]
fn serve_batch_size_rejected_before_download() {
    let result = prepare_cuda_cli(&args(&["serve", "--batch-size", "4"]), |_, _, _| {
        panic!("serve --batch-size must not download")
    });
    assert!(result.err().unwrap().contains("--max-num-seqs"));
}

#[test]
fn hub_options_default_and_parse_independently() {
    let defaults = parse(&["generate"]);
    assert!(defaults.revision.is_none());
    assert!(defaults.draft_revision.is_none());
    assert!(!defaults.offline);
    assert!(defaults.resolved_model.is_empty());

    let options = parse(&[
        "bench",
        "--revision",
        "refs/pr/12",
        "--draft-revision",
        "draft-commit",
        "--offline",
    ]);
    assert_eq!(options.revision.as_deref(), Some("refs/pr/12"));
    assert_eq!(options.draft_revision.as_deref(), Some("draft-commit"));
    assert!(options.offline);
    let target_only = parse(&["generate", "--revision", "target-commit"]);
    assert!(target_only.draft_revision.is_none());

    let aliased = parse(&[
        "serve",
        "--model-path",
        "org/aliased",
        "--tensor-parallel-size",
        "4",
        "--kv-cache-dtype",
        "bf16",
        "--enforce-eager",
    ]);
    assert_eq!(aliased.model, "org/aliased");
    assert_eq!(aliased.tp, 4);
    assert_eq!(aliased.kv_format, "bf16");
    assert!(!aliased.use_cuda_graph);

    let sglang_aliased = parse(&["serve", "--tp-size", "2", "--disable-cuda-graph"]);
    assert_eq!(sglang_aliased.tp, 2);
    assert!(!sglang_aliased.use_cuda_graph);
}

#[test]
fn revision_values_must_be_nonempty_and_not_options() {
    for flag in ["--revision", "--draft-revision"] {
        assert!(parse_cuda_options(&args(&["generate", flag])).is_err());
        for value in ["", "  ", "--offline", "--model", "-h"] {
            let error = parse_cuda_options(&args(&["generate", flag, value]))
                .err()
                .unwrap();
            assert_eq!(error, format!("missing {flag} value"));
        }
    }
}

#[test]
fn invalid_commands_and_options_never_resolve() {
    let cases: &[(&[&str], &str)] = &[
        (&["unknown"], "unknown subcommand"),
        (&["generate", "--unknown"], "unknown argument"),
        (&["generate", "--tp", "0"], "--tp must be 1, 2, or 4"),
        (&["generate", "--tp", "3"], "--tp must be 1, 2, or 4"),
        (&["generate", "--tp", "5"], "--tp must be 1, 2, or 4"),
        (&["greedy-diff"], "--golden"),
        (&["bench", "--batch-size", "0"], "--batch-size"),
        (&["generate", "--kv-format", "unknown"], "--kv-format"),
        (
            &["generate", "--revision", "--offline"],
            "missing --revision",
        ),
        (
            &["generate", "--draft-revision", "draft-ref"],
            "--draft-revision requires --draft-model",
        ),
        (
            &[
                "generate",
                "--draft-revision",
                "draft-ref",
                "--draft-model",
                "org/draft",
            ],
            "--draft-model requires --speculative dflash2",
        ),
        (
            &["generate", "--speculative", "dflash2"],
            "requires --draft-model",
        ),
    ];
    for (values, message) in cases {
        let result = prepare_cuda_cli(&args(values), |_, _, _| {
            panic!("invalid arguments must not resolve models: {values:?}")
        });
        assert!(result.err().unwrap().contains(message), "{values:?}");
    }
}

#[test]
fn help_never_resolves() {
    for values in [
        vec![],
        vec!["--help"],
        vec!["-h"],
        vec!["generate", "--help"],
        vec!["greedy-diff", "--tp", "3", "--help"],
    ] {
        let result = prepare_cuda_cli(&args(&values), |_, _, _| {
            panic!("help must not resolve models")
        });
        assert!(result.unwrap().is_none());
    }
}

#[test]
fn engine_commands_resolve_target_but_preserve_model_id() {
    for sub in [
        "generate",
        "bench",
        "greedy-diff",
        "dump-diff",
        "serve",
        "chat",
    ] {
        let mut calls = 0;
        let mut values = args(&[sub, "--model", "org/target", "--tp", "2"]);
        if sub == "greedy-diff" {
            values.extend(args(&["--golden", "golden.json"]));
        }
        let prepared = prepare_cuda_cli(&values, |spec, options, role| {
            calls += 1;
            assert_eq!(spec, "org/target");
            assert!(options.revision.is_none());
            assert!(!options.offline);
            assert!(matches!(role, ModelRole::Target));
            Ok("/cache/target".into())
        })
        .unwrap()
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(prepared.options.model, "org/target");
        assert_eq!(prepared.options.resolved_model, "/cache/target");
        assert!(prepared.spec_config.is_none());
    }
}

#[test]
fn draft_resolution_uses_its_own_revision_and_preserves_original_id() {
    for draft_revision in [None, Some("draft-ref")] {
        let mut values = args(&[
            "generate",
            "--model",
            "org/target",
            "--revision",
            "target-ref",
            "--speculative",
            "dflash2",
            "--draft-model",
            "org/draft",
            "--offline",
        ]);
        if let Some(revision) = draft_revision {
            values.extend(args(&["--draft-revision", revision]));
        }
        let mut calls = Vec::new();
        let prepared = prepare_cuda_cli(&values, |spec, options, role| {
            assert!(options.offline);
            let path = match role {
                ModelRole::Target => {
                    assert_eq!(spec, "org/target");
                    assert_eq!(options.revision.as_deref(), Some("target-ref"));
                    "/cache/target"
                }
                ModelRole::Draft => {
                    assert_eq!(spec, "org/draft");
                    assert_eq!(options.revision.as_deref(), draft_revision);
                    "/cache/draft"
                }
                ModelRole::Lora => unreachable!("unexpected Lora role in draft test"),
            };
            calls.push(spec.to_string());
            Ok(path.into())
        })
        .unwrap()
        .unwrap();
        assert_eq!(calls, ["org/target", "org/draft"]);
        assert_eq!(prepared.options.model, "org/target");
        assert_eq!(prepared.options.draft_model.as_deref(), Some("org/draft"));
        assert_eq!(prepared.options.resolved_model, "/cache/target");
        assert_eq!(
            prepared.spec_config.unwrap().draft_model.as_deref(),
            Some("/cache/draft")
        );
    }
}

#[test]
fn profile_comparison_and_live_parent_never_resolve() {
    for extra in [
        vec![],
        vec!["--live"],
        vec!["--warp34", "profile.json"],
        vec!["--warp34", "profile.json", "--live"],
    ] {
        let mut values = args(&[
            "profile-diff",
            "--revision",
            "target-ref",
            "--speculative",
            "dflash2",
            "--draft-model",
            "org/draft",
            "--draft-revision",
            "draft-ref",
            "--offline",
        ]);
        values.extend(args(&extra));
        let prepared = prepare_cuda_cli(&values, |_, _, _| {
            panic!("profile-diff must defer resolution to a nested live bench")
        })
        .unwrap()
        .unwrap();
        assert!(prepared.options.resolved_model.is_empty());
        assert!(prepared.spec_config.is_none());
    }
}

#[test]
fn live_bench_forwards_hub_options_and_original_ids() {
    let options = parse(&[
        "profile-diff",
        "--live",
        "--model",
        "org/target",
        "--revision",
        "target-ref",
        "--speculative",
        "dflash2",
        "--draft-model",
        "org/draft",
        "--draft-revision",
        "draft-ref",
        "--offline",
        "--kv-pages",
        "128",
    ]);
    let forwarded = profile_bench_args(&options, Path::new("/bin/warp34"));
    assert_eq!(&forwarded[..2], &["/bin/warp34", "bench"]);
    let nested = parse_cuda_options(&forwarded[1..]).unwrap().unwrap();
    assert!(validate_cuda_options(&nested, "bench").is_ok());
    assert_eq!(nested.model, "org/target");
    assert_eq!(nested.draft_model.as_deref(), Some("org/draft"));
    assert_eq!(nested.revision.as_deref(), Some("target-ref"));
    assert_eq!(nested.draft_revision.as_deref(), Some("draft-ref"));
    assert!(nested.offline);
    assert_eq!(nested.kv_pages, Some(128));

    let defaults = profile_bench_args(&parse(&["profile-diff", "--live"]), Path::new("warp34"));
    assert!(!defaults.iter().any(|arg| matches!(
        arg.as_str(),
        "--revision" | "--draft-revision" | "--offline"
    )));
}

#[test]
fn resolution_errors_propagate_before_dispatch() {
    let result = prepare_cuda_cli(&args(&["generate"]), |_, _, _| {
        Err(std::io::Error::other("offline cache miss"))
    });
    assert_eq!(result.err().unwrap(), "offline cache miss");
}

#[cfg(unix)]
#[test]
fn non_utf8_resolved_paths_are_rejected() {
    use std::os::unix::ffi::OsStringExt;
    let path = std::ffi::OsString::from_vec(vec![b'/', 0xff]);
    let result = prepare_cuda_cli(&args(&["generate"]), |_, _, _| {
        Ok(std::path::PathBuf::from(path.clone()))
    });
    assert!(result.err().unwrap().contains("not valid UTF-8"));
}

#[test]
fn kv_policy_parsing_variants() {
    for (name, expected) in [
        ("fp16", KvFormatPolicy::Fp16),
        ("bf16", KvFormatPolicy::Bf16),
        ("fp8", KvFormatPolicy::Fp8),
        ("fp8_e4m3", KvFormatPolicy::Fp8),
        ("q8_0", KvFormatPolicy::Q8_0),
        ("tq4", KvFormatPolicy::Tq4),
        ("tq4_all", KvFormatPolicy::Tq4All),
    ] {
        assert_eq!(parse_kv_policy(name).unwrap(), expected);
    }
    assert!(parse_kv_policy("invalid").is_err());
}

#[test]
fn kv_policy_parsing_and_tensor_diff_metrics() {
    let (max_abs, rel_l2, cos) = tensor_diff_metrics(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert_eq!(max_abs, 0.0);
    assert!(rel_l2 < 1e-12);
    assert!((cos - 1.0).abs() < 1e-12);

    let (max_abs_z, rel_z, cos_z) = tensor_diff_metrics(&[0.0, 0.0], &[0.0, 0.0]);
    assert_eq!(max_abs_z, 0.0);
    assert_eq!(rel_z, 0.0);
    assert_eq!(cos_z, 1.0);
    let (_, _, cos_one_zero) = tensor_diff_metrics(&[0.0, 0.0], &[1.0, 1.0]);
    assert_eq!(cos_one_zero, 0.0);

    let ref_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut mm_ok = ref_vals.clone();
    mm_ok[0] = 1.6;
    mm_ok[1] = 2.6;
    mm_ok[4] = 5.05;
    let (_, last_rel, min_cos) =
        evaluate_golden_tensor_diff("final_norm", &[3, 2], &mm_ok, &ref_vals, true).unwrap();
    assert!(last_rel < 0.05);
    assert!(min_cos > 0.99);
    assert!(evaluate_golden_tensor_diff("final_norm", &[3, 2], &mm_ok, &ref_vals, false).is_err());
    assert!(
        evaluate_golden_tensor_diff("layer_0_hidden_after", &[3, 2], &mm_ok, &ref_vals, true)
            .is_err()
    );
    let mut bad_mid_row = ref_vals.clone();
    bad_mid_row[2] = -3.0;
    bad_mid_row[3] = -4.0;
    assert!(
        evaluate_golden_tensor_diff("final_norm", &[3, 2], &bad_mid_row, &ref_vals, true).is_err()
    );
    assert!(evaluate_golden_tensor_diff("final_norm", &[2, 2], &mm_ok, &ref_vals, true).is_err());

    assert!(
        evaluate_moe_or_golden_tensor_diff("final_norm", &[3, 2], &mm_ok, &ref_vals, false, true)
            .is_ok()
    );
    assert!(
        evaluate_moe_or_golden_tensor_diff(
            "layer_0_hidden_after",
            &[3, 2],
            &mm_ok,
            &ref_vals,
            false,
            true,
        )
        .is_err()
    );
    assert!(
        evaluate_moe_or_golden_tensor_diff(
            "final_norm",
            &[3, 2],
            &bad_mid_row,
            &ref_vals,
            true,
            true,
        )
        .is_err()
    );

    // Even when full_rel_l2 <= 0.05 across a long sequence, a corrupted final row must fail.
    let long_ref = vec![1.0f32; 512 * 2];
    let mut zeroed_last_row = long_ref.clone();
    zeroed_last_row[511 * 2] = 0.0;
    zeroed_last_row[511 * 2 + 1] = 0.0;
    assert!(
        evaluate_moe_or_golden_tensor_diff(
            "final_norm",
            &[512, 2],
            &zeroed_last_row,
            &long_ref,
            true,
            true,
        )
        .is_err()
    );

    // Sublayer deltas return the actual measured last-row relative L2 error without clipping to 0.05.
    let sub_ref = vec![1.0f32, 2.0, 3.0, 4.0];
    let sub_vals = vec![1.0f32, 2.0, 3.6, 4.8]; // last row scaled by 1.2 -> last_rel_l2 = 0.20
    let (_, reported_sub_rel, _) = evaluate_moe_or_golden_tensor_diff(
        "layer_1_mlp_out",
        &[2, 2],
        &sub_vals,
        &sub_ref,
        false,
        true,
    )
    .unwrap();
    assert!((reported_sub_rel - 0.20).abs() < 1e-6);
}

#[cfg(feature = "cuda")]
#[test]
fn compare_tp1_vs_tp2_tensors_enforces_presence_and_shapes() {
    let mut tp1_map = std::collections::HashMap::new();
    tp1_map.insert(
        "final_norm".to_string(),
        (vec![1, 4], vec![1.0f32, 2.0, 3.0, 4.0]),
    );
    tp1_map.insert("logits".to_string(), (vec![1, 8], vec![0.0f32; 8]));
    tp1_map.insert("logits_last".to_string(), (vec![8], vec![0.0f32; 8]));
    let mut tp2_map = std::collections::HashMap::new();
    tp2_map.insert(
        "final_norm".to_string(),
        (vec![1, 4], vec![1.0f32, 2.0, 3.0, 4.0]),
    );
    tp2_map.insert("logits".to_string(), (vec![1, 4], vec![0.0f32; 4]));
    tp2_map.insert("logits_last".to_string(), (vec![4], vec![0.0f32; 4]));
    let summary = compare_tp1_vs_tp2_tensors(&tp1_map, &tp2_map, false, false).unwrap();
    assert_eq!(summary.max_layer_diff, 0.0);

    let mut tp2_missing = tp2_map.clone();
    tp2_missing.remove("final_norm");
    assert!(compare_tp1_vs_tp2_tensors(&tp1_map, &tp2_missing, false, false).is_err());

    let mut tp2_extra = tp2_map.clone();
    tp2_extra.insert(
        "layer_0_hidden_after".to_string(),
        (vec![1, 4], vec![1.0f32; 4]),
    );
    assert!(compare_tp1_vs_tp2_tensors(&tp1_map, &tp2_extra, false, false).is_err());

    let mut tp2_bad_shape = tp2_map;
    tp2_bad_shape.insert(
        "final_norm".to_string(),
        (vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0]),
    );
    assert!(compare_tp1_vs_tp2_tensors(&tp1_map, &tp2_bad_shape, false, false).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn greedy_diff_image_rejects_speculation() {
    let err = prepare_cuda_cli(
        &args(&[
            "greedy-diff",
            "--golden",
            "gold",
            "--image",
            "rgb://64x64/128,128,128",
            "--speculative",
            "mtp",
        ]),
        |_, _, _| panic!("greedy-diff --image --speculative must fail before download"),
    )
    .err()
    .unwrap();
    assert!(err.contains("--image does not support --speculative"));
}

#[cfg(feature = "cuda")]
#[test]
fn serve_state_slot_seqs_promotes_small_max_num_seqs_for_dflash2() {
    let dflash = crate::proposer::SpeculativeConfig {
        kind: crate::proposer::SpeculativeKind::DFlash2,
        draft_model: Some("draft".to_string()),
        mtp_k: 3,
    };
    let mtp = crate::proposer::SpeculativeConfig {
        kind: crate::proposer::SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    assert_eq!(serve_state_slot_seqs(1, Some(&dflash)), 6);
    assert_eq!(serve_state_slot_seqs(2, Some(&dflash)), 6);
    assert_eq!(serve_state_slot_seqs(8, Some(&dflash)), 8);
    assert_eq!(serve_state_slot_seqs(2, Some(&mtp)), 2);
    assert_eq!(serve_state_slot_seqs(2, None), 2);
}

#[cfg(feature = "cuda")]
#[test]
fn lora_cli_flags_resolve_and_validate() {
    let values = args(&[
        "serve",
        "--model",
        "org/target",
        "--lora-modules",
        "heretic=nchapman/Qwen3.5-9B-Heretic-LoRA",
        "trial3=nchapman/Qwen3.5-9B-Heretic-LoRA/pareto/trial_3@rev123",
    ]);
    let mut roles = Vec::new();
    let prepared = prepare_cuda_cli(&values, |spec, opts, role| {
        roles.push((spec.to_string(), opts.revision.clone(), role));
        Ok(format!("/cache/{}", spec.replace('/', "_")).into())
    })
    .unwrap()
    .unwrap();
    assert_eq!(roles.len(), 3);
    assert_eq!(roles[0].2, ModelRole::Target);
    assert_eq!(roles[1].2, ModelRole::Lora);
    assert_eq!(roles[1].0, "nchapman/Qwen3.5-9B-Heretic-LoRA");
    assert_eq!(roles[2].2, ModelRole::Lora);
    assert_eq!(roles[2].1.as_deref(), Some("rev123"));
    assert_eq!(prepared.options.lora_modules.len(), 2);
    assert_eq!(prepared.options.lora_modules[0].name, "heretic");
    assert_eq!(prepared.options.lora_modules[1].name, "trial3");

    // Duplicate adapter names are rejected before resolution:
    let dup = args(&[
        "serve",
        "--model",
        "org/target",
        "--lora-modules",
        "dup=org/a",
        "dup=org/b",
    ]);
    let err = prepare_cuda_cli(&dup, |_, _, _| panic!("must not resolve"))
        .err()
        .unwrap();
    assert!(err.contains("duplicate LoRA adapter name"));

    // Unknown active adapter name is rejected both with and without --lora-modules:
    let missing_no_modules = args(&["generate", "--model", "org/target", "--lora", "missing"]);
    let err_no_mod = prepare_cuda_cli(&missing_no_modules, |_, _, _| panic!("must not resolve"))
        .err()
        .unwrap();
    assert!(err_no_mod.contains("unknown active LoRA adapter 'missing'"));

    let missing_with_modules = args(&[
        "generate",
        "--model",
        "org/target",
        "--lora-modules",
        "a0=org/a0",
        "--active-lora",
        "missing",
    ]);
    let err_with_mod =
        prepare_cuda_cli(&missing_with_modules, |_, _, _| panic!("must not resolve"))
            .err()
            .unwrap();
    assert!(err_with_mod.contains("unknown active LoRA adapter 'missing'"));

    // profile-diff forwards --lora-modules (with @revision) and --active-lora to nested bench:
    let prof_args = args(&[
        "profile-diff",
        "--model",
        "org/target",
        "--lora-modules",
        "heretic=nchapman/Qwen3.5-9B-Heretic-LoRA@rev1",
        "--active-lora",
        "heretic",
    ]);
    let prof_prepared = prepare_cuda_cli(&prof_args, |_, _, _| panic!("must not resolve"))
        .unwrap()
        .unwrap();
    let bench_argv =
        profile_bench_args(&prof_prepared.options, std::path::Path::new("/bin/warp34"));
    assert!(bench_argv.windows(2).any(|w| w
        == [
            "--lora-modules",
            "heretic=nchapman/Qwen3.5-9B-Heretic-LoRA@rev1"
        ]));
    assert!(
        bench_argv
            .windows(2)
            .any(|w| w == ["--active-lora", "heretic"])
    );
}

#[cfg(feature = "cuda")]
#[test]
fn token_counts_ctx_size_and_serve_effective_kv_pages() {
    assert_eq!(parse_token_count("4096", "--ctx-size").unwrap(), 4096);
    assert_eq!(parse_token_count("32_768", "--ctx-size").unwrap(), 32_768);
    assert_eq!(parse_token_count("4k", "--ctx-size").unwrap(), 4096);
    assert_eq!(parse_token_count("64K", "--ctx-size").unwrap(), 65_536);
    assert_eq!(parse_token_count("128k", "--ctx-size").unwrap(), 131_072);
    assert_eq!(parse_token_count("1m", "--ctx-size").unwrap(), 1_048_576);
    for bad in ["", " ", "-4", "k", "16x"] {
        assert!(parse_token_count(bad, "--ctx-size").is_err(), "{bad}");
    }

    // --ctx-size / -c / --context-length / --kv-tokens convert tokens to 16-token pages.
    for flag in ["--ctx-size", "-c", "--context-length", "--kv-tokens"] {
        let opts = parse(&["serve", flag, "32k"]);
        assert_eq!(opts.kv_pages, Some(2048));
        assert!(validate_cuda_options(&opts, "serve").is_ok());
    }
    let rounded = parse(&["generate", "-c", "100"]);
    assert_eq!(rounded.kv_pages, Some(7));

    // --ctx-size and --kv-pages cannot be combined, and --ctx-size 0 is rejected before download.
    for bad_args in [
        &["serve", "--ctx-size", "0"][..],
        &["serve", "--ctx-size", "2m"][..],
        &["serve", "--kv-pages", "100000"][..],
        &["serve", "--max-model-len", "2m"][..],
        &["serve", "--ctx-size", "16k", "--kv-pages", "1024"][..],
        &["serve", "-c", "4k", "--max-model-len", "8k"][..],
        &["serve", "--kv-pages", "256", "--max-model-len", "8192"][..],
        &["chat", "-i", "--image", "test.png"][..],
    ] {
        let res = prepare_cuda_cli(&args(bad_args), |_, _, _| {
            panic!("invalid context sizing must fail before model resolution: {bad_args:?}")
        });
        assert!(res.is_err(), "{bad_args:?}");
    }

    // ServeCliConfig::effective_kv_pages auto-expands target KV pages when --max-model-len
    // exceeds the default serving budget or when --speculative is active.
    let mtp = crate::proposer::SpeculativeConfig {
        kind: crate::proposer::SpeculativeKind::Mtp,
        draft_model: None,
        mtp_k: 3,
    };
    assert_eq!(ServeCliConfig::effective_kv_pages(None, None, None), None);
    assert_eq!(
        ServeCliConfig::effective_kv_pages(None, Some(16_384), None),
        None
    );
    assert_eq!(
        ServeCliConfig::effective_kv_pages(None, Some(65_536), None),
        Some(4096)
    );
    assert_eq!(
        ServeCliConfig::effective_kv_pages(None, None, Some(&mtp)),
        Some(256)
    );
    assert_eq!(
        ServeCliConfig::effective_kv_pages(None, Some(16_384), Some(&mtp)),
        Some(1024)
    );
    assert_eq!(
        ServeCliConfig::effective_kv_pages(Some(512), Some(65_536), Some(&mtp)),
        Some(512)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn positional_model_industry_aliases_and_sampling_validation() {
    // Positional <model> works across subcommands.
    for sub in ["serve", "chat", "generate", "bench"] {
        let opts = parse(&[sub, "Qwen/Qwen3.5-4B"]);
        assert_eq!(opts.model, "Qwen/Qwen3.5-4B");
    }
    // Combining positional <model> with --model is rejected.
    assert!(parse_cuda_options(&args(&["serve", "org/pos", "--model", "org/flag"])).is_err());
    // Extra unexpected positional arguments still fail with "unknown argument".
    let extra_err = parse_cuda_options(&args(&["serve", "org/pos", "extra"]))
        .err()
        .unwrap();
    assert!(extra_err.contains("unknown argument: extra"));

    // Subcommand-aware default max_tokens.
    assert_eq!(parse(&["chat"]).max_tokens, 512);
    assert_eq!(parse(&["generate"]).max_tokens, 256);
    assert_eq!(parse(&["greedy-diff", "--golden", "g.json"]).max_tokens, 64);

    // Industry-standard flag aliases (vLLM, SGLang, llama.cpp).
    let serve_opts = parse(&[
        "serve",
        "-hf",
        "org/model",
        "--alias",
        "my-model",
        "--max-running-requests",
        "8",
        "--max-model-len",
        "64k",
        "--kv-cache-dtype",
        "auto",
        "--speculative-algorithm",
        "nextn",
        "--num-speculative-tokens",
        "2",
    ]);
    assert_eq!(serve_opts.model, "org/model");
    assert_eq!(serve_opts.serve.model_name.as_deref(), Some("my-model"));
    assert_eq!(serve_opts.serve.options.max_num_seqs, 8);
    assert_eq!(serve_opts.serve.options.max_model_len, Some(65_536));
    assert_eq!(
        validate_cuda_options(&serve_opts, "serve").unwrap(),
        KvFormatPolicy::Tq4
    );
    assert_eq!(
        serve_opts.speculative_kind,
        Some(crate::proposer::SpeculativeKind::Mtp)
    );
    assert_eq!(serve_opts.mtp_k, 2);

    // Sampling flags on chat/generate and pre-download validation.
    let gen_opts = parse(&[
        "generate",
        "Qwen/Qwen3.5-4B",
        "--system",
        "You are a concise Rust assistant.",
        "--max-new-tokens",
        "1k",
        "--temp",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--min-p",
        "0.05",
        "--seed",
        "42",
        "--repeat-penalty",
        "1.05",
        "--presence-penalty",
        "0.2",
        "--frequency-penalty",
        "0.1",
    ]);
    assert!(gen_opts.sampling.is_custom());
    assert_eq!(gen_opts.max_tokens, 1024);
    assert_eq!(
        gen_opts.system_prompt.as_deref(),
        Some("You are a concise Rust assistant.")
    );
    assert!(validate_cuda_options(&gen_opts, "generate").is_ok());

    for bad_sampling in [
        &["generate", "--temp", "-0.5"][..],
        &["generate", "--temp", "2.5"][..],
        &["generate", "--top-p", "0.0"][..],
        &["generate", "--repetition-penalty", "0.0"][..],
        &["generate", "--presence-penalty", "3.0"][..],
    ] {
        assert!(
            prepare_cuda_cli(&args(bad_sampling), |_, _, _| {
                panic!("invalid sampling must fail before download")
            })
            .is_err(),
            "{bad_sampling:?}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn qwen_image_cli_options_and_validation() {
    // Qwen-Image CLI subcommands (`image`, `bench-image`) support positional <model>,
    // `--size <WxH>`, `-o` / `--output`, `--num-inference-steps`, `--guidance-scale`,
    // `--transparent`, and unified `--seed`.
    let img_opts = parse(&[
        "image",
        "Qwen/Qwen-Image-2.1",
        "-p",
        "A translucent glass chess piece",
        "--neg-prompt",
        "blurry",
        "--size",
        "768x512",
        "--num-inference-steps",
        "28",
        "--guidance-scale",
        "4.0",
        "--seed",
        "777",
        "--transparent",
        "-o",
        "/tmp/chess.png",
    ]);
    assert_eq!(img_opts.model, "Qwen/Qwen-Image-2.1");
    assert_eq!(img_opts.tp, 2);
    assert_eq!(img_opts.image_width, Some(768));
    assert_eq!(img_opts.image_height, Some(512));
    assert_eq!(img_opts.image_steps, Some(28));
    assert_eq!(img_opts.cfg_scale, 4.0);
    assert_eq!(img_opts.seed, 777);
    assert_eq!(img_opts.sampling.seed, Some(777));
    assert!(img_opts.rgba_output);
    assert_eq!(img_opts.out_json.as_deref(), Some("/tmp/chess.png"));
    assert_eq!(img_opts.negative_prompt.as_deref(), Some("blurry"));
    assert!(validate_cuda_options(&img_opts, "image").is_ok());

    for bad_img in [
        &["image", "--size", "512"][..],
        &["image", "--width", "32"][..],
        &["image", "--height", "4096"][..],
        &["image", "--resolution", "16"][..],
        &["image", "--steps", "0"][..],
        &["image", "--steps", "200"][..],
        &["image", "--cfg-scale", "-1.0"][..],
        &["image", "--tp", "4"][..],
        &["bench-image", "--tp", "4"][..],
        &["serve", "--size", "512x512"][..],
        &["serve", "--steps", "8"][..],
        &["serve", "--rgba"][..],
        &["serve", "-o", "out.png"][..],
    ] {
        assert!(
            prepare_cuda_cli(&args(bad_img), |_, _, _| {
                panic!("invalid image CLI flag must fail before download: {bad_img:?}")
            })
            .is_err(),
            "{bad_img:?}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn model_kind_detection_and_cli_subcommand_guards() {
    use crate::loader::ModelKind;

    // 1. Image pipeline directory -> ModelKind::Image
    let img_dir = tempfile::tempdir().unwrap();
    fs::write(
        img_dir.path().join("model_index.json"),
        r#"{"_class_name": "QwenImage21Pipeline"}"#,
    )
    .unwrap();
    let img_path = img_dir.path().to_str().unwrap();
    assert_eq!(
        ModelKind::detect_dir(img_dir.path()).unwrap(),
        ModelKind::Image
    );
    assert_eq!(ModelKind::Image.as_str(), "image");
    assert!(
        super::commands::ensure_image_model_dir("Qwen/Qwen-Image-2.1", img_path, "image").is_ok()
    );
    assert!(
        super::commands::ensure_llm_model_dir("Qwen/Qwen-Image-2.1", img_path, "chat")
            .unwrap_err()
            .contains("ModelKind::Image")
    );

    // 2. LLM directory -> ModelKind::Llm
    let llm_dir = tempfile::tempdir().unwrap();
    fs::write(
        llm_dir.path().join("config.json"),
        r#"{"model_type": "qwen3_5", "hidden_size": 256}"#,
    )
    .unwrap();
    let llm_path = llm_dir.path().to_str().unwrap();
    assert_eq!(
        ModelKind::detect_dir(llm_dir.path()).unwrap(),
        ModelKind::Llm
    );
    assert_eq!(ModelKind::Llm.as_str(), "llm");
    assert!(super::commands::ensure_llm_model_dir("Qwen/Qwen3.5-4B", llm_path, "chat").is_ok());
    assert!(
        super::commands::ensure_image_model_dir("Qwen/Qwen3.5-4B", llm_path, "image")
            .unwrap_err()
            .contains("is a llm model")
    );

    // 3. Decision directory (via decision_config.json, readout.safetensors, and noncausal_full_attention)
    let dec_dir = tempfile::tempdir().unwrap();
    fs::write(
        dec_dir.path().join("config.json"),
        r#"{"model_type": "qwen3_5", "layer_types": ["noncausal_full_attention"]}"#,
    )
    .unwrap();
    assert_eq!(
        ModelKind::detect_dir(dec_dir.path()).unwrap(),
        ModelKind::Decision
    );
    assert_eq!(ModelKind::Decision.as_str(), "decision");

    let dec_dir2 = tempfile::tempdir().unwrap();
    fs::write(
        dec_dir2.path().join("config.json"),
        r#"{"model_type": "qwen3_5"}"#,
    )
    .unwrap();
    fs::write(dec_dir2.path().join("decision_config.json"), b"{}").unwrap();
    assert_eq!(
        ModelKind::detect_dir(dec_dir2.path()).unwrap(),
        ModelKind::Decision
    );

    // 4. Unsupported pipeline class and missing config fail cleanly
    let bad_pipe = tempfile::tempdir().unwrap();
    fs::write(
        bad_pipe.path().join("model_index.json"),
        r#"{"_class_name": "StableDiffusionXLPipeline"}"#,
    )
    .unwrap();
    assert!(ModelKind::detect_dir(bad_pipe.path()).is_err());

    let empty_dir = tempfile::tempdir().unwrap();
    assert!(ModelKind::detect_dir(empty_dir.path()).is_err());
    let empty_str = empty_dir.path().to_str().unwrap();
    assert!(super::commands::ensure_image_model_dir("empty", empty_str, "image").is_err());
    assert!(super::commands::ensure_llm_model_dir("empty", empty_str, "chat").is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn load_cli_condition_images_handles_paths_file_urls_data_uris_and_limits() {
    use crate::qwen_image::pipeline::encode_png_bytes;

    let tmp = tempfile::tempdir().unwrap();
    let png = encode_png_bytes(&[100u8; 4 * 4 * 4], 4, 4, 4).unwrap();
    let file_path = tmp.path().join("cond.png");
    fs::write(&file_path, &png).unwrap();

    let file_str = file_path.to_str().unwrap().to_string();
    let file_url = format!("file://{file_str}");
    let data_uri = format!(
        "data:image/png;base64,{}",
        crate::server::encode_base64(&png)
    );

    let loaded =
        super::commands::load_cli_condition_images(&[file_str.clone(), file_url, data_uri], false)
            .unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[0].width, 4);
    assert_eq!(loaded[1].width, 4);
    assert_eq!(loaded[2].width, 4);

    // Remote URL rejected when allow_remote_images = false
    assert!(
        super::commands::load_cli_condition_images(
            &["https://example.com/a.png".to_string()],
            false
        )
        .is_err()
    );
    // > 4 images rejected
    assert!(super::commands::load_cli_condition_images(&vec![file_str; 5], false).is_err());
    // Non-existent local file rejected
    assert!(
        super::commands::load_cli_condition_images(
            &["/nonexistent/warp34_missing.png".to_string()],
            false
        )
        .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_qwen_image_cli_commands_and_dump_diff_end_to_end() {
    use std::process::ExitCode;

    let _gpu_guard = crate::LIB_GPU_ENGINE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if crate::cuda::CudaDevice::count().unwrap_or(0) < 1 {
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("synth_qwen_image");
    crate::qwen_image::tests::write_synthetic_qwen_image_pipeline_dir(&model_dir);
    let model_str = model_dir.to_str().unwrap().to_string();

    let out_png = tmp.path().join("cli_out.png");
    let out_png_str = out_png.to_str().unwrap().to_string();

    // 1. `run_image` writes valid 64x64 PNG
    let mut img_opts = parse(&[
        "image",
        &model_str,
        "--tp",
        "1",
        "-p",
        "synth fox",
        "--size",
        "64x64",
        "--resolution",
        "64",
        "--steps",
        "2",
        "-o",
        &out_png_str,
    ]);
    img_opts.resolved_model = model_str.clone();
    assert_eq!(
        super::commands::run_image(img_opts).unwrap(),
        ExitCode::SUCCESS
    );
    let decoded = crate::qwen_image::pipeline::QwenImage21ConditionImage::from_image_bytes(
        &fs::read(&out_png).unwrap(),
    )
    .unwrap();
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);

    // 2. `run_bench_image` writes JSON report to `--out`
    let bench_json_path = tmp.path().join("bench_image.json");
    let bench_json_str = bench_json_path.to_str().unwrap().to_string();
    let mut bench_opts = parse(&[
        "bench-image",
        &model_str,
        "--tp",
        "1",
        "-p",
        "synth fox",
        "--size",
        "64x64",
        "--resolution",
        "64",
        "--steps",
        "2",
        "--rounds",
        "1",
        "--out",
        &bench_json_str,
    ]);
    bench_opts.resolved_model = model_str.clone();
    assert_eq!(
        super::commands::run_bench_image(bench_opts).unwrap(),
        ExitCode::SUCCESS
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(&bench_json_path).unwrap()).unwrap();
    assert_eq!(report["tp"], 1);
    assert_eq!(report["width"], 64);
    assert_eq!(report["height"], 64);

    // 3. `run_dump_diff` dispatches to `run_qwen_image_dump_diff` and verifies `--golden <png>`
    let dump_json_path = tmp.path().join("dump_diff.json");
    let dump_json_str = dump_json_path.to_str().unwrap().to_string();
    let mut dump_opts = parse(&[
        "dump-diff",
        &model_str,
        "--tp",
        "1",
        "-p",
        "synth fox",
        "--size",
        "64x64",
        "--resolution",
        "64",
        "--steps",
        "2",
        "--golden",
        &out_png_str,
        "--out",
        &dump_json_str,
    ]);
    dump_opts.resolved_model = model_str.clone();
    assert_eq!(
        super::oracle::run_dump_diff(dump_opts, super::KvFormatPolicy::Fp16, None).unwrap(),
        ExitCode::SUCCESS
    );

    // 4. `run_chat` and `run_greedy_diff` reject `ModelKind::Image` before loading LLM weights
    let mut chat_opts = parse(&["chat", &model_str]);
    chat_opts.resolved_model = model_str.clone();
    assert!(
        super::commands::run_chat(chat_opts, super::KvFormatPolicy::Fp16, None)
            .unwrap_err()
            .contains("ModelKind::Image")
    );

    let mut greedy_opts = parse(&["greedy-diff", &model_str, "--golden", &out_png_str]);
    greedy_opts.resolved_model = model_str;
    assert!(
        super::oracle::run_greedy_diff(greedy_opts, super::KvFormatPolicy::Fp16, None)
            .unwrap_err()
            .contains("ModelKind::Image")
    );
}

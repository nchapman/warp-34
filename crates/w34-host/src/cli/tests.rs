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
        ("--w34", Some("file")),
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
        vec!["--w34", "profile.json"],
        vec!["--w34", "profile.json", "--live"],
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
    let forwarded = profile_bench_args(&options, Path::new("/bin/w34"));
    assert_eq!(&forwarded[..2], &["/bin/w34", "bench"]);
    let nested = parse_cuda_options(&forwarded[1..]).unwrap().unwrap();
    assert!(validate_cuda_options(&nested, "bench").is_ok());
    assert_eq!(nested.model, "org/target");
    assert_eq!(nested.draft_model.as_deref(), Some("org/draft"));
    assert_eq!(nested.revision.as_deref(), Some("target-ref"));
    assert_eq!(nested.draft_revision.as_deref(), Some("draft-ref"));
    assert!(nested.offline);
    assert_eq!(nested.kv_pages, Some(128));

    let defaults = profile_bench_args(&parse(&["profile-diff", "--live"]), Path::new("w34"));
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
fn kv_policy_parsing_and_tensor_diff_metrics() {
    assert_eq!(parse_kv_policy("fp16").unwrap(), KvFormatPolicy::Fp16);
    assert_eq!(parse_kv_policy("bf16").unwrap(), KvFormatPolicy::Bf16);
    assert_eq!(parse_kv_policy("q8_0").unwrap(), KvFormatPolicy::Q8_0);
    assert_eq!(parse_kv_policy("tq4").unwrap(), KvFormatPolicy::Tq4);
    assert_eq!(parse_kv_policy("tq4_all").unwrap(), KvFormatPolicy::Tq4All);
    assert!(parse_kv_policy("invalid").is_err());

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
}

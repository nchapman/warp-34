#[cfg(feature = "cuda")]
use super::*;

#[cfg(feature = "cuda")]
pub(crate) fn parse_kv_policy(s: &str) -> Result<KvFormatPolicy, String> {
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

pub fn print_usage() {
    eprintln!(
        "Usage: w34 <subcommand> [options]\n\
         Subcommands:\n\
           generate     --model <id|path> [--tp 1|2|4] [--prompt <str>] [--image <path|url>]... [--max-tokens <n>] [--kv-format tq4|tq4_all|q8_0|fp16]\n\
           bench        --model <id|path> [--tp 1|2|4] [--stage decode|radix|scheduler|vision|all] [--batch-size <n>] [--prompt-len <n>] [--gen-len <n>] [--kv-format tq4|tq4_all|q8_0|fp16] [--out <json>]\n\
           greedy-diff  --model <id|path> --golden <dir|json> [--tp 1|2|4] [--max-prompts <n>] [--max-tokens <n>] [--kv-format tq4|tq4_all|q8_0|fp16] [--verify-lossless] [--verify-radix-warm]\n\
           dump-diff    --model <id|path> [--tp 1|2|4] [--prompt <str>] [--image <path|url>]... [--compare-tp2] [--golden <safetensors>]\n\
           profile-diff [--oracle <sqlite|json>] [--preset <key>] [--w34 <sqlite|json>] [--live] [--rounds <n>]\n\
           serve        --model <id|path> [--tp 1|2|4] [--host <addr>] [--port <n>] [--kv-format tq4|tq4_all|q8_0|fp16]\n\
           chat         --model <id|path> [--tp 1|2|4] [--prompt <str>] [--image <path|url>]... [--max-tokens <n>] [--no-thinking]\n\
         Serve-only: --api-key <value> OR --api-key-env <name> (explicit lookup; no ambient default).\n\
           Protects API routes; dashboard, metrics and health remain public. Default: no authentication.\n\
           --served-model-name <name> advertises an alias (default: original model ID).\n\
           --max-num-seqs <1..16> (default 16; fixed graph limit).\n\
           --max-model-len <positive integer> caps total prompt + output admission only (default: engine capacity); does not size VRAM.\n\
         Hub options: --revision <ref> [--offline]; --draft-revision <ref> requires --draft-model.\n\
         Image options: --allow-remote-images enables public HTTP(S) fetch (16 MiB, 10 s, no redirects).\n\
         Speculative options (generate/bench/greedy-diff/dump-diff/profile-diff/serve/chat):\n\
           --speculative mtp|dflash2 [--draft-model <id|path>] [--mtp-k 1..3] [--kv-pages <n>]\n\
           Native MTP shares target weights; DFlash2 requires a draft model.\n\
           Speculative CLI mode is greedy (CUDA Graph verify by default; pass --no-cuda-graph for eager); serve also honors sampling parameters.\n\
           Default --kv-format is tq4 (rotated Q8_0 boundary layers + TQ4 interior). Native mode defaults to 256 target KV pages (4096 tokens)."
    );
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct ServeCliConfig {
    pub(crate) options: crate::server::ServeOptions,
    pub(crate) model_name: Option<String>,
    pub(crate) api_key: Option<String>,
    pub(crate) api_key_env: Option<String>,
    pub(crate) used: bool,
    pub(crate) rejected: Option<String>,
}

#[cfg(feature = "cuda")]
impl ServeCliConfig {
    fn parse(&mut self, args: &[String], i: &mut usize) -> Result<bool, String> {
        let flag = args[*i].as_str();
        if !matches!(
            flag,
            "--api-key"
                | "--api-key-env"
                | "--served-model-name"
                | "--max-num-seqs"
                | "--max-model-len"
        ) {
            if matches!(
                flag,
                "--batch-size"
                    | "--prompt"
                    | "-p"
                    | "--image"
                    | "--max-tokens"
                    | "-n"
                    | "--no-thinking"
                    | "--prompt-len"
                    | "--gen-len"
                    | "--stage"
                    | "--golden"
                    | "--out"
                    | "--max-prompts"
                    | "--compare-tp2"
                    | "--oracle"
                    | "--preset"
                    | "--w34"
                    | "--rounds"
                    | "--verify-lossless"
                    | "--verify-radix-warm"
                    | "--live"
            ) {
                self.rejected.get_or_insert_with(|| flag.to_string());
            }
            return Ok(false);
        }
        self.used = true;
        let value = revision_value(args, i)?;
        match flag {
            "--api-key" => self.api_key = Some(value),
            "--api-key-env" => self.api_key_env = Some(value),
            "--served-model-name" => self.model_name = Some(value),
            "--max-num-seqs" => {
                self.options.max_num_seqs = value.parse().map_err(|_| "invalid --max-num-seqs")?
            }
            "--max-model-len" => {
                self.options.max_model_len =
                    Some(value.parse().map_err(|_| "invalid --max-model-len")?)
            }
            _ => unreachable!(),
        }
        Ok(true)
    }

    fn validate(&self, sub: &str) -> Result<(), String> {
        if sub != "serve" && self.used {
            return Err("API key, served model name and admission options require serve".into());
        }
        if sub == "serve"
            && let Some(flag) = &self.rejected
        {
            return Err(if flag == "--batch-size" {
                "--batch-size is not supported by serve; use --max-num-seqs".into()
            } else {
                format!("{flag} is not supported by serve")
            });
        }
        if self.api_key.is_some() && self.api_key_env.is_some() {
            return Err("--api-key and --api-key-env are mutually exclusive".into());
        }
        if let Some(name) = &self.model_name
            && (name.trim().is_empty() || name.chars().any(char::is_control))
        {
            return Err(
                "--served-model-name must be nonempty and contain no control characters".into(),
            );
        }
        self.options.validate()
    }

    fn resolve_key(
        &mut self,
        mut lookup: impl FnMut(&str) -> Result<String, env::VarError>,
    ) -> Result<(), String> {
        let value = match self.api_key.take() {
            Some(value) => Some(value),
            None => self
                .api_key_env
                .take()
                .map(|name| {
                    lookup(&name).map_err(|_| {
                        "--api-key-env must name a set Unicode environment variable".to_string()
                    })
                })
                .transpose()?,
        };
        self.options.api_key = value
            .map(crate::api_auth::ApiKey::new)
            .transpose()
            .map_err(|_| "invalid API key".to_string())?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaCliOptions {
    pub(crate) serve: ServeCliConfig,
    pub(crate) model: String,
    // Populated centrally after validation; never used as the public model name.
    pub(crate) resolved_model: String,
    pub(crate) revision: Option<String>,
    pub(crate) draft_revision: Option<String>,
    pub(crate) offline: bool,
    pub(crate) tp: usize,
    pub(crate) prompt: String,
    pub(crate) max_tokens: usize,
    pub(crate) prompt_len: usize,
    pub(crate) gen_len: usize,
    pub(crate) kv_format: String,
    pub(crate) golden_path: Option<String>,
    pub(crate) out_json: Option<String>,
    pub(crate) max_prompts: usize,
    pub(crate) use_cuda_graph: bool,
    pub(crate) compare_tp2: bool,
    pub(crate) oracle_path: Option<String>,
    pub(crate) preset_key: Option<String>,
    pub(crate) w34_prof_path: Option<String>,
    pub(crate) prof_rounds: usize,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) enable_thinking: bool,
    pub(crate) speculative_kind: Option<crate::proposer::SpeculativeKind>,
    pub(crate) draft_model: Option<String>,
    pub(crate) mtp_k: usize,
    pub(crate) kv_pages: Option<usize>,
    pub(crate) bench_stage: String,
    pub(crate) batch_size: usize,
    pub(crate) verify_lossless: bool,
    pub(crate) verify_radix_warm: bool,
    pub(crate) live_profile: bool,
    pub(crate) image_urls: Vec<String>,
    pub(crate) allow_remote_images: bool,
}

#[cfg(feature = "cuda")]
pub(crate) fn revision_value(args: &[String], index: &mut usize) -> Result<String, String> {
    let flag = &args[*index];
    *index += 1;
    args.get(*index)
        .filter(|value| !value.trim().is_empty() && !value.starts_with('-'))
        .cloned()
        .ok_or_else(|| format!("missing {flag} value"))
}

// None means help was requested; stop parsing immediately, before validation.
#[cfg(feature = "cuda")]
pub(crate) fn parse_cuda_options(args: &[String]) -> Result<Option<CudaCliOptions>, String> {
    let mut serve = ServeCliConfig::default();
    let mut model = "nchapman/Qwen3.5-4B-W4A16-GPTQ".to_string();
    let mut revision = None;
    let mut draft_revision = None;
    let mut offline = false;
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
        if serve.parse(args, &mut i)? {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--revision" => revision = Some(revision_value(args, &mut i)?),
            "--draft-revision" => draft_revision = Some(revision_value(args, &mut i)?),
            "--offline" => offline = true,
            "--speculative" => {
                i += 1;
                speculative_kind = Some(match args.get(i).map(String::as_str) {
                    Some("mtp") => crate::proposer::SpeculativeKind::Mtp,
                    Some("dflash2") => crate::proposer::SpeculativeKind::DFlash2,
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
                return Ok(None);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    Ok(Some(CudaCliOptions {
        serve,
        model,
        resolved_model: String::new(),
        revision,
        draft_revision,
        offline,
        tp,
        prompt,
        max_tokens,
        prompt_len,
        gen_len,
        kv_format,
        golden_path,
        out_json,
        max_prompts,
        use_cuda_graph,
        compare_tp2,
        oracle_path,
        preset_key,
        w34_prof_path,
        prof_rounds,
        host,
        port,
        enable_thinking,
        speculative_kind,
        draft_model,
        mtp_k,
        kv_pages,
        bench_stage,
        batch_size,
        verify_lossless,
        verify_radix_warm,
        live_profile,
        image_urls,
        allow_remote_images,
    }))
}

#[cfg(feature = "cuda")]
pub(crate) fn validate_cuda_command(options: &CudaCliOptions, sub: &str) -> Result<(), String> {
    if !matches!(
        sub,
        "generate" | "bench" | "greedy-diff" | "dump-diff" | "profile-diff" | "serve" | "chat"
    ) {
        if options.speculative_kind.is_some() {
            return Err("speculation is supported by generate, bench, greedy-diff, dump-diff, profile-diff, serve and chat".into());
        }
        return Err(format!("unknown subcommand: {sub}"));
    }
    if !matches!(options.tp, 1 | 2 | 4) {
        return Err("--tp must be 1, 2, or 4".into());
    }
    if sub == "greedy-diff" && options.golden_path.is_none() {
        return Err("--golden <dir-or-json> is required".into());
    }
    if options.draft_revision.is_some() && options.draft_model.is_none() {
        return Err("--draft-revision requires --draft-model".into());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn validate_cuda_options(
    options: &CudaCliOptions,
    sub: &str,
) -> Result<KvFormatPolicy, String> {
    let CudaCliOptions {
        kv_format,
        kv_pages,
        mtp_k,
        draft_model,
        speculative_kind,
        verify_lossless,
        ..
    } = options;
    options.serve.validate(sub)?;
    let policy = parse_kv_policy(kv_format)?;
    if *kv_pages == Some(0) {
        return Err("--kv-pages must be positive".into());
    }
    if !matches!(mtp_k, 1..=3) {
        return Err("--mtp-k must be 1..3".into());
    }
    if draft_model.is_some() && *speculative_kind != Some(crate::proposer::SpeculativeKind::DFlash2)
    {
        return Err(
            "--draft-model requires --speculative dflash2; native MTP uses target weights".into(),
        );
    }
    if *speculative_kind == Some(crate::proposer::SpeculativeKind::DFlash2) && draft_model.is_none()
    {
        return Err("--speculative dflash2 requires --draft-model".into());
    }
    if *verify_lossless && speculative_kind.is_none() {
        return Err("--verify-lossless requires --speculative".into());
    }
    if options.max_prompts == 0 {
        return Err("--max-prompts must be >= 1".into());
    }
    validate_cuda_command(options, sub)?;
    Ok(policy)
}

#[cfg(feature = "cuda")]
pub(crate) fn load_cli_engine(
    model: &str,
    tp: usize,
    policy: KvFormatPolicy,
    kv_pages: Option<usize>,
    use_cuda_graph: bool,
    spec_config: &Option<crate::proposer::SpeculativeConfig>,
) -> crate::cuda::Result<Qwen3_5Engine> {
    // Leave VRAM for the resident drafter instead of filling it with target KV.
    // Speculative taps/logits are configured in enable_speculative before lazy graph capture.
    let pages = kv_pages.or_else(|| spec_config.as_ref().map(|_| 256));
    let mut engine = Qwen3_5Engine::load(model, tp, policy, pages, use_cuda_graph)?;
    if let Some(config) = spec_config {
        engine.enable_speculative(config)?;
    }
    Ok(engine)
}

#[cfg(feature = "cuda")]
pub(crate) struct PreparedCudaCli {
    pub(crate) options: CudaCliOptions,
    pub(crate) policy: KvFormatPolicy,
    pub(crate) spec_config: Option<crate::proposer::SpeculativeConfig>,
}

#[cfg(feature = "cuda")]
pub(crate) fn engine_path(path: std::path::PathBuf) -> Result<String, String> {
    path.into_os_string()
        .into_string()
        .map_err(|path| format!("resolved model path is not valid UTF-8: {path:?}"))
}

// Keep validation and resolution together so no command can download before
// argument checks, or initialize a GPU before all model paths are available.
#[cfg(feature = "cuda")]
pub(crate) fn prepare_cuda_cli(
    args: &[String],
    resolve: impl FnMut(&str, &HubOptions, ModelRole) -> std::io::Result<std::path::PathBuf>,
) -> Result<Option<PreparedCudaCli>, String> {
    prepare_cuda_cli_with_env(args, resolve, |name| env::var(name))
}

#[cfg(feature = "cuda")]
pub(crate) fn prepare_cuda_cli_with_env(
    args: &[String],
    mut resolve: impl FnMut(&str, &HubOptions, ModelRole) -> std::io::Result<std::path::PathBuf>,
    lookup: impl FnMut(&str) -> Result<String, env::VarError>,
) -> Result<Option<PreparedCudaCli>, String> {
    let Some(sub) = args.first().map(String::as_str) else {
        print_usage();
        return Ok(None);
    };
    if matches!(sub, "--help" | "-h") {
        print_usage();
        return Ok(None);
    }
    let Some(mut options) = parse_cuda_options(args)? else {
        return Ok(None);
    };
    let policy = validate_cuda_options(&options, sub)?;
    options.serve.resolve_key(lookup)?;
    // Comparison is local. Live profiling resolves in the nested bench command,
    // not here (and --w34 takes precedence over --live).
    if sub == "profile-diff" {
        return Ok(Some(PreparedCudaCli {
            options,
            policy,
            spec_config: None,
        }));
    }
    options.resolved_model = engine_path(
        resolve(
            &options.model,
            &HubOptions {
                revision: options.revision.clone(),
                offline: options.offline,
            },
            ModelRole::Target,
        )
        .map_err(|e| e.to_string())?,
    )?;
    let draft_model = options
        .draft_model
        .as_ref()
        .map(|model| {
            engine_path(
                resolve(
                    model,
                    &HubOptions {
                        revision: options.draft_revision.clone(),
                        offline: options.offline,
                    },
                    ModelRole::Draft,
                )
                .map_err(|e| e.to_string())?,
            )
        })
        .transpose()?;
    let spec_config = options
        .speculative_kind
        .map(|kind| crate::proposer::SpeculativeConfig {
            kind,
            draft_model,
            mtp_k: options.mtp_k,
        });
    Ok(Some(PreparedCudaCli {
        options,
        policy,
        spec_config,
    }))
}

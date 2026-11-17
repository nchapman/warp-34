#[cfg(feature = "cuda")]
use super::*;

#[cfg(feature = "cuda")]
pub(crate) fn parse_kv_policy(s: &str) -> Result<KvFormatPolicy, String> {
    match s.to_ascii_lowercase().as_str() {
        "fp16" | "f16" | "half" => Ok(KvFormatPolicy::Fp16),
        "bf16" | "bfloat16" => Ok(KvFormatPolicy::Bf16),
        "fp8" | "fp8_e4m3" | "fp8-e4m3" => Ok(KvFormatPolicy::Fp8),
        "q8_0" | "q8" => Ok(KvFormatPolicy::Q8_0),
        "tq4" | "auto" => Ok(KvFormatPolicy::Tq4),
        "tq4_all" | "tq4-all" => Ok(KvFormatPolicy::Tq4All),
        other => Err(format!(
            "unknown --kv-format '{other}' (expected tq4, tq4_all, fp8, q8_0, fp16, bf16, or auto)"
        )),
    }
}

/// Parses a token count or page count, accepting plain integers (`4096`), underscore
/// separators (`32_768`), or binary `k`/`K` (`× 1024`) and `m`/`M` (`× 1_048_576`) suffixes
/// (`4k` -> `4096`, `32k` -> `32768`, `128k` -> `131072`).
#[cfg(feature = "cuda")]
pub(crate) fn parse_token_count(raw: &str, flag: &str) -> Result<usize, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.starts_with('-') {
        return Err(format!("invalid {flag}"));
    }
    let cleaned: String = trimmed.chars().filter(|&c| c != '_').collect();
    if cleaned.is_empty() {
        return Err(format!("invalid {flag}"));
    }
    let (num_str, multiplier) = match cleaned.as_bytes().last().copied() {
        Some(b'k' | b'K') => (&cleaned[..cleaned.len() - 1], 1024usize),
        Some(b'm' | b'M') => (&cleaned[..cleaned.len() - 1], 1024usize * 1024),
        _ => (cleaned.as_str(), 1usize),
    };
    if num_str.is_empty() {
        return Err(format!("invalid {flag}"));
    }
    let base = num_str
        .parse::<usize>()
        .map_err(|_| format!("invalid {flag}"))?;
    base.checked_mul(multiplier)
        .ok_or_else(|| format!("invalid {flag}"))
}

/// Resolves `--tp auto` from `CUDA_VISIBLE_DEVICES` without initializing CUDA before Hub downloads.
#[cfg(feature = "cuda")]
pub(crate) fn parse_tp_arg(raw: &str) -> Result<usize, String> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("auto") {
        let visible = env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
        let count = visible
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != "-1")
            .count();
        return Ok(if count >= 4 {
            4
        } else if count >= 2 {
            2
        } else {
            1
        });
    }
    trimmed.parse::<usize>().map_err(|_| "invalid --tp".into())
}

pub fn print_usage() {
    eprintln!("{USAGE}");
}

pub(crate) const USAGE: &str = r#"Usage: warp34 <subcommand> [model] [options]

Subcommands:
  serve        [model] Start OpenAI/Anthropic HTTP server & embedded dashboard (default :8000)
  chat         [model] Interactive multi-turn chat REPL or single-turn chat (--prompt)
  generate     [model] Single-prompt text/multimodal generation
  bench        [model] Throughput & latency benchmark (--stage decode|radix|scheduler|vision|all)
  greedy-diff  [model] Token-level parity check against oracle golden JSON (--golden <dir|json>)
  dump-diff    [model] Layer-by-layer tensor parity check (--compare-tp2 | --golden <safetensors>)
  profile-diff         Nsight Systems SQLite kernel category comparison vs vLLM/SGLang

Model & Hub Options:
  <model>, -m, --model <id|path>      Hugging Face repo ID or local Safetensors directory
                                      (default: nchapman/Qwen3.5-4B-W4A16-HGPTQ)
  --revision <rev>                    Hub branch, tag, or full 40-char commit SHA
  --draft-revision <rev>              Hub revision for --draft-model (requires --draft-model)
  --offline                           Use local Hugging Face cache only (also respects HF_HUB_OFFLINE=1)
  --lora-modules <name=path|path>...  Preload up to 62 hot-swappable LoRA adapters (supports @rev)
  --lora, --active-lora <name|spec>   Activate a LoRA adapter for single-sequence CLI commands

Context & KV Cache Options:
  -c, --ctx-size <tokens>             Target KV pool capacity in tokens (default: auto up to 32k serve,
                                      16k chat, 4k generate; supports k/M suffixes: 4k, 16k, 32k, 64k, 128k, 256k;
                                      aliases: --context-length, --kv-tokens)
  --kv-format <policy>                tq4 (default, 3.07x), tq4_all (3.88x), fp8, q8_0, fp16, bf16, auto
                                      (aliases: --kv-cache-dtype, --cache-type-k)
  --kv-pages <n>                      Target KV pool in 16-token pages (low-level alternative to --ctx-size)

Parallelism & Speculative Decoding:
  --tp, --tp-size <1|2|4|auto>        Tensor parallelism degree (alias: --tensor-parallel-size)
  --speculative mtp|dflash2|none      Speculative decoding mode (native MTP or DFlash2 block drafter)
  --draft-model <id|path>             DFlash2 draft checkpoint (required for --speculative dflash2)
  --mtp-k <1..3>                      Draft tokens per MTP round (default: 3; alias: --num-speculative-tokens)
  --no-cuda-graph, --enforce-eager    Disable CUDA Graph capture and run eager launches

Serving Options (warp34 serve):
  --host <addr>                       Bind address (default: 127.0.0.1; select an interface for remote access)
  --port <n>                          Listen port (default: 8000)
  --max-num-seqs <1..16>              Max concurrent active sequences (default: 16; alias: --max-running-requests)
  --max-model-len <tokens>            Max total prompt + output tokens per request (supports 16k, 64k, etc.;
                                      auto-expands KV pool when --ctx-size/--kv-pages is omitted)
  --served-model-name, --alias <name> Public model ID advertised on /v1/models (default: model ID)
  --api-key <val> | --api-key-env <v> Optional bearer token auth for all API operations, including aliases (not only /v1)
                                      Metrics (/metrics and /api/metrics), minimal health, and enabled webui remain unauthenticated
  --no-webui                         Disable the web UI only; metrics and API remain available

Generation & Chat Options (warp34 chat / generate):
  -p, --prompt <text>                 Prompt text (omit in terminal with `warp34 chat` for interactive REPL)
  -f, --prompt-file <path>            Read prompt text from a UTF-8 file
  --system, --system-prompt <text>    System prompt prepended to the conversation
  -i, --interactive                   Force interactive multi-turn REPL mode in `warp34 chat`
  -n, --max-tokens <tokens>           Max generated tokens (default: 2048 chat, 512 generate, 64 greedy-diff)
  --temp, --temperature <0..2>        Sampling temperature (0 = greedy default)
  --top-p <0..1> / --top-k <n>        Nucleus (top-p) and top-k sampling filters
  --min-p <0..1> / --seed <u64>       Min-p filter threshold and deterministic RNG seed
  --repetition-penalty <float>        Repetition penalty (> 0.0; default: 1.0)
  --presence-penalty <float>          Presence penalty (-2.0..2.0; default: 0.0)
  --frequency-penalty <float>         Frequency penalty (-2.0..2.0; default: 0.0)
  --no-thinking / --thinking          Disable or enable <think> reasoning blocks (enabled by default in chat)
  --image <path|url>...               Attach image(s) for multimodal VL models (--allow-remote-images for URLs)

Benchmark Options (warp34 bench):
  --stage <decode|radix|scheduler|vision|all>  Benchmark stage (default: decode)
  --prompt-len <tokens>               Prompt length in tokens (default: 128; supports k/M suffixes)
  --gen-len <tokens>                  Generated output length in tokens (default: 128; supports k/M suffixes)
  --realistic-prompt, --realistic     Use realistic coding/systems prompt instead of synthetic repetition
  --batch-size <1..1024>              Concurrent batch size for scheduler stage (default: 4)
  --out <path>                        Write benchmark JSON summary to file

Examples:
  warp34 serve Qwen/Qwen3.5-4B -c 64k --speculative mtp
  warp34 serve nchapman/Qwen3.5-27B-W4A16-AutoRound --tp 2 -c 32k --speculative dflash2 --draft-model z-lab/Qwen3.5-27B-DFlash
  warp34 chat Qwen/Qwen3.5-4B --speculative mtp
  warp34 generate Qwen/Qwen3.5-4B -p "Write a lock-free ring buffer in Rust" -n 512 --temp 0.6"#;

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub(crate) struct CliSamplingOptions {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<i32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) seed: Option<u64>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
}

#[cfg(feature = "cuda")]
impl CliSamplingOptions {
    pub(crate) fn is_custom(&self) -> bool {
        self.temperature.is_some()
            || self.top_p.is_some()
            || self.top_k.is_some()
            || self.min_p.is_some()
            || self.seed.is_some()
            || self.repetition_penalty.is_some()
            || self.presence_penalty.is_some()
            || self.frequency_penalty.is_some()
    }

    pub(crate) fn to_sampling_params(
        &self,
        max_tokens: usize,
        eos_token_ids: &[i32],
    ) -> crate::parser::SamplingParams {
        let mut params = crate::parser::SamplingParams {
            max_tokens,
            stop_token_ids: eos_token_ids.to_vec(),
            eos_token_ids: eos_token_ids.to_vec(),
            ..Default::default()
        };
        if let Some(t) = self.temperature {
            params.temperature = t;
        }
        if let Some(p) = self.top_p {
            params.top_p = p;
        }
        if let Some(k) = self.top_k {
            params.top_k = k;
        }
        if let Some(mp) = self.min_p {
            params.min_p = mp;
        }
        if let Some(s) = self.seed {
            params.seed = s;
        }
        if let Some(rp) = self.repetition_penalty {
            params.repetition_penalty = rp;
        }
        if let Some(pp) = self.presence_penalty {
            params.presence_penalty = pp;
        }
        if let Some(fp) = self.frequency_penalty {
            params.frequency_penalty = fp;
        }
        params
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let params = self.to_sampling_params(1, &[]);
        params.validate().map_err(|e| e.to_string())
    }
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
    /// Derives the target KV page request for `warp34 serve`:
    /// - When `--ctx-size` or `--kv-pages` was explicitly set, honors `kv_pages`.
    /// - When `--speculative` is enabled, pins the serving KV page request to
    ///   `max_model_len.div_ceil(16)` or [`crate::model::DEFAULT_SERVING_TARGET_KV_PAGES`] (`2,048` pages = `32,768` tokens).
    /// - Otherwise, if `--max-model-len` was set and requires more pages than the default,
    ///   auto-expands the target KV page request to accommodate `max_model_len`.
    pub(crate) fn effective_kv_pages(
        kv_pages: Option<usize>,
        max_model_len: Option<usize>,
        spec_config: Option<&crate::proposer::SpeculativeConfig>,
    ) -> Option<usize> {
        if let Some(explicit) = kv_pages {
            return Some(explicit);
        }
        let needed_for_max_len = max_model_len.map(|tokens| tokens.div_ceil(16));
        if spec_config.is_some() {
            Some(
                needed_for_max_len
                    .unwrap_or(crate::model::DEFAULT_SERVING_TARGET_KV_PAGES)
                    .max(crate::model::DEFAULT_TARGET_KV_PAGES),
            )
        } else if let Some(needed) = needed_for_max_len
            && needed > crate::model::DEFAULT_SERVING_TARGET_KV_PAGES
        {
            Some(needed)
        } else {
            None
        }
    }

    fn parse(&mut self, args: &[String], i: &mut usize) -> Result<bool, String> {
        let flag = args[*i].as_str();
        if flag == "--no-webui" {
            self.used = true;
            self.options.webui = false;
            return Ok(true);
        }
        if !matches!(
            flag,
            "--api-key"
                | "--api-key-env"
                | "--served-model-name"
                | "--alias"
                | "-a"
                | "--max-num-seqs"
                | "--max-running-requests"
                | "--parallel"
                | "-np"
                | "--max-model-len"
        ) {
            if matches!(
                flag,
                "--batch-size"
                    | "--prompt"
                    | "-p"
                    | "--prompt-file"
                    | "-f"
                    | "--system"
                    | "--system-prompt"
                    | "-sys"
                    | "--interactive"
                    | "-i"
                    | "--image"
                    | "--max-tokens"
                    | "--max-new-tokens"
                    | "-n"
                    | "--no-thinking"
                    | "--disable-thinking"
                    | "--thinking"
                    | "--enable-thinking"
                    | "--temperature"
                    | "--temp"
                    | "--top-p"
                    | "--top-k"
                    | "--min-p"
                    | "--seed"
                    | "--repetition-penalty"
                    | "--repeat-penalty"
                    | "--presence-penalty"
                    | "--frequency-penalty"
                    | "--prompt-len"
                    | "--gen-len"
                    | "--realistic-prompt"
                    | "--stage"
                    | "--golden"
                    | "--out"
                    | "--max-prompts"
                    | "--compare-tp2"
                    | "--oracle"
                    | "--preset"
                    | "--warp34"
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
            "--served-model-name" | "--alias" | "-a" => self.model_name = Some(value),
            "--max-num-seqs" | "--max-running-requests" | "--parallel" | "-np" => {
                self.options.max_num_seqs = value.parse().map_err(|_| "invalid --max-num-seqs")?
            }
            "--max-model-len" => {
                self.options.max_model_len = Some(parse_token_count(&value, "--max-model-len")?)
            }
            _ => unreachable!(),
        }
        Ok(true)
    }

    fn validate(&self, sub: &str) -> Result<(), String> {
        if sub != "serve" && self.used {
            return Err(
                "API key, webui, served model name and admission options require serve".into(),
            );
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
    pub(crate) prompt_explicit: bool,
    pub(crate) prompt_file: Option<String>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) interactive: bool,
    pub(crate) sampling: CliSamplingOptions,
    pub(crate) max_tokens: usize,
    pub(crate) prompt_len: usize,
    pub(crate) gen_len: usize,
    pub(crate) realistic_prompt: bool,
    pub(crate) kv_format: String,
    pub(crate) golden_path: Option<String>,
    pub(crate) out_json: Option<String>,
    pub(crate) max_prompts: usize,
    pub(crate) use_cuda_graph: bool,
    pub(crate) compare_tp2: bool,
    pub(crate) oracle_path: Option<String>,
    pub(crate) preset_key: Option<String>,
    pub(crate) warp34_prof_path: Option<String>,
    pub(crate) prof_rounds: usize,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) enable_thinking: bool,
    pub(crate) speculative_kind: Option<crate::proposer::SpeculativeKind>,
    pub(crate) draft_model: Option<String>,
    pub(crate) mtp_k: usize,
    pub(crate) kv_pages: Option<usize>,
    pub(crate) ctx_tokens: Option<usize>,
    pub(crate) explicit_kv_pages: bool,
    pub(crate) bench_stage: String,
    pub(crate) batch_size: usize,
    pub(crate) verify_lossless: bool,
    pub(crate) verify_radix_warm: bool,
    pub(crate) live_profile: bool,
    pub(crate) image_urls: Vec<String>,
    pub(crate) allow_remote_images: bool,
    pub(crate) lora_modules: Vec<crate::lora::LoraModuleSpec>,
    pub(crate) active_lora: Option<String>,
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
    let sub = args.first().map(String::as_str).unwrap_or("");
    let mut serve = ServeCliConfig::default();
    let mut model = "nchapman/Qwen3.5-4B-W4A16-HGPTQ".to_string();
    let mut explicit_model_flag = false;
    let mut positional_model: Option<String> = None;
    let mut revision = None;
    let mut draft_revision = None;
    let mut offline = false;
    let mut tp = 1usize;
    let mut prompt =
        "Explain the difference between a mutex and a semaphore in two sentences.".to_string();
    let mut prompt_explicit = false;
    let mut prompt_file: Option<String> = None;
    let mut system_prompt: Option<String> = None;
    let mut interactive = false;
    let mut sampling = CliSamplingOptions::default();
    let mut max_tokens = match sub {
        "chat" => 2048usize,
        "generate" => 512usize,
        _ => 64usize,
    };
    let mut prompt_len = 128usize;
    let mut gen_len = 128usize;
    let mut realistic_prompt = false;
    let mut kv_format = "tq4".to_string();
    let mut golden_path: Option<String> = None;
    let mut out_json: Option<String> = None;
    let mut max_prompts = 12usize;
    let mut use_cuda_graph = true;
    let mut compare_tp2 = false;
    let mut oracle_path: Option<String> = None;
    let mut preset_key: Option<String> = None;
    let mut warp34_prof_path: Option<String> = None;
    let mut prof_rounds = 128usize;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8000u16;
    let mut enable_thinking = true;
    let mut speculative_kind = None;
    let mut draft_model = None;
    let mut mtp_k = 3usize;
    let mut kv_pages = None;
    let mut ctx_tokens = None;
    let mut explicit_kv_pages = false;
    let mut bench_stage = "decode".to_string();
    let mut batch_size = 8usize;
    let mut verify_lossless = false;
    let mut verify_radix_warm = false;
    let mut live_profile = false;
    let mut image_urls: Vec<String> = Vec::new();
    let mut allow_remote_images = false;
    let mut lora_modules: Vec<crate::lora::LoraModuleSpec> = Vec::new();
    let mut active_lora: Option<String> = None;

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
            "--speculative" | "--speculative-algorithm" => {
                i += 1;
                speculative_kind = match args.get(i).map(|s| s.to_ascii_lowercase()).as_deref() {
                    Some("mtp" | "nextn") => Some(crate::proposer::SpeculativeKind::Mtp),
                    Some("dflash2" | "dflash") => Some(crate::proposer::SpeculativeKind::DFlash2),
                    Some("none" | "off" | "disabled") => None,
                    _ => return Err("--speculative expects mtp or dflash2".into()),
                };
            }
            "--draft-model" | "--speculative-draft-model-path" | "--model-draft" | "-md" => {
                i += 1;
                draft_model = Some(args.get(i).cloned().ok_or("missing --draft-model")?);
            }
            "--mtp-k"
            | "--num-speculative-tokens"
            | "--draft-tokens"
            | "--speculative-num-draft-tokens" => {
                i += 1;
                mtp_k = args
                    .get(i)
                    .ok_or("missing --mtp-k")?
                    .parse()
                    .map_err(|_| "invalid --mtp-k")?;
            }
            "--ctx-size" | "-c" | "--context-length" | "--kv-tokens" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --ctx-size")?;
                ctx_tokens = Some(parse_token_count(raw, "--ctx-size")?);
            }
            "--kv-pages" | "--num-gpu-blocks-override" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --kv-pages")?;
                kv_pages = Some(parse_token_count(raw, "--kv-pages")?);
                explicit_kv_pages = true;
            }
            "--model" | "-m" | "--model-path" | "--hf-repo" | "-hf" => {
                i += 1;
                model = args.get(i).cloned().ok_or("missing --model value")?;
                explicit_model_flag = true;
            }
            "--tp" | "-tp" | "--tp-size" | "--tensor-parallel-size" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --tp value")?;
                tp = parse_tp_arg(raw)?;
            }
            "--prompt" | "-p" => {
                i += 1;
                prompt = args.get(i).cloned().ok_or("missing --prompt value")?;
                prompt_explicit = true;
            }
            "--prompt-file" | "-f" => {
                i += 1;
                prompt_file = Some(args.get(i).cloned().ok_or("missing --prompt-file value")?);
                prompt_explicit = true;
            }
            "--system" | "--system-prompt" | "-sys" => {
                i += 1;
                system_prompt = Some(args.get(i).cloned().ok_or("missing --system value")?);
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            "--max-tokens" | "-n" | "--max-new-tokens" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --max-tokens")?;
                max_tokens = parse_token_count(raw, "--max-tokens")?;
            }
            "--temperature" | "--temp" => {
                i += 1;
                sampling.temperature = Some(
                    args.get(i)
                        .ok_or("missing --temperature")?
                        .parse()
                        .map_err(|_| "invalid --temperature")?,
                );
            }
            "--top-p" => {
                i += 1;
                sampling.top_p = Some(
                    args.get(i)
                        .ok_or("missing --top-p")?
                        .parse()
                        .map_err(|_| "invalid --top-p")?,
                );
            }
            "--top-k" => {
                i += 1;
                sampling.top_k = Some(
                    args.get(i)
                        .ok_or("missing --top-k")?
                        .parse()
                        .map_err(|_| "invalid --top-k")?,
                );
            }
            "--min-p" => {
                i += 1;
                sampling.min_p = Some(
                    args.get(i)
                        .ok_or("missing --min-p")?
                        .parse()
                        .map_err(|_| "invalid --min-p")?,
                );
            }
            "--seed" => {
                i += 1;
                sampling.seed = Some(
                    args.get(i)
                        .ok_or("missing --seed")?
                        .parse()
                        .map_err(|_| "invalid --seed")?,
                );
            }
            "--repetition-penalty" | "--repeat-penalty" => {
                i += 1;
                sampling.repetition_penalty = Some(
                    args.get(i)
                        .ok_or("missing --repetition-penalty")?
                        .parse()
                        .map_err(|_| "invalid --repetition-penalty")?,
                );
            }
            "--presence-penalty" => {
                i += 1;
                sampling.presence_penalty = Some(
                    args.get(i)
                        .ok_or("missing --presence-penalty")?
                        .parse()
                        .map_err(|_| "invalid --presence-penalty")?,
                );
            }
            "--frequency-penalty" => {
                i += 1;
                sampling.frequency_penalty = Some(
                    args.get(i)
                        .ok_or("missing --frequency-penalty")?
                        .parse()
                        .map_err(|_| "invalid --frequency-penalty")?,
                );
            }
            "--prompt-len" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --prompt-len")?;
                prompt_len = parse_token_count(raw, "--prompt-len")?;
            }
            "--gen-len" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --gen-len")?;
                gen_len = parse_token_count(raw, "--gen-len")?;
            }
            "--realistic-prompt" | "--realistic" => {
                realistic_prompt = true;
            }
            "--kv-format" | "--kv-cache-dtype" | "--cache-type-k" => {
                i += 1;
                kv_format = args.get(i).cloned().ok_or("missing --kv-format")?;
            }
            "--golden" => {
                i += 1;
                golden_path = Some(args.get(i).cloned().ok_or("missing --golden")?);
            }
            "--out" | "--out-json" | "-o" => {
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
            "--no-cuda-graph" | "--enforce-eager" | "--disable-cuda-graph" => {
                use_cuda_graph = false;
            }
            "--cuda-graph" => {
                use_cuda_graph = true;
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
            "--warp34" => {
                i += 1;
                warp34_prof_path = Some(args.get(i).cloned().ok_or("missing --warp34")?);
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
            "--port" | "-P" => {
                i += 1;
                port = args
                    .get(i)
                    .ok_or("missing --port")?
                    .parse()
                    .map_err(|_| "invalid --port")?;
            }
            "--no-thinking" | "--disable-thinking" => {
                enable_thinking = false;
            }
            "--thinking" | "--enable-thinking" => {
                enable_thinking = true;
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
            "--lora-modules" | "--lora-module" => {
                let mut count = 0usize;
                while i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    let spec =
                        crate::lora::LoraModuleSpec::parse(&args[i]).map_err(|e| e.to_string())?;
                    lora_modules.push(spec);
                    count += 1;
                }
                if count == 0 {
                    return Err("missing --lora-modules value".into());
                }
            }
            "--lora" => {
                i += 1;
                let val = args.get(i).cloned().ok_or("missing --lora value")?;
                if val.starts_with('-') || val.trim().is_empty() {
                    return Err("missing --lora value".into());
                }
                if val.contains('=') || val.contains('/') || val.contains('.') || val.contains('{')
                {
                    let spec =
                        crate::lora::LoraModuleSpec::parse(&val).map_err(|e| e.to_string())?;
                    active_lora = Some(spec.name.clone());
                    lora_modules.push(spec);
                } else {
                    active_lora = Some(val);
                }
            }
            "--active-lora" => {
                i += 1;
                let val = args.get(i).cloned().ok_or("missing --active-lora value")?;
                if val.starts_with('-') || val.trim().is_empty() {
                    return Err("missing --active-lora value".into());
                }
                active_lora = Some(val);
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
            other if !other.starts_with('-') && positional_model.is_none() => {
                positional_model = Some(other.to_string());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    if let Some(pos_model) = positional_model {
        if explicit_model_flag {
            return Err("cannot specify both positional <model> and --model".into());
        }
        model = pos_model;
    }

    if let Some(tokens) = ctx_tokens
        && !explicit_kv_pages
        && tokens > 0
    {
        kv_pages = Some(tokens.div_ceil(16));
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
        prompt_explicit,
        prompt_file,
        system_prompt,
        interactive,
        sampling,
        max_tokens,
        prompt_len,
        gen_len,
        realistic_prompt,
        kv_format,
        golden_path,
        out_json,
        max_prompts,
        use_cuda_graph,
        compare_tp2,
        oracle_path,
        preset_key,
        warp34_prof_path,
        prof_rounds,
        host,
        port,
        enable_thinking,
        speculative_kind,
        draft_model,
        mtp_k,
        kv_pages,
        ctx_tokens,
        explicit_kv_pages,
        bench_stage,
        batch_size,
        verify_lossless,
        verify_radix_warm,
        live_profile,
        image_urls,
        allow_remote_images,
        lora_modules,
        active_lora,
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
    if sub == "greedy-diff"
        && !options.image_urls.is_empty()
        && (options.speculative_kind.is_some() || options.verify_lossless)
    {
        return Err(
            "greedy-diff with --image does not support --speculative or --verify-lossless".into(),
        );
    }
    if options.draft_revision.is_some() && options.draft_model.is_none() {
        return Err("--draft-revision requires --draft-model".into());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) const MAX_CLI_KV_PAGES: usize = 65_536;
#[cfg(feature = "cuda")]
pub(crate) const MAX_CLI_KV_TOKENS: usize = MAX_CLI_KV_PAGES * 16;

#[cfg(feature = "cuda")]
pub(crate) fn bench_required_stage_tokens(stage: &str, prompt_len: usize, gen_len: usize) -> usize {
    let decode_tokens = prompt_len.saturating_add(gen_len.max(1));
    let radix_prompt_len = (prompt_len.max(128) / 64)
        .saturating_mul(64)
        .saturating_add(8);
    let radix_tokens = radix_prompt_len.saturating_add(gen_len.clamp(4, 32));
    match stage {
        "decode" => decode_tokens,
        "radix" => radix_tokens,
        "all" => decode_tokens.max(radix_tokens),
        _ => 0,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn validate_cuda_options(
    options: &CudaCliOptions,
    sub: &str,
) -> Result<KvFormatPolicy, String> {
    let CudaCliOptions {
        kv_format,
        kv_pages,
        ctx_tokens,
        explicit_kv_pages,
        mtp_k,
        draft_model,
        speculative_kind,
        verify_lossless,
        sampling,
        ..
    } = options;
    options.serve.validate(sub)?;
    let policy = parse_kv_policy(kv_format)?;
    if ctx_tokens.is_some() && *explicit_kv_pages {
        return Err(
            "--ctx-size and --kv-pages are mutually exclusive; specify context in tokens (--ctx-size) or 16-token pages (--kv-pages)".into(),
        );
    }
    if *ctx_tokens == Some(0) {
        return Err("--ctx-size must be positive".into());
    }
    if let Some(tokens) = *ctx_tokens
        && tokens > MAX_CLI_KV_TOKENS
    {
        return Err(format!(
            "--ctx-size ({tokens} tokens) exceeds maximum supported context ({MAX_CLI_KV_TOKENS} tokens / 1M)"
        ));
    }
    if *kv_pages == Some(0) {
        return Err("--kv-pages must be positive".into());
    }
    if let Some(pages) = *kv_pages
        && pages > MAX_CLI_KV_PAGES
    {
        return Err(format!(
            "--kv-pages ({pages}) exceeds maximum supported capacity ({MAX_CLI_KV_PAGES} pages / {MAX_CLI_KV_TOKENS} tokens)"
        ));
    }
    if sub == "serve"
        && let Some(max_len) = options.serve.options.max_model_len
    {
        if max_len > MAX_CLI_KV_TOKENS {
            return Err(format!(
                "--max-model-len ({max_len} tokens) exceeds maximum supported context ({MAX_CLI_KV_TOKENS} tokens / 1M)"
            ));
        }
        if let Some(pages) = *kv_pages {
            let cap_tokens = pages.saturating_mul(16);
            if max_len > cap_tokens {
                return Err(format!(
                    "--max-model-len ({max_len} tokens) exceeds allocated KV capacity ({cap_tokens} tokens from {pages} pages; need at least --ctx-size {max_len} or --kv-pages {})",
                    max_len.div_ceil(16)
                ));
            }
        }
    }
    sampling.validate()?;
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
    if *speculative_kind == Some(crate::proposer::SpeculativeKind::DFlash2) {
        let max_dflash_tokens = crate::dflash::DFLASH2_MAX_KV_CAPACITY;
        let max_dflash_pages = max_dflash_tokens / 16;
        if let Some(pages) = *kv_pages
            && pages > max_dflash_pages
        {
            return Err(format!(
                "--speculative dflash2 supports at most {max_dflash_tokens} tokens ({max_dflash_pages} KV pages); got {} tokens ({pages} pages)",
                pages.saturating_mul(16)
            ));
        }
        if sub == "serve"
            && let Some(max_len) = options.serve.options.max_model_len
            && max_len > max_dflash_tokens
        {
            return Err(format!(
                "--max-model-len ({max_len} tokens) exceeds --speculative dflash2 maximum capacity ({max_dflash_tokens} tokens / {max_dflash_pages} pages)"
            ));
        }
        if sub == "bench" {
            let bench_tokens = bench_required_stage_tokens(
                &options.bench_stage,
                options.prompt_len,
                options.gen_len,
            );
            if bench_tokens > max_dflash_tokens {
                return Err(format!(
                    "bench stage '{}' requires {bench_tokens} tokens (--prompt-len {} + --gen-len {}), which exceeds --speculative dflash2 maximum capacity ({max_dflash_tokens} tokens / {max_dflash_pages} pages)",
                    options.bench_stage, options.prompt_len, options.gen_len
                ));
            }
        }
    }
    if sub == "bench" {
        if options.prompt_len == 0 {
            return Err("--prompt-len must be >= 1".into());
        }
        if options.gen_len == 0 {
            return Err("--gen-len must be >= 1".into());
        }
        let total_bench_tokens =
            bench_required_stage_tokens(&options.bench_stage, options.prompt_len, options.gen_len);
        if total_bench_tokens > MAX_CLI_KV_TOKENS {
            return Err(format!(
                "bench stage '{}' requires {total_bench_tokens} tokens (--prompt-len {} + --gen-len {}), which exceeds maximum supported context ({MAX_CLI_KV_TOKENS} tokens / 1M)",
                options.bench_stage, options.prompt_len, options.gen_len
            ));
        }
        if let Some(pages) = *kv_pages {
            let cap_tokens = pages.saturating_mul(16);
            if total_bench_tokens > cap_tokens {
                return Err(format!(
                    "bench stage '{}' requires {total_bench_tokens} tokens (--prompt-len {} + --gen-len {}), which exceeds allocated KV capacity ({cap_tokens} tokens from {pages} pages; need at least --ctx-size {total_bench_tokens} or --kv-pages {})",
                    options.bench_stage,
                    options.prompt_len,
                    options.gen_len,
                    total_bench_tokens.div_ceil(16)
                ));
            }
        }
    }
    if *verify_lossless && speculative_kind.is_none() {
        return Err("--verify-lossless requires --speculative".into());
    }
    if options.interactive && !options.image_urls.is_empty() {
        return Err("--interactive cannot be combined with --image".into());
    }
    if options.max_prompts == 0 {
        return Err("--max-prompts must be >= 1".into());
    }
    if options.lora_modules.len() > crate::lora::MAX_LORA_ADAPTERS {
        return Err(format!(
            "at most {} LoRA adapters can be loaded simultaneously",
            crate::lora::MAX_LORA_ADAPTERS
        ));
    }
    for (idx, a) in options.lora_modules.iter().enumerate() {
        if options.lora_modules[..idx].iter().any(|b| b.name == a.name) {
            return Err(format!("duplicate LoRA adapter name '{}'", a.name));
        }
    }
    if let Some(ref active) = options.active_lora
        && !options.lora_modules.iter().any(|m| &m.name == active)
    {
        return Err(format!("unknown active LoRA adapter '{active}'"));
    }
    validate_cuda_command(options, sub)?;
    Ok(policy)
}

#[cfg(feature = "cuda")]
pub(crate) fn attach_cli_lora(
    engine: &mut Qwen3_5Engine,
    lora_modules: &[crate::lora::LoraModuleSpec],
    active_lora: Option<&str>,
    activate_for_single_seq: bool,
) -> crate::cuda::Result<()> {
    if lora_modules.is_empty() {
        if let Some(name) = active_lora {
            return Err(crate::model::loader_err_msg(format!(
                "unknown active LoRA adapter '{name}'"
            )));
        }
        return Ok(());
    }
    if engine.lora_registry.is_empty() {
        engine.attach_lora_adapters(lora_modules)?;
    }
    if activate_for_single_seq {
        let lid = if let Some(name) = active_lora {
            engine.lora_registry.resolve_name(name).ok_or_else(|| {
                crate::model::loader_err_msg(format!("unknown active LoRA adapter '{name}'"))
            })?
        } else {
            0
        };
        engine.set_active_lora_id(lid);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_cli_engine_with_lora(
    model: &str,
    tp: usize,
    policy: KvFormatPolicy,
    kv_pages: Option<usize>,
    use_cuda_graph: bool,
    spec_config: &Option<crate::proposer::SpeculativeConfig>,
    lora_modules: &[crate::lora::LoraModuleSpec],
    active_lora: Option<&str>,
) -> crate::cuda::Result<Qwen3_5Engine> {
    // Leave VRAM for the resident drafter instead of filling it with target KV.
    // Speculative taps/logits are configured in enable_speculative before lazy graph capture.
    let pages = kv_pages.or_else(|| spec_config.as_ref().map(|_| 256));
    let mut engine =
        Qwen3_5Engine::load_with_lora(model, tp, policy, pages, use_cuda_graph, lora_modules)?;
    attach_cli_lora(&mut engine, lora_modules, active_lora, true)?;
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
    if matches!(sub, "--help" | "-h" | "help") {
        print_usage();
        return Ok(None);
    }
    if matches!(sub, "--version" | "-V" | "version") {
        println!("warp34 {}", env!("CARGO_PKG_VERSION"));
        return Ok(None);
    }
    let Some(mut options) = parse_cuda_options(args)? else {
        return Ok(None);
    };
    let policy = validate_cuda_options(&options, sub)?;
    if let Some(ref pf) = options.prompt_file {
        options.prompt = fs::read_to_string(pf)
            .map_err(|e| format!("failed to read --prompt-file '{pf}': {e}"))?;
    }
    options.serve.resolve_key(lookup)?;
    // Comparison is local. Live profiling resolves in the nested bench command,
    // not here (and --warp34 takes precedence over --live).
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
    for spec in &mut options.lora_modules {
        let dir = resolve(
            &spec.path,
            &HubOptions {
                revision: spec.revision.clone(),
                offline: options.offline,
            },
            ModelRole::Lora,
        )
        .map_err(|e| e.to_string())?;
        spec.resolved_dir = Some(dir);
    }
    Ok(Some(PreparedCudaCli {
        options,
        policy,
        spec_config,
    }))
}

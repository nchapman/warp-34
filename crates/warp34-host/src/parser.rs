//! `SamplerV1` (penalties, temperature, `top_k`, `top_p`, `min_p`, seeded RNG,
//! and structured-output `TokenBitmaskConstraint` with speculative rollback) and
//! `Qwen3Parser` (`vllm/parser/qwen3.py` parity for `<think>...</think>` reasoning and
//! Qwen3-XML `<tool_call>` extraction).

use std::collections::BTreeMap;

mod native_schema;
type ParameterGrammars = BTreeMap<String, native_schema::ValueGrammar>;
type ToolGrammars = BTreeMap<String, ParameterGrammars>;

use serde::{Deserialize, Serialize};

/// Per-request sampler-level thinking token budget (`vLLM` `ThinkingBudgetStateHolder` /
/// `llama.cpp` `common/reasoning-budget.cpp` parity).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingBudgetConfig {
    pub budget: usize,
    pub start_token_ids: Vec<i32>,
    pub end_token_ids: Vec<i32>,
    pub forced_token_ids: Vec<i32>,
    /// Optional additional stop sequences (e.g., `<tool_call>` when tools are active)
    /// that implicitly terminate reasoning even without an explicit `</think>`.
    #[serde(default)]
    pub extra_end_token_ids: Vec<Vec<i32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThinkingBudgetPhase {
    Idle,
    Counting { remaining: usize },
    Forcing { pos: usize },
    Done { exhausted: bool },
}

#[derive(Clone, Debug)]
struct ThinkingBudgetTracker {
    phase: ThinkingBudgetPhase,
    history: Vec<ThinkingBudgetPhase>,
}

fn find_last_subseq(haystack: &[i32], needle: &[i32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .rposition(|window| window == needle)
}

impl ThinkingBudgetTracker {
    fn new(cfg: &ThinkingBudgetConfig, prompt_tokens: &[i32]) -> Self {
        // Scope prompt detection to the trailing assistant generation header so literal
        // `<think>` tokens inside user content cannot falsely arm the tracker on templates
        // that do not open `<think>` in the assistant header.
        let tail_window = 16.max(cfg.start_token_ids.len() + cfg.end_token_ids.len() + 4);
        let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(tail_window)..];
        let last_start = find_last_subseq(tail, &cfg.start_token_ids);
        let last_end = find_last_subseq(tail, &cfg.end_token_ids);
        let in_think = match (last_start, last_end) {
            (Some(s), Some(e)) => s > e,
            (Some(_), None) => true,
            _ => false,
        };
        let phase = if cfg.forced_token_ids.is_empty() {
            ThinkingBudgetPhase::Done { exhausted: false }
        } else if in_think {
            if cfg.budget == 0 {
                ThinkingBudgetPhase::Forcing { pos: 0 }
            } else {
                ThinkingBudgetPhase::Counting {
                    remaining: cfg.budget,
                }
            }
        } else {
            ThinkingBudgetPhase::Idle
        };
        Self {
            phase,
            history: Vec::new(),
        }
    }

    fn forced_token(&self, cfg: &ThinkingBudgetConfig) -> Option<i32> {
        match self.phase {
            ThinkingBudgetPhase::Forcing { pos } => cfg.forced_token_ids.get(pos).copied(),
            _ => None,
        }
    }

    fn advance(
        &mut self,
        cfg: &ThinkingBudgetConfig,
        output_history: &[i32],
        exited_reasoning_gate: bool,
    ) {
        self.history.push(self.phase);
        if cfg.forced_token_ids.is_empty() {
            self.phase = ThinkingBudgetPhase::Done { exhausted: false };
            return;
        }
        let ends_with_start =
            !cfg.start_token_ids.is_empty() && output_history.ends_with(&cfg.start_token_ids);
        let ends_with_end = (!cfg.end_token_ids.is_empty()
            && output_history.ends_with(&cfg.end_token_ids))
            || cfg
                .extra_end_token_ids
                .iter()
                .any(|seq| !seq.is_empty() && output_history.ends_with(seq));
        self.phase = match self.phase {
            ThinkingBudgetPhase::Done { exhausted } => ThinkingBudgetPhase::Done { exhausted },
            ThinkingBudgetPhase::Idle => {
                if exited_reasoning_gate {
                    ThinkingBudgetPhase::Done { exhausted: false }
                } else if ends_with_start {
                    if cfg.budget == 0 {
                        ThinkingBudgetPhase::Forcing { pos: 0 }
                    } else {
                        ThinkingBudgetPhase::Counting {
                            remaining: cfg.budget,
                        }
                    }
                } else {
                    ThinkingBudgetPhase::Idle
                }
            }
            ThinkingBudgetPhase::Counting { remaining } => {
                if exited_reasoning_gate || ends_with_end {
                    ThinkingBudgetPhase::Done { exhausted: false }
                } else if remaining <= 1 {
                    ThinkingBudgetPhase::Forcing { pos: 0 }
                } else {
                    ThinkingBudgetPhase::Counting {
                        remaining: remaining - 1,
                    }
                }
            }
            ThinkingBudgetPhase::Forcing { pos } => {
                if exited_reasoning_gate || pos + 1 >= cfg.forced_token_ids.len() {
                    ThinkingBudgetPhase::Done { exhausted: true }
                } else {
                    ThinkingBudgetPhase::Forcing { pos: pos + 1 }
                }
            }
        };
    }

    fn rollback(&mut self, num_rejected: usize) {
        for _ in 0..num_rejected {
            if let Some(prev) = self.history.pop() {
                self.phase = prev;
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<i32>,
    pub stop_strings: Vec<String>,
    pub enable_thinking: bool,
    #[serde(default)]
    pub logit_bias: BTreeMap<i32, f32>,
    #[serde(default)]
    pub min_tokens: usize,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default)]
    pub top_logprobs: usize,
    #[serde(default)]
    pub eos_token_ids: Vec<i32>,
    #[serde(default)]
    pub thinking_budget: Option<ThinkingBudgetConfig>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 42,
            max_tokens: 256,
            stop_token_ids: vec![151643, 151645, 248044, 248046],
            stop_strings: Vec::new(),
            enable_thinking: true,
            logit_bias: BTreeMap::new(),
            min_tokens: 0,
            ignore_eos: false,
            logprobs: false,
            top_logprobs: 0,
            eos_token_ids: vec![151643, 151645, 248044, 248046],
            thinking_budget: None,
        }
    }
}

impl SamplingParams {
    #[inline]
    pub fn is_plain_greedy(&self, has_constraint: bool) -> bool {
        !has_constraint
            && self.thinking_budget.is_none()
            && (self.temperature <= 1e-6 || self.top_k == 1)
            && (self.repetition_penalty - 1.0).abs() <= 1e-6
            && self.frequency_penalty.abs() <= 1e-6
            && self.presence_penalty.abs() <= 1e-6
            && self.logit_bias.is_empty()
            && self.min_tokens == 0
            && !self.ignore_eos
            && !self.logprobs
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite()
            || !(0.0..=2.0).contains(&self.temperature)
            || self.top_k < -1
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
            || !self.min_p.is_finite()
            || !(0.0..=1.0).contains(&self.min_p)
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
            || !self.frequency_penalty.is_finite()
            || !(-2.0..=2.0).contains(&self.frequency_penalty)
            || !self.presence_penalty.is_finite()
            || !(-2.0..=2.0).contains(&self.presence_penalty)
            || self.top_logprobs > warp34_kernels::ffi::WARP34_MAX_TOP_LOGPROBS
        {
            return Err("invalid sampling parameters".into());
        }
        for (&tid, &bias) in &self.logit_bias {
            if tid < 0 || !bias.is_finite() || !(-100.0..=100.0).contains(&bias) {
                return Err(format!("invalid logit_bias for token {tid}"));
            }
        }
        if self.stop_strings.iter().any(String::is_empty) {
            return Err("stop strings must not be empty".into());
        }
        Ok(())
    }

    #[inline]
    pub fn is_eos_token(&self, tid: i32, model_eos: &[i32]) -> bool {
        model_eos.contains(&tid) || self.eos_token_ids.contains(&tid)
    }

    #[inline]
    pub fn is_stop_or_eos(&self, tid: i32, model_eos: &[i32]) -> bool {
        if self.is_eos_token(tid, model_eos) {
            return !self.ignore_eos;
        }
        self.stop_token_ids.contains(&tid)
    }

    /// Loads default serving sampling parameters from `<model_dir>/generation_config.json`
    /// (`vLLM` `ModelConfig._load_generation_config_dict` parity), falling back to
    /// [`SamplingParams::default`] when the file is absent or omits sampling fields.
    pub fn from_generation_config_dir(model_dir: &std::path::Path) -> Self {
        std::fs::read_to_string(model_dir.join("generation_config.json"))
            .ok()
            .map_or_else(Self::default, |text| {
                Self::from_generation_config_json(&text)
            })
    }

    /// Parses default serving sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`,
    /// `repetition_penalty`, `frequency_penalty`, `presence_penalty`, and `do_sample: false`)
    /// from a `generation_config.json` payload.
    pub fn from_generation_config_json(json_text: &str) -> Self {
        let mut params = Self::default();
        let Ok(serde_json::Value::Object(obj)) = serde_json::from_str(json_text) else {
            return params;
        };
        if let Some(t) = obj
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .filter(|t| t.is_finite() && (0.0..=2.0).contains(t))
        {
            params.temperature = t;
        }
        if let Some(p) = obj
            .get("top_p")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .filter(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
        {
            params.top_p = p;
        }
        if let Some(k) = obj
            .get("top_k")
            .and_then(serde_json::Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
            .filter(|k| *k >= -1)
        {
            params.top_k = if k == 0 { -1 } else { k };
        }
        if let Some(mp) = obj
            .get("min_p")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .filter(|mp| mp.is_finite() && (0.0..=1.0).contains(mp))
        {
            params.min_p = mp;
        }
        if let Some(rp) = obj
            .get("repetition_penalty")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .filter(|rp| rp.is_finite() && *rp > 0.0)
        {
            params.repetition_penalty = rp;
        }
        if let Some(fp) = obj
            .get("frequency_penalty")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .filter(|fp| fp.is_finite() && (-2.0..=2.0).contains(fp))
        {
            params.frequency_penalty = fp;
        }
        if let Some(pp) = obj
            .get("presence_penalty")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .filter(|pp| pp.is_finite() && (-2.0..=2.0).contains(pp))
        {
            params.presence_penalty = pp;
        }
        if obj.get("do_sample").and_then(serde_json::Value::as_bool) == Some(false) {
            params.temperature = 0.0;
        }
        params
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: i32,
    pub logprob: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepLogprobs {
    pub token_id: i32,
    pub logprob: f32,
    pub top_logprobs: Vec<TokenLogprob>,
}

pub use crate::constraints::TokenBitmaskConstraint;

/// Transactional `SamplerV1` implementing `vLLM`'s 9-step sampling pipeline with
/// deterministic per-(seed, position) RNG and speculative rollback.
#[derive(Clone, Debug)]
pub struct SamplerV1 {
    pub params: SamplingParams,
    pub prompt_counts: BTreeMap<i32, u32>,
    pub output_counts: BTreeMap<i32, u32>,
    pub output_history: Vec<i32>,
    pub constraint: Option<TokenBitmaskConstraint>,
    thinking_tracker: Option<ThinkingBudgetTracker>,
}

impl SamplerV1 {
    pub fn new(
        params: SamplingParams,
        prompt_tokens: &[i32],
        constraint: Option<TokenBitmaskConstraint>,
    ) -> Self {
        let mut prompt_counts = BTreeMap::new();
        for &t in prompt_tokens {
            *prompt_counts.entry(t).or_insert(0) += 1;
        }
        let thinking_tracker = params
            .thinking_budget
            .as_ref()
            .map(|cfg| ThinkingBudgetTracker::new(cfg, prompt_tokens));
        Self {
            params,
            prompt_counts,
            output_counts: BTreeMap::new(),
            output_history: Vec::new(),
            constraint,
            thinking_tracker,
        }
    }

    #[inline]
    pub fn thinking_budget_exhausted(&self) -> bool {
        self.thinking_tracker.as_ref().is_some_and(|t| {
            matches!(
                t.phase,
                ThinkingBudgetPhase::Forcing { .. } | ThinkingBudgetPhase::Done { exhausted: true }
            )
        })
    }

    #[inline]
    pub fn forced_thinking_token(&self) -> Option<i32> {
        let cfg = self.params.thinking_budget.as_ref()?;
        let tracker = self.thinking_tracker.as_ref()?;
        let forced = tracker.forced_token(cfg)?;
        if let Some(ref c) = self.constraint
            && !c.is_token_allowed(forced)
        {
            return None;
        }
        Some(forced)
    }

    /// Deterministic SplitMix64 uniform float in `[0.0, 1.0)` keyed by `(seed, step_pos)`.
    /// Stateless: callers may domain-separate RNG lanes using distinct derived seeds.
    /// `step_pos` is the absolute output position, independent of proposer consumption.
    pub fn uniform_f32(seed: u64, step_pos: usize) -> f32 {
        let mut z = seed
            .wrapping_add((step_pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as u32 as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Returns normalized dense vocabulary probabilities without advancing RNG, history,
    /// or grammar state. Greedy policy yields a point mass (ties use lowest token ID).
    /// Masked logits may be nonfinite; allowed logits and processed scores must be finite.
    pub fn processed_distribution(&self, logits: &[f32]) -> Result<Vec<f32>, String> {
        let probs = self.processed_candidates(logits)?;
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        let mut dense = vec![0.0; logits.len()];
        for (tid, p) in probs {
            dense[tid as usize] = p / sum;
        }
        Ok(dense)
    }

    // Keep sorted weights (including the pre-CDF nucleus sum) for legacy RNG ordering.
    fn processed_candidates(&self, logits: &[f32]) -> Result<Vec<(i32, f32)>, String> {
        self.params.validate()?;
        if logits.is_empty() || logits.len() > i32::MAX as usize {
            return Err("empty or oversized vocabulary".into());
        }
        if let Some(forced_tid) = self.forced_thinking_token()
            && forced_tid >= 0
            && (forced_tid as usize) < logits.len()
        {
            return Ok(vec![(forced_tid, 1.0)]);
        }
        let honor_bans = (self.params.ignore_eos
            || self.output_history.len() < self.params.min_tokens)
            && (0..logits.len() as i32).any(|tid| {
                self.constraint
                    .as_ref()
                    .is_none_or(|c| c.is_token_allowed(tid))
                    && !self.is_token_soft_banned(tid, &[])
            });
        for (tid, &logit) in logits.iter().enumerate() {
            let tid_i32 = tid as i32;
            let allowed = self
                .constraint
                .as_ref()
                .is_none_or(|c| c.is_token_allowed(tid_i32))
                && !(honor_bans && self.is_token_soft_banned(tid_i32, &[]));
            if allowed && !logit.is_finite() {
                return Err(format!("nonfinite allowed logit at token {tid}"));
            }
        }
        let mut candidates: Vec<(i32, f32)> = logits
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| {
                let tid = idx as i32;
                if let Some(ref c) = self.constraint
                    && !c.is_token_allowed(tid)
                {
                    return None;
                }
                if honor_bans && self.is_token_soft_banned(tid, &[]) {
                    return None;
                }
                let mut l = val + self.params.logit_bias.get(&tid).copied().unwrap_or(0.0);
                let in_prompt = self.prompt_counts.get(&tid).copied().unwrap_or(0) > 0;
                let out_cnt = self.output_counts.get(&tid).copied().unwrap_or(0);
                if (in_prompt || out_cnt > 0) && (self.params.repetition_penalty - 1.0).abs() > 1e-6
                {
                    if l > 0.0 {
                        l /= self.params.repetition_penalty;
                    } else {
                        l *= self.params.repetition_penalty;
                    }
                }
                if out_cnt > 0 {
                    l -= self.params.frequency_penalty * (out_cnt as f32);
                    l -= self.params.presence_penalty;
                }
                Some((tid, l))
            })
            .collect();

        if candidates.is_empty() {
            return Err("grammar has no support in the vocabulary".into());
        }
        if candidates.iter().any(|c| !c.1.is_finite()) {
            return Err("nonfinite penalized logit".into());
        }

        // Greedy fast-path when temperature <= 0 or top_k == 1
        if self.params.temperature <= 1e-6 || self.params.top_k == 1 {
            candidates.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            return Ok(vec![(candidates[0].0, 1.0)]);
        }

        let inv_temp = 1.0 / self.params.temperature;
        for c in &mut candidates {
            c.1 *= inv_temp;
            if !c.1.is_finite() {
                return Err("nonfinite temperature-scaled logit".into());
            }
        }
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        if self.params.top_k > 0 && (self.params.top_k as usize) < candidates.len() {
            candidates.truncate(self.params.top_k as usize);
        }

        let max_logit = candidates[0].1;
        let mut probs: Vec<(i32, f32)> = candidates
            .iter()
            .map(|&(tid, l)| (tid, (l - max_logit).exp()))
            .collect();
        let sum_p: f32 = probs.iter().map(|x| x.1).sum();
        if sum_p > 0.0 {
            for p in &mut probs {
                p.1 /= sum_p;
            }
        }

        // Apply `min_p` relative to top-1 probability
        if self.params.min_p > 0.0 && !probs.is_empty() {
            let threshold = probs[0].1 * self.params.min_p;
            let keep = probs.iter().take_while(|p| p.1 >= threshold).count().max(1);
            probs.truncate(keep);
            let renorm: f32 = probs.iter().map(|x| x.1).sum();
            if renorm > 0.0 {
                for p in &mut probs {
                    p.1 /= renorm;
                }
            }
        }

        // Apply `top_p` nucleus cutoff
        if self.params.top_p < 1.0 && probs.len() > 1 {
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for p in &probs {
                cum += p.1;
                keep += 1;
                if cum >= self.params.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }

        Ok(probs)
    }

    /// Samples and commits a token using the same policy as `processed_distribution`.
    /// Empty input retains the legacy non-committing return value of zero.
    /// Panics on other invalid inputs before changing transactional state; use
    /// `processed_distribution` for fallible validation.
    pub fn sample_from_logits(&mut self, logits: &[f32]) -> i32 {
        if logits.is_empty() {
            return 0;
        }
        self.try_sample_from_logits(logits)
            .unwrap_or_else(|err| panic!("invalid sampling input: {err}"))
    }

    pub fn try_sample_from_logits(&mut self, logits: &[f32]) -> Result<i32, String> {
        self.materialize_pending_constraint()?;
        let probs = self.processed_candidates(logits)?;
        let norm_sum: f32 = probs.iter().map(|x| x.1).sum();
        let u = Self::uniform_f32(self.params.seed, self.output_history.len()) * norm_sum;
        let mut acc = 0.0f32;
        let mut chosen = probs.last().unwrap().0;
        for &(tid, p) in &probs {
            acc += p;
            if u <= acc {
                chosen = tid;
                break;
            }
        }
        self.try_commit_token(chosen)?;
        Ok(chosen)
    }

    #[inline]
    pub fn is_token_soft_banned(&self, tid: i32, model_eos: &[i32]) -> bool {
        if self.params.ignore_eos && self.params.is_eos_token(tid, model_eos) {
            return true;
        }
        if self.output_history.len() < self.params.min_tokens
            && (self.params.is_eos_token(tid, model_eos)
                || self.params.stop_token_ids.contains(&tid))
        {
            return true;
        }
        false
    }

    /// Returns true when this row satisfies `warp34_sampler_step_fast_greedy_f16` (`is_row_fast_greedy`)
    /// AND its `Warp34SamplerRowCfg` + `Warp34SamplerSparseEntry` slice are invariant across subsequent
    /// decode steps (no stateful grammar constraint, no history-dependent penalties or `min_tokens`
    /// transition, and no multi-token `stop_strings`).
    pub fn is_fast_greedy_stateless(&self, model_eos_len: usize) -> bool {
        let is_greedy = self.params.temperature <= 1e-6 || self.params.top_k == 1;
        let lp_at_most_zero = !self.params.logprobs || self.params.top_logprobs == 0;
        let no_penalties = (self.params.repetition_penalty - 1.0).abs() <= 1e-6
            && self.params.frequency_penalty.abs() <= 1e-6
            && self.params.presence_penalty.abs() <= 1e-6;
        let past_min_tokens = self.output_history.len() >= self.params.min_tokens;
        let max_sparse = self.params.logit_bias.len()
            + if self.params.ignore_eos {
                model_eos_len + self.params.eos_token_ids.len()
            } else {
                0
            };
        self.constraint.is_none()
            && self.thinking_tracker.is_none()
            && is_greedy
            && lp_at_most_zero
            && no_penalties
            && past_min_tokens
            && self.params.stop_strings.is_empty()
            && max_sparse <= 32
    }

    /// Appends one row's GPU sampler config, optional packed grammar bitmask words, and
    /// deduplicated sparse penalty/bias/ban entries for `warp34_sampler_step_f16`.
    #[allow(clippy::too_many_arguments)]
    pub fn append_gpu_row(
        &self,
        vocab_size: usize,
        model_eos: &[i32],
        uniform_sample: f32,
        recovery_uniform: f32,
        draft_token_id: i32,
        use_token_id_cdf: bool,
        bitmask_words_out: &mut Vec<u32>,
        sparse_out: &mut Vec<warp34_kernels::ffi::Warp34SamplerSparseEntry>,
    ) -> Result<warp34_kernels::ffi::Warp34SamplerRowCfg, String> {
        self.params.validate()?;
        let mask_words_per_row = vocab_size.div_ceil(32);
        let (has_bitmask, row_bitmask_word_offset) = if let Some(forced_tid) =
            self.forced_thinking_token()
            && forced_tid >= 0
            && (forced_tid as usize) < vocab_size
        {
            let off = i32::try_from(bitmask_words_out.len())
                .map_err(|_| "bitmask buffer overflow".to_string())?;
            let start = bitmask_words_out.len();
            bitmask_words_out.resize(start + mask_words_per_row, 0u32);
            let tid_u = forced_tid as usize;
            bitmask_words_out[start + (tid_u / 32)] = 1u32 << (tid_u % 32);
            (1, off)
        } else if let Some(ref c) = self.constraint {
            if c.has_pending() {
                return Err("unmaterialized token constraint before append_gpu_row".to_string());
            }
            let off = i32::try_from(bitmask_words_out.len())
                .map_err(|_| "bitmask buffer overflow".to_string())?;
            let start = bitmask_words_out.len();
            bitmask_words_out.resize(start + mask_words_per_row, 0u32);
            let words = c.mask_words();
            let copy_len = words.len().min(mask_words_per_row);
            bitmask_words_out[start..start + copy_len].copy_from_slice(&words[..copy_len]);
            (1, off)
        } else {
            (0, 0)
        };

        let sparse_start = sparse_out.len();
        let has_rep = (self.params.repetition_penalty - 1.0).abs() > 1e-6;
        let has_freq_pres =
            self.params.frequency_penalty.abs() > 1e-6 || self.params.presence_penalty.abs() > 1e-6;
        let under_min = self.output_history.len() < self.params.min_tokens;
        let has_bans = self.params.ignore_eos || under_min;

        if has_rep || has_freq_pres || !self.params.logit_bias.is_empty() || has_bans {
            let vocab_i32 = i32::try_from(vocab_size).unwrap_or(i32::MAX);
            let empty_map = std::collections::BTreeMap::new();
            let prompt_src = if has_rep {
                &self.prompt_counts
            } else {
                &empty_map
            };
            let output_src = if has_rep || has_freq_pres {
                &self.output_counts
            } else {
                &empty_map
            };
            let mut p_iter = prompt_src
                .iter()
                .filter(|&(&tid, &cnt)| cnt > 0 && tid >= 0 && tid < vocab_i32)
                .peekable();
            let mut o_iter = output_src
                .iter()
                .filter(|&(&tid, &cnt)| cnt > 0 && tid >= 0 && tid < vocab_i32)
                .peekable();
            sparse_out.reserve(prompt_src.len() + output_src.len());
            loop {
                match (p_iter.peek(), o_iter.peek()) {
                    (Some(&(&p_tid, _)), Some(&(&o_tid, &o_cnt))) if p_tid == o_tid => {
                        p_iter.next();
                        o_iter.next();
                        sparse_out.push(warp34_kernels::ffi::Warp34SamplerSparseEntry {
                            token_id: p_tid,
                            output_count: o_cnt,
                            in_prompt: 1,
                            ..Default::default()
                        });
                    }
                    (Some(&(&p_tid, _)), Some(&(&o_tid, _))) if p_tid < o_tid => {
                        p_iter.next();
                        sparse_out.push(warp34_kernels::ffi::Warp34SamplerSparseEntry {
                            token_id: p_tid,
                            in_prompt: 1,
                            ..Default::default()
                        });
                    }
                    (Some(&(&p_tid, _)), None) => {
                        p_iter.next();
                        sparse_out.push(warp34_kernels::ffi::Warp34SamplerSparseEntry {
                            token_id: p_tid,
                            in_prompt: 1,
                            ..Default::default()
                        });
                    }
                    (_, Some(&(&o_tid, &o_cnt))) => {
                        o_iter.next();
                        sparse_out.push(warp34_kernels::ffi::Warp34SamplerSparseEntry {
                            token_id: o_tid,
                            output_count: o_cnt,
                            ..Default::default()
                        });
                    }
                    (None, None) => break,
                }
            }
            for (&tid, &bias) in &self.params.logit_bias {
                if tid >= 0 && tid < vocab_i32 {
                    match sparse_out[sparse_start..].binary_search_by_key(&tid, |e| e.token_id) {
                        Ok(idx) => sparse_out[sparse_start + idx].logit_bias = bias,
                        Err(idx) => sparse_out.insert(
                            sparse_start + idx,
                            warp34_kernels::ffi::Warp34SamplerSparseEntry {
                                token_id: tid,
                                logit_bias: bias,
                                ..Default::default()
                            },
                        ),
                    }
                }
            }
            if has_bans {
                let iter_eos = model_eos
                    .iter()
                    .chain(self.params.eos_token_ids.iter())
                    .copied()
                    .filter(|&tid| {
                        tid >= 0 && tid < vocab_i32 && self.is_token_soft_banned(tid, model_eos)
                    });
                let iter_stop = self
                    .params
                    .stop_token_ids
                    .iter()
                    .copied()
                    .filter(|&tid| under_min && tid >= 0 && tid < vocab_i32);
                for tid in iter_eos.chain(iter_stop) {
                    match sparse_out[sparse_start..].binary_search_by_key(&tid, |e| e.token_id) {
                        Ok(idx) => sparse_out[sparse_start + idx].banned = 1,
                        Err(idx) => sparse_out.insert(
                            sparse_start + idx,
                            warp34_kernels::ffi::Warp34SamplerSparseEntry {
                                token_id: tid,
                                banned: 1,
                                ..Default::default()
                            },
                        ),
                    }
                }
            }
        }
        let sparse_count = sparse_out.len() - sparse_start;
        let top_logprobs_k = if self.params.logprobs {
            self.params.top_logprobs as i32
        } else {
            -1
        };

        Ok(warp34_kernels::ffi::Warp34SamplerRowCfg {
            temperature: self.params.temperature,
            top_k: self.params.top_k,
            top_p: self.params.top_p,
            min_p: self.params.min_p,
            repetition_penalty: self.params.repetition_penalty,
            frequency_penalty: self.params.frequency_penalty,
            presence_penalty: self.params.presence_penalty,
            uniform_sample,
            recovery_uniform,
            draft_token_id,
            has_bitmask,
            row_bitmask_word_offset,
            sparse_offset: i32::try_from(sparse_start)
                .map_err(|_| "sparse offset overflow".to_string())?,
            sparse_count: i32::try_from(sparse_count)
                .map_err(|_| "sparse count overflow".to_string())?,
            top_logprobs_k,
            use_token_id_cdf: i32::from(use_token_id_cdf),
        })
    }

    pub fn materialize_pending_constraint(&mut self) -> Result<(), String> {
        if let Some(ref mut c) = self.constraint {
            c.materialize_pending()?;
        }
        Ok(())
    }

    pub fn commit_token(&mut self, token_id: i32) {
        self.try_commit_token(token_id)
            .expect("invalid sampler token commit");
    }

    pub fn try_commit_token(&mut self, token_id: i32) -> Result<(), String> {
        if let Some(ref mut c) = self.constraint {
            c.advance(token_id)?;
        }
        self.output_history.push(token_id);
        *self.output_counts.entry(token_id).or_insert(0) += 1;
        let exited_gate = self
            .constraint
            .as_ref()
            .is_some_and(|c| !c.in_reasoning_gate());
        if let (Some(cfg), Some(tracker)) = (
            self.params.thinking_budget.as_ref(),
            self.thinking_tracker.as_mut(),
        ) {
            tracker.advance(cfg, &self.output_history, exited_gate);
        }
        Ok(())
    }

    /// Commits `token_id` to `output_history` / `output_counts` immediately while deferring
    /// `XGrammar` mask materialization until the next GPU forward pass is already in flight.
    pub fn try_commit_token_deferred(&mut self, token_id: i32) -> Result<(), String> {
        if let Some(ref mut c) = self.constraint {
            c.advance_deferred(token_id)?;
        }
        self.output_history.push(token_id);
        *self.output_counts.entry(token_id).or_insert(0) += 1;
        let exited_gate = self
            .constraint
            .as_ref()
            .is_some_and(|c| !c.in_reasoning_gate());
        if let (Some(cfg), Some(tracker)) = (
            self.params.thinking_budget.as_ref(),
            self.thinking_tracker.as_mut(),
        ) {
            tracker.advance(cfg, &self.output_history, exited_gate);
        }
        Ok(())
    }

    /// Rolls back `num_rejected` speculative tokens from `output_history`, `output_counts`,
    /// `thinking_tracker`, and `constraint`.
    pub fn rollback_tokens(&mut self, num_rejected: usize) {
        self.try_rollback_tokens(num_rejected)
            .expect("invalid sampler rollback");
    }

    pub fn try_rollback_tokens(&mut self, num_rejected: usize) -> Result<(), String> {
        if num_rejected > self.output_history.len() {
            return Err("sampler rollback exceeds committed tokens".into());
        }
        if let Some(ref mut c) = self.constraint {
            c.rollback(num_rejected)?;
        }
        if let Some(ref mut tracker) = self.thinking_tracker {
            tracker.rollback(num_rejected);
        }
        for _ in 0..num_rejected {
            if let Some(tid) = self.output_history.pop()
                && let Some(cnt) = self.output_counts.get_mut(&tid)
            {
                if *cnt <= 1 {
                    self.output_counts.remove(&tid);
                } else {
                    *cnt -= 1;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedFunctionCall {
    pub name: String,
    /// JSON-serialized arguments object string (OpenAI `function.arguments` contract).
    pub arguments: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ParsedFunctionCall,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedAssistantTurn {
    pub reasoning_content: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments_delta: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamParseDelta {
    pub reasoning_delta: Option<String>,
    pub content_delta: Option<String>,
    pub tool_call_deltas: Vec<StreamToolCallDelta>,
    pub completed_tool_calls: Vec<ParsedToolCall>,
}

#[derive(Clone, Debug)]
enum ActiveParamStream {
    StreamString {
        value_start: usize,
        value_scan_pos: usize,
        skipped_leading_nl: bool,
    },
    Buffered {
        name: String,
        value_start: usize,
        is_first: bool,
    },
}

#[derive(Clone, Debug, Default)]
struct ToolStreamState {
    aborted: bool,
    func_name: Option<String>,
    header_emitted: bool,
    cursor: usize,
    seen_params: Vec<String>,
    streamed_args: String,
    active_param: Option<ActiveParamStream>,
}

enum NativeToolStep {
    InProgress,
    Completed(usize),
    Aborted,
}

/// Unified Qwen 3.5 / 3.6 / 3.8 reasoning and tool parser (`vllm/parser/qwen3.py` parity).
///
/// Supports:
/// - `enable_thinking: bool` (`<think>...</think>` extraction into `reasoning_content`)
/// - Implicit `</think>` transition when `<tool_call>` appears directly inside reasoning
/// - Duplicate `</think>` absorption in content state
/// - **Qwen3-XML** (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`)
#[derive(Clone, Debug)]
pub struct Qwen3Parser {
    pub enable_thinking: bool,
    in_reasoning: bool,
    in_tool_call: bool,
    finishing: bool,
    buffer: String,
    reasoning_accum: String,
    content_accum: String,
    tool_calls: Vec<ParsedToolCall>,
    next_tool_idx: usize,
    tools: Option<Vec<serde_json::Value>>,
    value_grammars: ToolGrammars,
    tool_stream: ToolStreamState,
}

impl Qwen3Parser {
    pub const THINK_START: &'static str = "<think>";
    pub const THINK_END: &'static str = "</think>";
    pub const TOOL_START: &'static str = "<tool_call>";
    pub const TOOL_END: &'static str = "</tool_call>";

    pub fn new(enable_thinking: bool) -> Self {
        Self {
            enable_thinking,
            in_reasoning: enable_thinking,
            in_tool_call: false,
            finishing: false,
            buffer: String::new(),
            reasoning_accum: String::new(),
            content_accum: String::new(),
            tool_calls: Vec::new(),
            next_tool_idx: 0,
            tools: None,
            value_grammars: ToolGrammars::new(),
            tool_stream: ToolStreamState::default(),
        }
    }

    /// Parse native Qwen XML using canonical OpenAI function tool schemas.
    /// Unlike `new`, only the native canonical envelope is executable and string
    /// parameters are raw text, not JSON string literals. The active generation
    /// grammar owns schema validation; this parser only resolves argument types.
    /// An empty tool list disables call extraction (including `tool_choice: none`).
    pub fn with_tools(enable_thinking: bool, tools: &[serde_json::Value]) -> Result<Self, String> {
        let mut parser = Self::new(enable_thinking);
        let mut metadata = tools.to_vec();
        for tool in &mut metadata {
            let name = tool
                .pointer("/function/name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned();
            if let Some(parameters) = tool.pointer_mut("/function/parameters") {
                let root = crate::constraints::sanitize_tool_schema(parameters)?;
                let mut budget = 4096;
                *parameters = Self::resolve_local_refs(&root, &root, true, 0, &mut budget)?;
                let mut grammars = ParameterGrammars::new();
                if let Some(properties) = parameters
                    .get("properties")
                    .and_then(serde_json::Value::as_object)
                {
                    for (key, schema) in properties {
                        grammars.insert(key.clone(), native_schema::compile(&root, schema)?);
                    }
                }
                parser.value_grammars.insert(name, grammars);
            }
        }
        parser.tools = Some(metadata);
        Ok(parser)
    }

    /// Returns whether the parser is currently inside a `<think>...</think>` reasoning block.
    #[inline]
    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    /// Returns the number of tool calls completed or currently streaming a header.
    #[inline]
    pub fn emitted_tool_calls_count(&self) -> usize {
        self.tool_calls.len() + usize::from(self.tool_stream.header_emitted)
    }

    // Resolve only metadata used for top-level XML argument conversion. Nested
    // object properties and array items are JSON data and need no expansion.
    fn resolve_local_refs(
        schema: &serde_json::Value,
        root: &serde_json::Value,
        parameters: bool,
        depth: usize,
        budget: &mut usize,
    ) -> Result<serde_json::Value, String> {
        if depth >= 32 {
            return Err("tool parser metadata resolution depth limit (32) exceeded".into());
        }
        if *budget == 0 {
            return Err("tool parser metadata resolution budget (4096) exceeded".into());
        }
        *budget -= 1;
        let Some(map) = schema.as_object() else {
            return Ok(schema.clone());
        };
        let mut result = serde_json::Map::new();
        if let Some(reference) = map.get("$ref") {
            let pointer = reference
                .as_str()
                .and_then(|r| r.strip_prefix('#'))
                .ok_or("tool parser metadata requires a local $ref")?;
            let target = root
                .pointer(pointer)
                .ok_or("tool parser metadata $ref target not found")?;
            let resolved = Self::resolve_local_refs(target, root, parameters, depth + 1, budget)?;
            result = resolved
                .as_object()
                .cloned()
                .ok_or("tool parser metadata $ref target must be an object")?;
        }
        for (key, value) in map {
            match key.as_str() {
                "type" | "const" | "enum" => {
                    result.insert(key.clone(), value.clone());
                }
                "properties" if parameters => {
                    if let Some(properties) = value.as_object() {
                        let mut resolved = serde_json::Map::new();
                        for (name, schema) in properties {
                            resolved.insert(
                                name.clone(),
                                Self::resolve_local_refs(schema, root, false, depth + 1, budget)?,
                            );
                        }
                        result.insert(key.clone(), resolved.into());
                    }
                }
                "anyOf" | "oneOf" | "allOf" => {
                    if let Some(choices) = value.as_array() {
                        let resolved = choices
                            .iter()
                            .map(|choice| {
                                Self::resolve_local_refs(
                                    choice,
                                    root,
                                    parameters,
                                    depth + 1,
                                    budget,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        result.insert(key.clone(), resolved.into());
                    }
                }
                "$ref" => {}
                _ => {
                    result.insert(key.clone(), value.clone());
                }
            }
        }
        Ok(result.into())
    }

    fn properties<'a>(body: &str, tools: &'a [serde_json::Value]) -> Option<&'a serde_json::Value> {
        let header = body
            .trim()
            .strip_prefix('<')?
            .trim_start()
            .strip_prefix("function")?
            .trim_start()
            .strip_prefix('=')?;
        let name = header.split_once('>')?.0.trim();
        tools.iter().find_map(|tool| {
            let function = tool.get("function")?;
            (function.get("name")?.as_str()? == name)
                .then(|| function.pointer("/parameters/properties"))
                .flatten()
        })
    }

    fn schema_types(schema: &serde_json::Value, types: &mut Vec<String>) {
        if let Some(t) = schema.get("type") {
            if let Some(t) = t.as_str() {
                types.push(t.to_owned());
            }
            if let Some(ts) = t.as_array() {
                types.extend(ts.iter().filter_map(|t| t.as_str().map(str::to_owned)));
            }
        }
        {
            let values = schema.get("enum").and_then(|v| v.as_array());
            for v in values.into_iter().flatten().chain(schema.get("const")) {
                types.push(
                    match v {
                        serde_json::Value::Null => "null",
                        serde_json::Value::Bool(_) => "boolean",
                        serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                        serde_json::Value::Number(_) => "number",
                        serde_json::Value::String(_) => "string",
                        serde_json::Value::Array(_) => "array",
                        serde_json::Value::Object(_) => "object",
                    }
                    .to_owned(),
                );
            }
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(choices) = schema.get(key).and_then(|v| v.as_array()) {
                for choice in choices {
                    Self::schema_types(choice, types);
                }
            }
        }
    }

    fn convert_arg(
        raw: &str,
        schema: Option<&serde_json::Value>,
        grammar: Option<&native_schema::ValueGrammar>,
    ) -> serde_json::Value {
        let mut types = Vec::new();
        if let Some(schema) = schema {
            Self::schema_types(schema, &mut types);
        }
        if schema.is_some() && types.is_empty() {
            return serde_json::from_str(raw).unwrap_or_else(|_| {
                serde_json::Value::String(Self::trim_wrapping_newlines(raw).to_owned())
            });
        }
        for candidate in ["null", "integer", "number", "boolean", "object", "array"] {
            if !types.iter().any(|t| t == candidate) {
                continue;
            }
            let value = match candidate {
                "null" if raw.trim() == "null" => Some(serde_json::Value::Null),
                "integer" => raw
                    .trim()
                    .parse::<i64>()
                    .ok()
                    .map(Into::into)
                    .or_else(|| raw.trim().parse::<u64>().ok().map(Into::into)),
                "number" => serde_json::from_str::<serde_json::Number>(raw.trim())
                    .ok()
                    .map(Into::into),
                "boolean" => match raw.trim() {
                    "true" => Some(true.into()),
                    "false" => Some(false.into()),
                    _ => None,
                },
                "object" => serde_json::from_str::<serde_json::Value>(raw)
                    .ok()
                    .filter(|v| v.is_object()),
                "array" => serde_json::from_str::<serde_json::Value>(raw)
                    .ok()
                    .filter(|v| v.is_array()),
                _ => None,
            };
            if let Some(value) = value
                && native_schema::accepts(&value, grammar)
            {
                return value;
            }
        }
        // Strip Qwen3-XML's single wrapping newline on each side first (matching
        // `vllm`'s `_trim_wrapping_newlines`), while still allowing `const`/`enum`
        // rules (which use `max_whitespace_cnt: 1`) to fall back to `raw` or
        // single-byte whitespace-trimmed variants.
        let trimmed_nl = Self::trim_wrapping_newlines(raw);
        let leading = raw.strip_prefix([' ', '\n', '\r', '\t']).unwrap_or(raw);
        for candidate in [
            trimmed_nl,
            raw,
            leading,
            raw.strip_suffix([' ', '\n', '\r', '\t']).unwrap_or(raw),
            leading
                .strip_suffix([' ', '\n', '\r', '\t'])
                .unwrap_or(leading),
        ] {
            let value = serde_json::Value::String(candidate.to_owned());
            if native_schema::accepts(&value, grammar) {
                return value;
            }
        }
        serde_json::Value::String(trimmed_nl.to_owned())
    }

    fn trim_wrapping_newlines(s: &str) -> &str {
        let s = s.strip_prefix('\n').unwrap_or(s);
        s.strip_suffix('\n').unwrap_or(s)
    }

    /// Legacy tolerant parsing for `new`; server parsing uses the native scanner.
    fn find_tool_delimiter(text: &str, delimiter: &str) -> Option<usize> {
        fn skip_json(text: &str, start: usize) -> Result<usize, ()> {
            let rest = &text[start..];
            let trimmed = rest.trim_start();
            let offset = start + rest.len() - trimmed.len();
            if !trimmed.starts_with(['"', '{', '[']) {
                return Ok(start);
            }
            let mut values =
                serde_json::Deserializer::from_str(trimmed).into_iter::<serde_json::Value>();
            match values.next() {
                Some(Ok(_)) => Ok(offset + values.byte_offset()),
                Some(Err(error)) if error.is_eof() => Err(()),
                _ => Ok(start), // Native XML also allows unquoted text values.
            }
        }
        fn match_tag(rest: &str, delimiter: &str) -> bool {
            if rest.starts_with(delimiter) {
                return true;
            }
            let Some(after_lt) = rest.strip_prefix('<') else {
                return false;
            };
            let after_lt = after_lt.trim_start();
            if let Some(after_delim_lt) = delimiter.strip_prefix('<') {
                if after_delim_lt.starts_with('/') {
                    let Some(after_slash) = after_lt.strip_prefix('/') else {
                        return false;
                    };
                    let tag = after_delim_lt.trim_start_matches('/').trim_end_matches('>');
                    let after_tag = after_slash.trim_start();
                    if let Some(after_match) = after_tag.strip_prefix(tag) {
                        return after_match.trim_start().starts_with('>');
                    }
                } else if after_delim_lt.ends_with('=') {
                    let tag = after_delim_lt.trim_end_matches('=');
                    if let Some(after_tag) = after_lt.strip_prefix(tag) {
                        return after_tag.trim_start().starts_with('=');
                    }
                }
            }
            false
        }
        let mut cursor = skip_json(text, 0).ok()?;
        while cursor < text.len() {
            let next = cursor + text[cursor..].find('<')?;
            let rest = &text[next..];
            if match_tag(rest, delimiter) {
                return Some(next);
            }
            if match_tag(rest, "<parameter=") {
                let value_start = next + rest.find('>')? + 1;
                cursor = skip_json(text, value_start).ok()?;
            } else {
                cursor = next + 1;
            }
        }
        None
    }

    /// Native values own all bytes until their parameter close. Outer tags in
    /// raw strings are data; JSON alternatives additionally shield quoted closes.
    fn native_parameter_end(
        text: &str,
        schema: Option<&serde_json::Value>,
        grammar: Option<&native_schema::ValueGrammar>,
        finishing: bool,
    ) -> Option<usize> {
        let mut types = Vec::new();
        if let Some(schema) = schema {
            Self::schema_types(schema, &mut types);
        }
        if types.is_empty() || types.iter().any(|t| t == "object" || t == "array") {
            let trimmed = text.trim_start();
            if trimmed.starts_with(['{', '[']) {
                let mut values =
                    serde_json::Deserializer::from_str(trimmed).into_iter::<serde_json::Value>();
                match values.next() {
                    Some(Ok(value)) if native_schema::accepts(&value, grammar) => {
                        let end = text.len() - trimmed.len() + values.byte_offset();
                        return text[end..].find("</parameter>").map(|n| end + n);
                    }
                    // A failed JSON alternative may still be a complete raw
                    // string. Wait until EOS before choosing that interpretation;
                    // otherwise a split inside JSON could expose quoted markup.
                    Some(Err(_)) if !finishing => return None,
                    _ => {}
                }
            }
        }
        // Native const/enum rules may themselves contain the parameter suffix.
        // A boundary is usable only once the value satisfies its conversion metadata.
        text.match_indices("</parameter>").find_map(|(end, _)| {
            native_schema::accepts(&Self::convert_arg(&text[..end], schema, grammar), grammar)
                .then_some(end)
        })
    }

    fn native_tool_end(
        text: &str,
        properties: Option<&serde_json::Value>,
        grammars: Option<&ParameterGrammars>,
        finishing: bool,
    ) -> Option<usize> {
        let mut cursor = 0;
        while cursor < text.len() {
            let next = cursor + text[cursor..].find('<')?;
            let rest = &text[next..];
            if rest.starts_with(Self::TOOL_END) {
                return Some(next);
            }
            if let Some(header) = rest.strip_prefix("<parameter=") {
                let angle = header.find('>')?;
                let name = &header[..angle];
                let start = next + "<parameter=".len() + angle + 1;
                cursor = start
                    + Self::native_parameter_end(
                        &text[start..],
                        properties.and_then(|p| p.get(name)),
                        grammars.and_then(|g| g.get(name)),
                        finishing,
                    )?
                    + "</parameter>".len();
            } else {
                cursor = next + 1;
            }
        }
        None
    }

    fn parse_native_body(
        body: &str,
        call_idx: usize,
        tools: &[serde_json::Value],
        grammars: &ToolGrammars,
        finishing: bool,
    ) -> Option<ParsedToolCall> {
        let header = body.strip_prefix("\n<function=")?;
        let (name, rest) = header.split_once(">\n")?;
        tools
            .iter()
            .find(|tool| tool.pointer("/function/name").and_then(|v| v.as_str()) == Some(name))?;
        let properties = Self::properties(body, tools);
        let grammars = grammars.get(name);
        let mut rest = rest.strip_suffix("\n</function>\n")?;
        let mut params = serde_json::Map::new();
        while !rest.trim().is_empty() {
            // Native parameter-list whitespace is outside the value wrapper.
            rest = rest.trim_start_matches([' ', '\n', '\r', '\t']);
            let header = rest.strip_prefix("<parameter=")?;
            let (name, value) = header.split_once('>')?;
            if name.trim() != name || params.contains_key(name) {
                return None;
            }
            let schema = properties.and_then(|p| p.get(name));
            let grammar = grammars.and_then(|g| g.get(name));
            let end = Self::native_parameter_end(value, schema, grammar, finishing)?;
            let converted = Self::convert_arg(&value[..end], schema, grammar);
            if !native_schema::accepts(&converted, grammar) {
                return None;
            }
            params.insert(name.to_owned(), converted);
            rest = &value[end + "</parameter>".len()..];
        }
        Some(ParsedToolCall {
            id: format!("call_warp34_{call_idx}"),
            call_type: "function".into(),
            function: ParsedFunctionCall {
                name: name.into(),
                arguments: serde_json::to_string(&params).ok()?,
            },
        })
    }

    /// Parses Qwen3-XML (`<function=name><parameter=k>v</parameter></function>`)
    /// inside a `<tool_call>...</tool_call>` body.
    pub fn parse_tool_body(raw_body: &str, call_idx: usize) -> Option<ParsedToolCall> {
        Self::parse_tool_body_with_tools(raw_body, call_idx, None, &ToolGrammars::new(), false)
    }

    fn parse_tool_body_with_tools(
        raw_body: &str,
        call_idx: usize,
        tools: Option<&[serde_json::Value]>,
        grammars: &ToolGrammars,
        finishing: bool,
    ) -> Option<ParsedToolCall> {
        if let Some(tools) = tools {
            return Self::parse_native_body(raw_body, call_idx, tools, grammars, finishing);
        }
        let body = raw_body.trim();
        if body.is_empty() {
            return None;
        }

        // Check Qwen3-XML format: `<function=func_name>...`
        let xml_fn = if let Some(after_lt) = body.strip_prefix('<') {
            let after_func = after_lt.trim_start();
            if let Some(after_func_word) = after_func.strip_prefix("function") {
                if let Some(after_eq) = after_func_word.trim_start().strip_prefix('=') {
                    let close_angle = after_eq.find('>')?;
                    let fn_name = after_eq[..close_angle].trim().to_string();
                    let rest = &after_eq[close_angle + 1..];
                    let fn_body = match Self::find_tool_delimiter(rest, "</function>") {
                        Some(end_idx) => &rest[..end_idx],
                        None => rest,
                    };
                    Some((fn_name, fn_body))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some((fn_name, fn_body)) = xml_fn {
            let mut params = serde_json::Map::new();
            let mut search = fn_body;
            while let Some(p_start) = Self::find_tool_delimiter(search, "<parameter=") {
                let after_p = &search[p_start..];
                let Some(p_angle) = after_p.find('>') else {
                    break;
                };
                let tag_inside = &after_p[1..p_angle];
                let Some((_, param_name)) = tag_inside.split_once('=') else {
                    break;
                };
                let param_name = param_name.trim().to_string();
                if param_name.is_empty() || params.contains_key(&param_name) {
                    return None;
                }
                let val_start = &after_p[p_angle + 1..];
                let (raw_val, next_slice) = if let Some(p_end) =
                    Self::find_tool_delimiter(val_start, "</parameter>")
                {
                    let after_close = match val_start[p_end..].find('>') {
                        Some(close_gt) => &val_start[p_end + close_gt + 1..],
                        None => "",
                    };
                    (&val_start[..p_end], after_close)
                } else if let Some(next_p) = Self::find_tool_delimiter(val_start, "<parameter=") {
                    (&val_start[..next_p], &val_start[next_p..])
                } else {
                    (val_start, "")
                };
                let trimmed = Self::trim_wrapping_newlines(raw_val);
                // Preserve structured JSON numbers/booleans/arrays/objects when valid JSON,
                // or store string value (`_qwen3_arg_converter`).
                let json_val = serde_json::from_str::<serde_json::Value>(trimmed)
                    .unwrap_or_else(|_| serde_json::Value::String(trimmed.to_string()));
                params.insert(param_name, json_val);
                search = next_slice;
            }

            let args_json =
                serde_json::to_string(&serde_json::Value::Object(params)).unwrap_or_default();
            return Some(ParsedToolCall {
                id: format!("call_warp34_{call_idx}"),
                call_type: "function".to_string(),
                function: ParsedFunctionCall {
                    name: fn_name,
                    arguments: args_json,
                },
            });
        }

        None
    }

    fn longest_tag_prefix_suffix(s: &str, tags: &[&str]) -> usize {
        let mut max_keep = 0usize;
        for &tag in tags {
            let max_check = (tag.len() - 1).min(s.len());
            for k in (1..=max_check).rev() {
                if s.ends_with(&tag[..k]) {
                    max_keep = max_keep.max(k);
                    break;
                }
            }
        }
        max_keep
    }

    fn escape_json_string_inner(s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }
        let quoted = serde_json::to_string(s).expect("valid str serializes to JSON");
        quoted[1..quoted.len() - 1].to_owned()
    }

    fn push_tool_header_delta(
        index: usize,
        id: String,
        name: String,
        out: &mut Vec<StreamToolCallDelta>,
    ) {
        if let Some(last) = out.last_mut().filter(|d| d.index == index) {
            if last.id.is_none() {
                last.id = Some(id);
            }
            if last.name.is_none() {
                last.name = Some(name);
            }
        } else {
            out.push(StreamToolCallDelta {
                index,
                id: Some(id),
                name: Some(name),
                arguments_delta: None,
            });
        }
    }

    fn push_tool_arg_delta(index: usize, arg_chunk: &str, out: &mut Vec<StreamToolCallDelta>) {
        if arg_chunk.is_empty() {
            return;
        }
        if let Some(last) = out.last_mut().filter(|d| d.index == index) {
            last.arguments_delta
                .get_or_insert_with(String::new)
                .push_str(arg_chunk);
        } else {
            out.push(StreamToolCallDelta {
                index,
                id: None,
                name: None,
                arguments_delta: Some(arg_chunk.to_owned()),
            });
        }
    }

    fn ensure_tool_header_emitted(&mut self, out: &mut Vec<StreamToolCallDelta>) {
        if !self.tool_stream.header_emitted
            && let Some(name) = self.tool_stream.func_name.clone()
        {
            self.tool_stream.header_emitted = true;
            Self::push_tool_header_delta(
                self.next_tool_idx,
                format!("call_warp34_{}", self.next_tool_idx),
                name,
                out,
            );
        }
    }

    fn emit_streamed_arg_chunk(&mut self, chunk: &str, out: &mut Vec<StreamToolCallDelta>) {
        if chunk.is_empty() {
            return;
        }
        self.ensure_tool_header_emitted(out);
        self.tool_stream.streamed_args.push_str(chunk);
        Self::push_tool_arg_delta(self.next_tool_idx, chunk, out);
    }

    fn properties_for_func(
        &self,
        func_name: &str,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.tools
            .as_deref()?
            .iter()
            .find(|t| t.pointer("/function/name").and_then(|v| v.as_str()) == Some(func_name))?
            .pointer("/function/parameters/properties")?
            .as_object()
    }

    fn is_unconstrained_string_param(&self, func_name: &str, param_name: &str) -> bool {
        let Some(schema) = self
            .properties_for_func(func_name)
            .and_then(|props| props.get(param_name))
        else {
            return false;
        };
        native_schema::is_unconstrained_string_schema(schema)
    }

    fn advance_native_tool_stream(
        &mut self,
        out_deltas: &mut Vec<StreamToolCallDelta>,
    ) -> NativeToolStep {
        if self.tool_stream.aborted {
            return NativeToolStep::Aborted;
        }
        let Some(tools) = self.tools.as_deref() else {
            return NativeToolStep::Aborted;
        };
        if tools.is_empty() {
            self.tool_stream.aborted = true;
            return NativeToolStep::Aborted;
        }

        if self.tool_stream.func_name.is_none() {
            const PREFIX: &str = "\n<function=";
            if !self.buffer.starts_with(PREFIX) {
                if PREFIX.starts_with(&self.buffer) && !self.finishing {
                    return NativeToolStep::InProgress;
                }
                self.tool_stream.aborted = true;
                return NativeToolStep::Aborted;
            }
            let after_prefix = &self.buffer[PREFIX.len()..];
            let Some(gt_pos) = after_prefix.find('>') else {
                if after_prefix.contains('\n') || self.finishing {
                    self.tool_stream.aborted = true;
                    return NativeToolStep::Aborted;
                }
                return NativeToolStep::InProgress;
            };
            if gt_pos + 1 >= after_prefix.len() {
                if self.finishing {
                    self.tool_stream.aborted = true;
                    return NativeToolStep::Aborted;
                }
                return NativeToolStep::InProgress;
            }
            if after_prefix.as_bytes()[gt_pos + 1] != b'\n' {
                self.tool_stream.aborted = true;
                return NativeToolStep::Aborted;
            }
            let name = &after_prefix[..gt_pos];
            let is_known = tools.iter().any(|tool| {
                tool.pointer("/function/name")
                    .and_then(serde_json::Value::as_str)
                    == Some(name)
            });
            if !is_known {
                self.tool_stream.aborted = true;
                return NativeToolStep::Aborted;
            }
            self.tool_stream.func_name = Some(name.to_owned());
            self.tool_stream.cursor = PREFIX.len() + gt_pos + 2;
        }

        let func_name = self.tool_stream.func_name.clone().unwrap_or_default();

        loop {
            if let Some(active) = self.tool_stream.active_param.clone() {
                match active {
                    ActiveParamStream::StreamString {
                        value_start,
                        mut value_scan_pos,
                        mut skipped_leading_nl,
                    } => {
                        if !skipped_leading_nl {
                            if self.buffer.len() <= value_start {
                                if self.finishing {
                                    self.tool_stream.aborted = true;
                                    return NativeToolStep::Aborted;
                                }
                                return NativeToolStep::InProgress;
                            }
                            if self.buffer.as_bytes()[value_start] == b'\n' {
                                value_scan_pos = value_start + 1;
                            } else {
                                value_scan_pos = value_start;
                            }
                            skipped_leading_nl = true;
                        }
                        let avail = &self.buffer[value_scan_pos..];
                        if let Some(end_rel) = avail.find("</parameter>") {
                            let rem_val = &avail[..end_rel];
                            let to_emit = rem_val.strip_suffix('\n').unwrap_or(rem_val);
                            let mut chunk = Self::escape_json_string_inner(to_emit);
                            chunk.push('"');
                            let next_cursor = value_scan_pos + end_rel + "</parameter>".len();
                            self.emit_streamed_arg_chunk(&chunk, out_deltas);
                            self.tool_stream.cursor = next_cursor;
                            self.tool_stream.active_param = None;
                            continue;
                        }
                        if self.finishing {
                            self.tool_stream.aborted = true;
                            return NativeToolStep::Aborted;
                        }
                        let keep = Self::longest_tag_prefix_suffix(
                            avail,
                            &["\n</parameter>", "</parameter>"],
                        );
                        let emit_len = avail.len() - keep;
                        if emit_len > 0 {
                            let escaped = Self::escape_json_string_inner(&avail[..emit_len]);
                            self.emit_streamed_arg_chunk(&escaped, out_deltas);
                            value_scan_pos += emit_len;
                        }
                        self.tool_stream.active_param = Some(ActiveParamStream::StreamString {
                            value_start,
                            value_scan_pos,
                            skipped_leading_nl,
                        });
                        return NativeToolStep::InProgress;
                    }
                    ActiveParamStream::Buffered {
                        name,
                        value_start,
                        is_first,
                    } => {
                        let properties = self.properties_for_func(&func_name);
                        let schema = properties.and_then(|p| p.get(&name));
                        let grammar = self
                            .value_grammars
                            .get(&func_name)
                            .and_then(|g| g.get(&name));
                        let end_opt = Self::native_parameter_end(
                            &self.buffer[value_start..],
                            schema,
                            grammar,
                            self.finishing,
                        );
                        if let Some(end_rel) = end_opt {
                            let raw_val = &self.buffer[value_start..value_start + end_rel];
                            let converted = Self::convert_arg(raw_val, schema, grammar);
                            if !native_schema::accepts(&converted, grammar) {
                                self.tool_stream.aborted = true;
                                return NativeToolStep::Aborted;
                            }
                            let key_json =
                                serde_json::to_string(&name).expect("param name serializes");
                            let val_json =
                                serde_json::to_string(&converted).unwrap_or_else(|_| "null".into());
                            let chunk = format!(
                                "{}{key_json}:{val_json}",
                                if is_first { "{" } else { "," }
                            );
                            let next_cursor = value_start + end_rel + "</parameter>".len();
                            self.emit_streamed_arg_chunk(&chunk, out_deltas);
                            self.tool_stream.cursor = next_cursor;
                            self.tool_stream.active_param = None;
                            continue;
                        }
                        if self.finishing {
                            self.tool_stream.aborted = true;
                            return NativeToolStep::Aborted;
                        }
                        return NativeToolStep::InProgress;
                    }
                }
            }

            let tail = &self.buffer[self.tool_stream.cursor..];
            let trimmed_after_ws = tail.trim_start_matches([' ', '\n', '\r', '\t']);
            let ws_len = tail.len() - trimmed_after_ws.len();

            if trimmed_after_ws.is_empty() {
                if self.finishing {
                    self.tool_stream.aborted = true;
                    return NativeToolStep::Aborted;
                }
                return NativeToolStep::InProgress;
            }

            if let Some(header) = trimmed_after_ws.strip_prefix("<parameter=") {
                let Some(angle) = header.find('>') else {
                    if self.finishing
                        || header
                            .bytes()
                            .any(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
                    {
                        self.tool_stream.aborted = true;
                        return NativeToolStep::Aborted;
                    }
                    return NativeToolStep::InProgress;
                };
                let param_name = &header[..angle];
                if param_name.is_empty()
                    || param_name.trim() != param_name
                    || self.tool_stream.seen_params.iter().any(|p| p == param_name)
                {
                    self.tool_stream.aborted = true;
                    return NativeToolStep::Aborted;
                }
                let param_name_owned = param_name.to_owned();
                let val_start = self.tool_stream.cursor + ws_len + "<parameter=".len() + angle + 1;
                let is_first = self.tool_stream.seen_params.is_empty();
                self.tool_stream.seen_params.push(param_name_owned.clone());
                self.tool_stream.cursor = val_start;

                let is_stream_str =
                    self.is_unconstrained_string_param(&func_name, &param_name_owned);
                self.tool_stream.active_param = Some(if is_stream_str {
                    let key_json =
                        serde_json::to_string(&param_name_owned).expect("param name serializes");
                    let prefix = format!("{}{key_json}:\"", if is_first { "{" } else { "," });
                    self.emit_streamed_arg_chunk(&prefix, out_deltas);
                    ActiveParamStream::StreamString {
                        value_start: val_start,
                        value_scan_pos: val_start,
                        skipped_leading_nl: false,
                    }
                } else {
                    ActiveParamStream::Buffered {
                        name: param_name_owned,
                        value_start: val_start,
                        is_first,
                    }
                });
                continue;
            }

            if "<parameter=".starts_with(trimmed_after_ws) {
                if self.finishing {
                    self.tool_stream.aborted = true;
                    return NativeToolStep::Aborted;
                }
                return NativeToolStep::InProgress;
            }

            const CLOSE_SEQ: &str = "</function>\n</tool_call>";
            if tail[..ws_len].ends_with('\n') {
                if trimmed_after_ws.starts_with(CLOSE_SEQ) {
                    let drain_end = self.tool_stream.cursor + ws_len + CLOSE_SEQ.len();
                    let closing = if self.tool_stream.seen_params.is_empty() {
                        "{}"
                    } else {
                        "}"
                    };
                    self.emit_streamed_arg_chunk(closing, out_deltas);
                    return NativeToolStep::Completed(drain_end);
                }
                if CLOSE_SEQ.starts_with(trimmed_after_ws) {
                    if self.finishing {
                        self.tool_stream.aborted = true;
                        return NativeToolStep::Aborted;
                    }
                    return NativeToolStep::InProgress;
                }
            }

            self.tool_stream.aborted = true;
            return NativeToolStep::Aborted;
        }
    }

    fn advance_legacy_tool_stream(&mut self, out_deltas: &mut Vec<StreamToolCallDelta>) {
        if self.tool_stream.aborted || self.tools.is_some() {
            return;
        }
        if self.tool_stream.func_name.is_none() {
            let trimmed = self.buffer.trim_start_matches([' ', '\n', '\r', '\t']);
            if trimmed.is_empty() {
                return;
            }
            if !trimmed.starts_with('<') {
                self.tool_stream.aborted = true;
                return;
            }
            let after_lt = trimmed[1..].trim_start_matches([' ', '\n', '\r', '\t']);
            if after_lt.is_empty() {
                return;
            }
            let Some(after_func) = after_lt.strip_prefix("function") else {
                if !"function".starts_with(after_lt) {
                    self.tool_stream.aborted = true;
                }
                return;
            };
            let after_func_ws = after_func.trim_start_matches([' ', '\n', '\r', '\t']);
            if after_func_ws.is_empty() {
                return;
            }
            if !after_func_ws.starts_with('=') {
                self.tool_stream.aborted = true;
                return;
            }
            let after_eq = &after_func_ws[1..];
            let Some(close_angle) = after_eq.find('>') else {
                return;
            };
            let fn_name = after_eq[..close_angle].trim().to_string();
            if fn_name.is_empty() {
                self.tool_stream.aborted = true;
                return;
            }
            let eq_offset = self.buffer.len() - after_eq.len();
            self.tool_stream.cursor = eq_offset + close_angle + 1;
            self.tool_stream.func_name = Some(fn_name);
            self.ensure_tool_header_emitted(out_deltas);
        }

        loop {
            if let Some(active) = self.tool_stream.active_param.clone() {
                match active {
                    ActiveParamStream::StreamString {
                        value_start,
                        mut value_scan_pos,
                        mut skipped_leading_nl,
                    } => {
                        if !skipped_leading_nl {
                            if self.buffer.len() <= value_start {
                                return;
                            }
                            if self.buffer.as_bytes()[value_start] == b'\n' {
                                value_scan_pos = value_start + 1;
                            } else {
                                value_scan_pos = value_start;
                            }
                            skipped_leading_nl = true;
                        }
                        let avail = &self.buffer[value_scan_pos..];
                        let close_param = Self::find_tool_delimiter(avail, "</parameter>");
                        let next_param = Self::find_tool_delimiter(avail, "<parameter=");
                        let close_fn = Self::find_tool_delimiter(avail, "</function>");
                        let close_tc = Self::find_tool_delimiter(avail, Self::TOOL_END);
                        let earliest_delim = [
                            close_param.map(|p| (p, 0u8)),
                            next_param.map(|p| (p, 1u8)),
                            close_fn.map(|p| (p, 2u8)),
                            close_tc.map(|p| (p, 2u8)),
                        ]
                        .into_iter()
                        .flatten()
                        .min_by_key(|&(pos, _)| pos);

                        if let Some((end_rel, delim_kind)) = earliest_delim {
                            let tag_advance = if delim_kind == 0 {
                                let Some(gt_rel) = avail[end_rel..].find('>') else {
                                    return;
                                };
                                end_rel + gt_rel + 1
                            } else {
                                end_rel
                            };
                            let rem_val = &avail[..end_rel];
                            let to_emit = rem_val.strip_suffix('\n').unwrap_or(rem_val);
                            let mut chunk = Self::escape_json_string_inner(to_emit);
                            chunk.push('"');
                            let next_cursor = value_scan_pos + tag_advance;
                            self.emit_streamed_arg_chunk(&chunk, out_deltas);
                            self.tool_stream.cursor = next_cursor;
                            self.tool_stream.active_param = None;
                            if delim_kind <= 1 {
                                continue;
                            }
                            return;
                        }

                        let unclosed_lt = avail.rfind('<').filter(|&lt| !avail[lt..].contains('>'));
                        let hold_from = match unclosed_lt {
                            Some(lt) if lt > 0 && avail.as_bytes()[lt - 1] == b'\n' => lt - 1,
                            Some(lt) => lt,
                            None if avail.ends_with('\n') => avail.len() - 1,
                            None => avail.len(),
                        };
                        if hold_from > 0 {
                            let escaped = Self::escape_json_string_inner(&avail[..hold_from]);
                            self.emit_streamed_arg_chunk(&escaped, out_deltas);
                            value_scan_pos += hold_from;
                        }
                        self.tool_stream.active_param = Some(ActiveParamStream::StreamString {
                            value_start,
                            value_scan_pos,
                            skipped_leading_nl,
                        });
                        return;
                    }
                    ActiveParamStream::Buffered {
                        name,
                        value_start,
                        is_first,
                    } => {
                        let avail = &self.buffer[value_start..];
                        let close_param = Self::find_tool_delimiter(avail, "</parameter>");
                        let next_param = Self::find_tool_delimiter(avail, "<parameter=");
                        let close_fn = Self::find_tool_delimiter(avail, "</function>");
                        let close_tc = Self::find_tool_delimiter(avail, Self::TOOL_END);
                        let earliest_delim = [
                            close_param.map(|p| (p, 0u8)),
                            next_param.map(|p| (p, 1u8)),
                            close_fn.map(|p| (p, 2u8)),
                            close_tc.map(|p| (p, 2u8)),
                        ]
                        .into_iter()
                        .flatten()
                        .min_by_key(|&(pos, _)| pos);
                        if let Some((end_rel, delim_kind)) = earliest_delim {
                            let tag_advance = if delim_kind == 0 {
                                let Some(gt_rel) = avail[end_rel..].find('>') else {
                                    return;
                                };
                                end_rel + gt_rel + 1
                            } else {
                                end_rel
                            };
                            let raw_val = &avail[..end_rel];
                            let trimmed = Self::trim_wrapping_newlines(raw_val);
                            let json_val = serde_json::from_str::<serde_json::Value>(trimmed)
                                .unwrap_or_else(|_| serde_json::Value::String(trimmed.to_string()));
                            let key_json =
                                serde_json::to_string(&name).expect("param name serializes");
                            let val_json =
                                serde_json::to_string(&json_val).unwrap_or_else(|_| "null".into());
                            let chunk = format!(
                                "{}{key_json}:{val_json}",
                                if is_first { "{" } else { "," }
                            );
                            let next_cursor = value_start + tag_advance;
                            self.emit_streamed_arg_chunk(&chunk, out_deltas);
                            self.tool_stream.cursor = next_cursor;
                            self.tool_stream.active_param = None;
                            if delim_kind <= 1 {
                                continue;
                            }
                            return;
                        }
                        return;
                    }
                }
            }

            let tail = &self.buffer[self.tool_stream.cursor..];
            let Some(p_rel) = Self::find_tool_delimiter(tail, "<parameter=") else {
                return;
            };
            let close_fn = Self::find_tool_delimiter(tail, "</function>");
            let close_tc = Self::find_tool_delimiter(tail, Self::TOOL_END);
            if close_fn.is_some_and(|fn_pos| fn_pos < p_rel)
                || close_tc.is_some_and(|tc_pos| tc_pos < p_rel)
            {
                return;
            }
            let after_p = &tail[p_rel..];
            let Some(eq_pos) = after_p.find('=') else {
                return;
            };
            let Some(gt_pos) = after_p[eq_pos + 1..].find('>') else {
                return;
            };
            let param_name = after_p[eq_pos + 1..eq_pos + 1 + gt_pos].trim().to_string();
            if param_name.is_empty()
                || self
                    .tool_stream
                    .seen_params
                    .iter()
                    .any(|p| p == &param_name)
            {
                self.tool_stream.aborted = true;
                return;
            }
            let val_start = self.tool_stream.cursor + p_rel + eq_pos + 1 + gt_pos + 1;
            if self.buffer.len() <= val_start {
                return;
            }
            let first_idx = if self.buffer.as_bytes()[val_start] == b'\n' {
                if self.buffer.len() <= val_start + 1 {
                    return;
                }
                val_start + 1
            } else {
                val_start
            };
            let first_b = self.buffer.as_bytes()[first_idx];
            let is_stream_str = !matches!(
                first_b,
                b'"' | b'{' | b'[' | b'0'
                    ..=b'9' | b'-' | b't' | b'f' | b'n' | b' ' | b'\t' | b'\r' | b'\n' | b'<'
            );
            let is_first = self.tool_stream.seen_params.is_empty();
            self.tool_stream.seen_params.push(param_name.clone());
            self.tool_stream.cursor = val_start;
            self.tool_stream.active_param = Some(if is_stream_str {
                let key_json = serde_json::to_string(&param_name).expect("param name serializes");
                let prefix = format!("{}{key_json}:\"", if is_first { "{" } else { "," });
                self.emit_streamed_arg_chunk(&prefix, out_deltas);
                ActiveParamStream::StreamString {
                    value_start: val_start,
                    value_scan_pos: val_start,
                    skipped_leading_nl: false,
                }
            } else {
                ActiveParamStream::Buffered {
                    name: param_name,
                    value_start: val_start,
                    is_first,
                }
            });
        }
    }

    fn push_content_slice(&mut self, raw: &str, c_out: &mut String) {
        let text = if self.content_accum.is_empty() {
            raw.trim_start_matches('\n')
        } else if !self.tool_calls.is_empty() && raw.bytes().all(|b| b == b'\n') {
            ""
        } else {
            raw
        };
        if !text.is_empty() {
            c_out.push_str(text);
            self.content_accum.push_str(text);
        }
    }

    /// Feeds a streaming text chunk and returns incremental `reasoning_delta`, `content_delta`,
    /// and any newly completed `tool_calls`.
    pub fn push_str(&mut self, chunk: &str) -> StreamParseDelta {
        self.buffer.push_str(chunk);
        let mut delta = StreamParseDelta::default();
        let mut r_out = String::new();
        let mut c_out = String::new();

        loop {
            if self.in_tool_call {
                if self.tools.is_some() {
                    match self.advance_native_tool_stream(&mut delta.tool_call_deltas) {
                        NativeToolStep::InProgress => break,
                        NativeToolStep::Completed(drain_end) => {
                            let tc = ParsedToolCall {
                                id: format!("call_warp34_{}", self.next_tool_idx),
                                call_type: "function".to_string(),
                                function: ParsedFunctionCall {
                                    name: self.tool_stream.func_name.take().unwrap_or_default(),
                                    arguments: std::mem::take(&mut self.tool_stream.streamed_args),
                                },
                            };
                            self.next_tool_idx += 1;
                            self.tool_calls.push(tc.clone());
                            delta.completed_tool_calls.push(tc);
                            self.buffer.drain(..drain_end);
                            self.in_tool_call = false;
                            self.tool_stream = ToolStreamState::default();
                            continue;
                        }
                        NativeToolStep::Aborted => {}
                    }
                } else {
                    self.advance_legacy_tool_stream(&mut delta.tool_call_deltas);
                }

                let properties = self
                    .tools
                    .as_deref()
                    .and_then(|tools| Self::properties(&self.buffer, tools));
                let end = if self.tools.is_some() {
                    Self::native_tool_end(
                        &self.buffer,
                        properties,
                        self.buffer
                            .strip_prefix("\n<function=")
                            .and_then(|s| s.split_once('>'))
                            .and_then(|(name, _)| self.value_grammars.get(name)),
                        self.finishing,
                    )
                } else {
                    Self::find_tool_delimiter(&self.buffer, Self::TOOL_END)
                };
                if let Some(end_pos) = end {
                    let body = self.buffer[..end_pos].to_string();
                    let drain_end = match self.buffer[end_pos..].find('>') {
                        Some(gt) => end_pos + gt + 1,
                        None => end_pos + Self::TOOL_END.len(),
                    };
                    if let Some(tc) = Self::parse_tool_body_with_tools(
                        &body,
                        self.next_tool_idx,
                        self.tools.as_deref(),
                        &self.value_grammars,
                        self.finishing,
                    ) {
                        if !self.tool_stream.header_emitted {
                            Self::push_tool_header_delta(
                                self.next_tool_idx,
                                tc.id.clone(),
                                tc.function.name.clone(),
                                &mut delta.tool_call_deltas,
                            );
                            Self::push_tool_arg_delta(
                                self.next_tool_idx,
                                &tc.function.arguments,
                                &mut delta.tool_call_deltas,
                            );
                        } else if let Some(rem) = tc
                            .function
                            .arguments
                            .strip_prefix(&self.tool_stream.streamed_args)
                        {
                            Self::push_tool_arg_delta(
                                self.next_tool_idx,
                                rem,
                                &mut delta.tool_call_deltas,
                            );
                        }
                        self.next_tool_idx += 1;
                        self.tool_calls.push(tc.clone());
                        delta.completed_tool_calls.push(tc);
                    } else {
                        // Unrecognized envelopes remain text rather than being dispatched or lost.
                        let content = format!("{}{}", Self::TOOL_START, &self.buffer[..drain_end]);
                        if !self.tool_stream.header_emitted {
                            c_out.push_str(&content);
                        }
                        self.content_accum.push_str(&content);
                    }
                    self.buffer.drain(..drain_end);
                    self.in_tool_call = false;
                    self.tool_stream = ToolStreamState::default();
                    continue;
                }
                break;
            } else if self.in_reasoning {
                // Absorb leading `<think>` if emitted explicitly
                if let Some(rest) = self.buffer.strip_prefix(Self::THINK_START) {
                    self.buffer = rest.to_string();
                    continue;
                }
                let think_end = self.buffer.find(Self::THINK_END);
                let tool_start = self
                    .tools
                    .as_ref()
                    .is_none_or(|tools| !tools.is_empty())
                    .then(|| self.buffer.find(Self::TOOL_START))
                    .flatten();
                match (think_end, tool_start) {
                    (Some(te), Some(ts)) if ts < te => {
                        // Implicit `</think>` on `<tool_call>` directly from reasoning
                        let text = &self.buffer[..ts];
                        r_out.push_str(text);
                        self.reasoning_accum.push_str(text);
                        self.buffer.drain(..ts + Self::TOOL_START.len());
                        self.in_reasoning = false;
                        self.in_tool_call = true;
                        self.tool_stream = ToolStreamState::default();
                    }
                    (Some(te), _) => {
                        let text = &self.buffer[..te];
                        r_out.push_str(text);
                        self.reasoning_accum.push_str(text);
                        self.buffer.drain(..te + Self::THINK_END.len());
                        self.in_reasoning = false;
                    }
                    (None, Some(ts)) => {
                        let text = &self.buffer[..ts];
                        r_out.push_str(text);
                        self.reasoning_accum.push_str(text);
                        self.buffer.drain(..ts + Self::TOOL_START.len());
                        self.in_reasoning = false;
                        self.in_tool_call = true;
                        self.tool_stream = ToolStreamState::default();
                    }
                    (None, None) => {
                        let keep = Self::longest_tag_prefix_suffix(
                            &self.buffer,
                            &[Self::THINK_START, Self::THINK_END, Self::TOOL_START],
                        );
                        let emit_len = self.buffer.len() - keep;
                        if emit_len > 0 {
                            let text = &self.buffer[..emit_len];
                            r_out.push_str(text);
                            self.reasoning_accum.push_str(text);
                            self.buffer.drain(..emit_len);
                        }
                        break;
                    }
                }
                continue;
            }
            // In CONTENT state: check for `<think>` (if thinking wasn't initially active),
            // absorb duplicate `</think>`, or transition to `<tool_call>`.
            let think_start = self.buffer.find(Self::THINK_START);
            let dup_think_end = self.buffer.find(Self::THINK_END);
            let tool_start = self
                .tools
                .as_ref()
                .is_none_or(|tools| !tools.is_empty())
                .then(|| self.buffer.find(Self::TOOL_START))
                .flatten();

            let earliest = [
                think_start.map(|p| (p, 0u8)),
                dup_think_end.map(|p| (p, 1u8)),
                tool_start.map(|p| (p, 2u8)),
            ]
            .into_iter()
            .flatten()
            .min_by_key(|&(pos, _)| pos);

            if let Some((pos, kind)) = earliest {
                let text = self.buffer[..pos].to_string();
                self.push_content_slice(&text, &mut c_out);
                match kind {
                    0 => {
                        self.buffer.drain(..pos + Self::THINK_START.len());
                        self.in_reasoning = true;
                    }
                    1 => {
                        // Absorb duplicate `</think>` silently (`vllm/parser/qwen3.py:130-134`)
                        self.buffer.drain(..pos + Self::THINK_END.len());
                    }
                    _ => {
                        self.buffer.drain(..pos + Self::TOOL_START.len());
                        self.in_tool_call = true;
                        self.tool_stream = ToolStreamState::default();
                    }
                }
                continue;
            }

            let keep = Self::longest_tag_prefix_suffix(
                &self.buffer,
                &[Self::THINK_START, Self::THINK_END, Self::TOOL_START],
            );
            let emit_len = self.buffer.len() - keep;
            if emit_len > 0 {
                let text = self.buffer[..emit_len].to_string();
                self.push_content_slice(&text, &mut c_out);
                self.buffer.drain(..emit_len);
            }
            break;
        }

        if !r_out.is_empty() {
            delta.reasoning_delta = Some(r_out);
        }
        if !c_out.is_empty() {
            delta.content_delta = Some(c_out);
        }
        delta
    }

    /// Flushes any trailing buffered text at end-of-sequence and returns the final turn.
    pub fn finish(&mut self) -> (StreamParseDelta, ParsedAssistantTurn) {
        self.finishing = true;
        let mut delta = self.push_str("");
        if !self.buffer.is_empty() {
            let rem = std::mem::take(&mut self.buffer);
            if self.in_tool_call {
                // A truncated call is never executable. In particular, a
                // delimiter inside an unfinished JSON string is not a close tag.
                let fallback = format!("{}{rem}", Self::TOOL_START);
                self.content_accum.push_str(&fallback);
                if !self.tool_stream.header_emitted {
                    delta
                        .content_delta
                        .get_or_insert_default()
                        .push_str(&fallback);
                }
                self.in_tool_call = false;
                self.tool_stream = ToolStreamState::default();
            } else if self.in_reasoning {
                self.reasoning_accum.push_str(&rem);
                delta.reasoning_delta.get_or_insert_default().push_str(&rem);
            } else {
                let mut c_out = String::new();
                self.push_content_slice(&rem, &mut c_out);
                if !c_out.is_empty() {
                    delta.content_delta.get_or_insert_default().push_str(&c_out);
                }
            }
        }

        let trimmed_content = self.content_accum.trim_matches('\n');
        let turn = ParsedAssistantTurn {
            reasoning_content: if self.reasoning_accum.is_empty() {
                None
            } else {
                Some(self.reasoning_accum.clone())
            },
            content: if trimmed_content.is_empty() {
                None
            } else {
                Some(trimmed_content.to_string())
            },
            tool_calls: self.tool_calls.clone(),
        };
        (delta, turn)
    }

    /// One-shot helper for non-streaming completions.
    pub fn parse_complete(text: &str, enable_thinking: bool) -> ParsedAssistantTurn {
        let mut p = Self::new(enable_thinking);
        let _ = p.push_str(text);
        p.finish().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_limits_fail_without_weakening_or_mutating_schema() {
        use serde_json::json;
        let mut defs = serde_json::Map::new();
        for i in 0..40 {
            defs.insert(
                format!("s{i}"),
                json!({"$ref":format!("#/$defs/s{}", i + 1)}),
            );
        }
        defs.insert("s40".into(), json!({"type":"string"}));
        let deep =
            json!({"$defs":defs,"type":"object","properties":{"label":{"$ref":"#/$defs/s0"}}});
        let mut properties = serde_json::Map::new();
        for i in 0..4096 {
            properties.insert(format!("label{i}"), json!({"type":"string"}));
        }
        let wide = json!({"type":"object","properties":properties});
        for (schema, expected) in [(deep, "depth limit"), (wide, "budget")] {
            let tools = vec![json!({"function":{"name":"emit","parameters":schema}})];
            let original = tools.clone();
            let error = Qwen3Parser::with_tools(false, &tools).unwrap_err();
            assert!(error.contains(expected), "{error}");
            assert_eq!(tools, original);
        }
    }

    #[test]
    fn recursive_json_argument_preserves_nested_strings() {
        use serde_json::json;
        let tools = vec![json!({"function":{"name":"emit","parameters":{
            "$defs":{"node":{"type":"object","properties":{
                "label":{"type":"string"}, "next":{"$ref":"#/$defs/node"}
            }}},
            "type":"object","properties":{"node":{"$ref":"#/$defs/node"}}
        }}})];
        let original = tools.clone();
        let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
        parser.push_str(
            r#"<tool_call>
<function=emit>
<parameter=node>{"label":"123","next":{"label":"true"}}</parameter>
</function>
</tool_call>"#,
        );
        let turn = parser.finish().1;
        let args: serde_json::Value =
            serde_json::from_str(&turn.tool_calls[0].function.arguments).unwrap();
        assert_eq!(
            args,
            json!({"node":{"label":"123","next":{"label":"true"}}})
        );
        assert_eq!(tools, original);
    }

    #[test]
    fn native_refs_consts_and_untyped_json() {
        use serde_json::json;
        let tools = vec![json!({"function":{"name":"emit","parameters":{
            "$defs": {
                "count":{"const":123}, "label":{"type":"string"},
                "loop":{"type":"object","properties":{"next":{"$ref":"#/$defs/loop"}}},
                "props":{"type":"object","properties":{
                    "count":{"$ref":"#/$defs/count"},
                    "label":{"$ref":"#/$defs/label"},
                    "data":{}, "recursive":{"$ref":"#/$defs/loop"}
                }}
            },
            "$ref":"#/$defs/props"
        }}})];
        let original = tools.clone();
        for label in ["true", "123"] {
            let text = format!(
                "<tool_call>\n<function=emit>\n<parameter=count>123</parameter><parameter=label>{label}</parameter><parameter=data>{{\"nested\":[\"</parameter>\",\"</function>\",\"</tool_call>\",{{\"ok\":true}}]}}</parameter>\n</function>\n</tool_call>"
            );
            for split in 0..=text.len() {
                let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
                parser.push_str(&text[..split]);
                parser.push_str(&text[split..]);
                let turn = parser.finish().1;
                assert_eq!(turn.tool_calls.len(), 1, "split {split}");
                let args: serde_json::Value =
                    serde_json::from_str(&turn.tool_calls[0].function.arguments).unwrap();
                assert_eq!(
                    args,
                    json!({"count":123,"label":label,"data":{
                        "nested":["</parameter>","</function>","</tool_call>",{"ok":true}]
                    }}),
                    "split {split}"
                );
            }
        }
        assert_eq!(tools, original);
    }

    #[test]
    fn native_qwen_unsigned_integer_preserves_numeric_type() {
        let tools = [
            serde_json::json!({"type":"function","function":{"name":"emit",
            "parameters":{"type":"object","properties":{"value":{"type":"integer"}}}}}),
        ];
        let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
        parser.push_str("<tool_call>\n<function=emit>\n<parameter=value>18446744073709551615</parameter>\n</function>\n</tool_call>");
        let turn = parser.finish().1;
        let args: serde_json::Value =
            serde_json::from_str(&turn.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["value"], serde_json::json!(u64::MAX));
    }

    #[test]
    fn native_qwen_schema_types_and_streaming() {
        use serde_json::json;
        let tools = vec![json!({"type":"function", "function": {
            "name":"emit", "parameters":{"type":"object", "properties": {
                "label":{"type":"string"}, "count":{"type":"integer"},
                "enabled":{"type":"boolean"}, "data":{"type":"object","additionalProperties":true},
                "items":{"type":"array","items":true}, "maybe":{"type":["string","null"]},
                "choice":{"enum":["true","123"]},
                "union":{"anyOf":[{"type":"integer"},{"type":"string"}]}
            }}
        }})];
        for label in [
            "true",
            "123",
            "\"quoted\"",
            "\"unclosed",
            "trailing\"",
            "{not JSON",
            "[raw",
            "  spaced  ",
            "héllo",
        ] {
            let text = format!(
                "<tool_call>\n<function=emit>\n<parameter=label>{label}</parameter><parameter=count>123</parameter><parameter=enabled>true</parameter><parameter=data>{{\"nested\":[true,123,\"</tool_call>\",\"</parameter>\"]}}</parameter><parameter=items>[1,{{\"x\":false}}]</parameter><parameter=maybe>null</parameter><parameter=choice>true</parameter><parameter=union>42</parameter><parameter=unknown>123</parameter>\n</function>\n</tool_call>"
            );
            let expected = json!({"label":label, "count":123, "enabled":true,
                "data":{"nested":[true,123,"</tool_call>","</parameter>"]},
                "items":[1,{"x":false}], "maybe":null, "choice":"true", "union":42,
                "unknown":"123"});
            // Every two-chunk split, including splits in tags and raw quotes.
            for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
                let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
                let a = parser.push_str(&text[..split]);
                let b = parser.push_str(&text[split..]);
                assert_eq!(
                    a.completed_tool_calls.len() + b.completed_tool_calls.len(),
                    1,
                    "{label:?} split {split}"
                );
                let turn = parser.finish().1;
                assert_eq!(turn.content, None);
                let args: serde_json::Value =
                    serde_json::from_str(&turn.tool_calls[0].function.arguments).unwrap();
                assert_eq!(args, expected, "{label:?} split {split}");
            }
            let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
            for c in text.chars() {
                parser.push_str(&c.to_string());
            }
            assert_eq!(parser.finish().1.tool_calls.len(), 1);
        }
    }

    #[test]
    fn native_qwen_truncated_raw_string() {
        let tools = [
            serde_json::json!({"function":{"name":"emit","parameters":{"type":"object","properties":{"label":{"type":"string"}}}}}),
        ];
        let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
        parser.push_str("<tool_call>\n<function=emit>\n<parameter=label>\"unfinished");
        let turn = parser.finish().1;
        assert!(turn.tool_calls.is_empty());
        assert!(turn.content.unwrap().ends_with("\"unfinished"));
    }

    #[test]
    fn native_qwen_strips_single_wrapping_newlines_like_vllm() {
        use serde_json::json;
        let tools = [json!({"type":"function","function":{
            "name":"bash",
            "parameters":{"type":"object","properties":{
                "command":{"type":"string"},
                "raw":{},
                "spaced":{"type":"string"}
            }}
        }})];
        let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
        parser.push_str(
            "<tool_call>\n<function=bash>\n<parameter=command>\nnode --version\n</parameter>\n<parameter=raw>\nhello world\n</parameter>\n<parameter=spaced>\n  spaced  \n</parameter>\n</function>\n</tool_call>",
        );
        let turn = parser.finish().1;
        assert_eq!(turn.tool_calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&turn.tool_calls[0].function.arguments).unwrap();
        assert_eq!(
            args,
            json!({
                "command": "node --version",
                "raw": "hello world",
                "spaced": "  spaced  "
            })
        );
    }

    #[test]
    fn fast_greedy_stateless_classification() {
        let base = SamplingParams {
            temperature: 0.0,
            logprobs: true,
            top_logprobs: 0,
            ignore_eos: true,
            ..SamplingParams::default()
        };
        assert!(SamplerV1::new(base.clone(), &[], None).is_fast_greedy_stateless(2));

        let top_k_one = SamplingParams {
            temperature: 0.8,
            top_k: 1,
            logprobs: true,
            ..base.clone()
        };
        assert!(SamplerV1::new(top_k_one, &[], None).is_fast_greedy_stateless(2));

        // Plain greedy rows are also stateless-fast-greedy compatible (allowing mixed
        // plain + logprob=0 batches to pipeline together), while an all-plain batch
        // selects the lighter `PlainGreedy` graph via `!needs_gpu`.
        let plain = SamplingParams {
            temperature: 0.0,
            logprobs: false,
            ..SamplingParams::default()
        };
        assert!(plain.is_plain_greedy(false));
        assert!(SamplerV1::new(plain, &[], None).is_fast_greedy_stateless(2));

        // Stateful or stochastic parameters must not be classified as stateless fast greedy.
        for bad in [
            SamplingParams {
                temperature: 0.7,
                ..base.clone()
            },
            SamplingParams {
                top_logprobs: 1,
                ..base.clone()
            },
            SamplingParams {
                min_tokens: 1,
                ignore_eos: false,
                ..base.clone()
            },
            SamplingParams {
                repetition_penalty: 1.1,
                ..base.clone()
            },
            SamplingParams {
                frequency_penalty: 0.1,
                ..base.clone()
            },
            SamplingParams {
                presence_penalty: 0.1,
                ..base.clone()
            },
            SamplingParams {
                stop_strings: vec!["STOP".into()],
                ..base.clone()
            },
        ] {
            assert!(!SamplerV1::new(bad, &[], None).is_fast_greedy_stateless(2));
        }

        let mask = TokenBitmaskConstraint::from_allowed_tokens(1024, vec![1, 2, 3]);
        assert!(!SamplerV1::new(base.clone(), &[], Some(mask)).is_fast_greedy_stateless(2));

        let mut too_many_biases = base;
        for tok in 0..27 {
            too_many_biases.logit_bias.insert(tok, 1.0);
        }
        assert!(SamplerV1::new(too_many_biases.clone(), &[], None).is_fast_greedy_stateless(1));
        assert!(!SamplerV1::new(too_many_biases, &[], None).is_fast_greedy_stateless(2));
    }

    #[test]
    fn test_sampling_params_validation_bounds() {
        let valid = SamplingParams::default();
        assert!(valid.validate().is_ok());

        let invalid_temp = SamplingParams {
            temperature: -0.1,
            ..Default::default()
        };
        assert!(invalid_temp.validate().is_err());

        let invalid_top_k = SamplingParams {
            top_k: -2,
            ..Default::default()
        };
        assert!(invalid_top_k.validate().is_err());

        let valid_top_k_neg1 = SamplingParams {
            top_k: -1, // -1 means disabled, valid in HF
            ..Default::default()
        };
        assert!(valid_top_k_neg1.validate().is_ok());

        let invalid_top_p = SamplingParams {
            top_p: 1.5,
            ..Default::default()
        };
        assert!(invalid_top_p.validate().is_err());

        let invalid_min_p = SamplingParams {
            min_p: -0.05,
            ..Default::default()
        };
        assert!(invalid_min_p.validate().is_err());

        let invalid_rep_pen = SamplingParams {
            repetition_penalty: -1.0,
            ..Default::default()
        };
        assert!(invalid_rep_pen.validate().is_err());
    }

    #[test]
    fn test_qwen3_parser_reasoning_and_content_separation() {
        let mut parser = Qwen3Parser::new(true);

        let delta1 = parser.push_str("<think>Let me calculate ");
        assert_eq!(delta1.reasoning_delta.as_deref(), Some("Let me calculate "));
        assert_eq!(delta1.content_delta, None);

        let delta2 = parser.push_str("2 + 2 = 4.</think>The answer is 4.");
        assert_eq!(delta2.reasoning_delta.as_deref(), Some("2 + 2 = 4."));
        assert_eq!(delta2.content_delta.as_deref(), Some("The answer is 4."));

        let (_, turn) = parser.finish();
        assert_eq!(
            turn.reasoning_content.as_deref(),
            Some("Let me calculate 2 + 2 = 4.")
        );
        assert_eq!(turn.content.as_deref(), Some("The answer is 4."));
    }

    #[test]
    fn streaming_parser_strips_template_newlines_after_think_and_between_tools() {
        let mut parser = Qwen3Parser::new(true);
        let d1 = parser.push_str("thinking...</think>");
        assert_eq!(d1.reasoning_delta.as_deref(), Some("thinking..."));
        assert_eq!(d1.content_delta, None);

        // Template separator `\n\n` emitted as a separate token right after `</think>` must not
        // produce a spurious `content_delta: Some("\n\n")` before `<tool_call>` or content.
        let d2 = parser.push_str("\n\n");
        assert_eq!(d2.content_delta, None);

        let d3 = parser.push_str(
            "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n",
        );
        assert_eq!(d3.content_delta, None);
        assert_eq!(d3.completed_tool_calls.len(), 1);

        let (df, turn) = parser.finish();
        assert_eq!(df.content_delta, None);
        assert_eq!(turn.content, None);
        assert_eq!(turn.tool_calls.len(), 1);
    }

    #[test]
    fn test_qwen3_parser_xml_tool_call_extraction() {
        let mut parser = Qwen3Parser::new(false);
        let stream_text = "<tool_call><function=get_current_weather><parameter=location>Tokyo</parameter></function></tool_call>";
        let delta = parser.push_str(stream_text);

        assert_eq!(delta.completed_tool_calls.len(), 1);
        let tc = &delta.completed_tool_calls[0];
        assert_eq!(tc.function.name, "get_current_weather");
        assert_eq!(tc.function.arguments, "{\"location\":\"Tokyo\"}");
    }

    #[test]
    fn test_qwen3_parser_xml_whitespace_tolerance() {
        let mut parser = Qwen3Parser::new(false);
        // Models occasionally emit spaces around delimiters: <function = name >, <parameter = key >, and </ tool_call >
        let stream_text = "<tool_call><function = get_current_weather ><parameter = location >Tokyo</ parameter ></ function ></ tool_call >";
        let delta = parser.push_str(stream_text);

        assert_eq!(delta.completed_tool_calls.len(), 1);
        let tc = &delta.completed_tool_calls[0];
        assert_eq!(tc.function.name, "get_current_weather");
        assert_eq!(tc.function.arguments, "{\"location\":\"Tokyo\"}");
        assert_eq!(delta.content_delta, None);

        let (_, turn) = parser.finish();
        assert_eq!(turn.content.as_deref().unwrap_or(""), "");
    }

    #[test]
    fn unrecognized_tool_envelope_is_ordinary_content() {
        let tools = [
            serde_json::json!({"function":{"name":"search_web","parameters":{
                "type":"object","properties":{"query":{"type":"string"}}
            }}}),
        ];
        for text in [
            "<tool_call>not a function call</tool_call>",
            r#"<tool_call>{"name":"search_web","arguments":{"query":"warp34"}}</tool_call>"#,
            r#"<tool_call>{"name":"search_web","arguments":{"query":"</tool_call>"}}</tool_call>"#,
        ] {
            assert!(
                Qwen3Parser::parse_tool_body(
                    text.trim_start_matches("<tool_call>")
                        .trim_end_matches("</tool_call>"),
                    0
                )
                .is_none()
            );
            for native in [false, true] {
                for split in 0..=text.len() {
                    let mut parser = if native {
                        Qwen3Parser::with_tools(false, &tools).unwrap()
                    } else {
                        Qwen3Parser::new(false)
                    };
                    let a = parser.push_str(&text[..split]);
                    let b = parser.push_str(&text[split..]);
                    let (c, turn) = parser.finish();
                    assert!(a.completed_tool_calls.is_empty());
                    assert!(b.completed_tool_calls.is_empty());
                    assert!(c.completed_tool_calls.is_empty());
                    assert!(turn.tool_calls.is_empty());
                    assert_eq!(turn.content.as_deref(), Some(text), "split {split}");
                    assert_eq!(
                        format!(
                            "{}{}{}",
                            a.content_delta.unwrap_or_default(),
                            b.content_delta.unwrap_or_default(),
                            c.content_delta.unwrap_or_default()
                        ),
                        text
                    );
                }
            }
        }
    }

    #[test]
    fn incremental_tool_call_deltas_stream_token_by_token_and_match_completed_turn() {
        use serde_json::json;
        let tools = vec![
            json!({"function":{"name":"write","parameters":{
                "type":"object",
                "properties":{
                    "path":{"type":"string","description":"target path"},
                    "content":{"type":"string","description":"file contents"}
                }
            }}}),
            json!({"function":{"name":"read","parameters":{
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "offset":{"type":"integer"},
                    "raw":{"type":"boolean"},
                    "meta":{"type":"object","properties":{"tag":{"type":"string"}}},
                    "note":{"type":"string"}
                }
            }}}),
            json!({"function":{"name":"ping","parameters":{
                "type":"object","properties":{}
            }}}),
        ];

        let stream = concat!(
            "<think>Planning scene update...</think>\n\n",
            "<tool_call>\n<function=write>\n",
            "<parameter=path>\nsrc/scene.js\n</parameter>\n",
            "<parameter=content>\n",
            "const msg = \"hello \\\\ 🦀 </function> </tool_call> \\n </param still string\";\n",
            "export default msg;\n",
            "</parameter>\n",
            "</function>\n</tool_call>\n",
            "<tool_call>\n<function=read>\n",
            "<parameter=path>src/scene.js</parameter>\n",
            "<parameter=offset>\n10\n</parameter>\n",
            "<parameter=raw>true</parameter>\n",
            "<parameter=meta>{\"tag\":\"v1\"}</parameter>\n",
            "<parameter=note>\n</parameter>\n",
            "</function>\n</tool_call>\n",
            "<tool_call>\n<function=ping>\n\n</function>\n</tool_call>"
        );

        for chunk_chars in [1usize, 2, 3, 5, 13, stream.len()] {
            let mut parser = Qwen3Parser::with_tools(true, &tools).unwrap();
            let mut deltas = Vec::new();
            let mut streamed_so_far = String::new();
            let mut saw_mid_content_stream = false;
            let chars: Vec<char> = stream.chars().collect();
            for piece in chars.chunks(chunk_chars) {
                let s: String = piece.iter().collect();
                let d = parser.push_str(&s);
                for tc in &d.tool_call_deltas {
                    if tc.index == 0
                        && let Some(a) = tc.arguments_delta.as_deref()
                    {
                        streamed_so_far.push_str(a);
                    }
                }
                if d.completed_tool_calls.is_empty()
                    && streamed_so_far.contains("export default msg;")
                {
                    saw_mid_content_stream = true;
                }
                deltas.extend(d.tool_call_deltas);
            }
            let (final_d, turn) = parser.finish();
            deltas.extend(final_d.tool_call_deltas);

            if chunk_chars <= 5 {
                assert!(
                    saw_mid_content_stream,
                    "expected content parameter to stream before </tool_call> for chunk_chars={chunk_chars}"
                );
            }
            assert_eq!(turn.tool_calls.len(), 3);
            for (idx, completed) in turn.tool_calls.iter().enumerate() {
                let for_idx: Vec<&StreamToolCallDelta> =
                    deltas.iter().filter(|d| d.index == idx).collect();
                assert!(!for_idx.is_empty(), "missing deltas for tool {idx}");
                assert_eq!(for_idx[0].id.as_deref(), Some(completed.id.as_str()));
                assert_eq!(
                    for_idx[0].name.as_deref(),
                    Some(completed.function.name.as_str())
                );
                for later in &for_idx[1..] {
                    assert_eq!(later.id, None);
                    assert_eq!(later.name, None);
                }
                let joined_args: String = for_idx
                    .iter()
                    .filter_map(|d| d.arguments_delta.as_deref())
                    .collect();
                assert_eq!(
                    joined_args, completed.function.arguments,
                    "chunk_chars={chunk_chars} tool={idx}"
                );
            }
        }
    }

    #[test]
    fn legacy_tool_call_deltas_stream_and_handle_unclosed_and_duplicate_params() {
        let legacy_stream = concat!(
            "<tool_call>\n<function=write>\n",
            "<parameter=path>\nsrc/scene.js\n</parameter>\n",
            "<parameter=count>\n42\n</parameter>\n",
            "</function>\n</tool_call>"
        );
        let mut legacy_parser = Qwen3Parser::new(false);
        let mut legacy_deltas = Vec::new();
        let mut legacy_so_far = String::new();
        let mut saw_mid_legacy_path = false;
        for ch in legacy_stream.chars() {
            let d = legacy_parser.push_str(&ch.to_string());
            for tc in &d.tool_call_deltas {
                if let Some(a) = tc.arguments_delta.as_deref() {
                    legacy_so_far.push_str(a);
                }
            }
            if d.completed_tool_calls.is_empty() && legacy_so_far.contains("src/scene.js") {
                saw_mid_legacy_path = true;
            }
            legacy_deltas.extend(d.tool_call_deltas);
        }
        let (final_d, legacy_turn) = legacy_parser.finish();
        legacy_deltas.extend(final_d.tool_call_deltas);
        assert!(saw_mid_legacy_path);
        assert_eq!(legacy_turn.tool_calls.len(), 1);
        let joined_legacy: String = legacy_deltas
            .iter()
            .filter_map(|d| d.arguments_delta.as_deref())
            .collect();
        assert_eq!(joined_legacy, legacy_turn.tool_calls[0].function.arguments);
        assert_eq!(joined_legacy, r#"{"path":"src/scene.js","count":42}"#);

        // Legacy unclosed </parameter> before next <parameter=> and duplicate parameter rejection.
        let unclosed_legacy = "<tool_call><function=write><parameter=count>42<parameter=path>a.js</function></tool_call>";
        let mut p_unclosed = Qwen3Parser::new(false);
        let mut unclosed_deltas = Vec::new();
        for ch in unclosed_legacy.chars() {
            unclosed_deltas.extend(p_unclosed.push_str(&ch.to_string()).tool_call_deltas);
        }
        let (df_unclosed, turn_unclosed) = p_unclosed.finish();
        unclosed_deltas.extend(df_unclosed.tool_call_deltas);
        assert_eq!(turn_unclosed.tool_calls.len(), 1);
        let joined_unclosed: String = unclosed_deltas
            .iter()
            .filter_map(|d| d.arguments_delta.as_deref())
            .collect();
        assert_eq!(
            joined_unclosed,
            turn_unclosed.tool_calls[0].function.arguments
        );
        assert_eq!(joined_unclosed, r#"{"count":42,"path":"a.js"}"#);

        let dup_legacy = "<tool_call><function=write><parameter=a>1</parameter><parameter=a>2</parameter></function></tool_call>";
        let mut p_dup = Qwen3Parser::new(false);
        let _ = p_dup.push_str(dup_legacy);
        assert!(p_dup.finish().1.tool_calls.is_empty());
    }

    #[test]
    fn native_tool_stream_abort_and_fallback_paths() {
        use serde_json::json;
        let tools = vec![json!({"function":{"name":"echo","parameters":{
            "type":"object",
            "properties":{
                "v":{"type":"string"},
                "count":{"type":"integer"}
            }
        }}})];

        // 1. Unknown function name or invalid buffered parameter value aborts before emitting
        // any tool header delta and reconciles streamed content_delta with turn.content.
        for bad in [
            "<tool_call>\n<function=unknown>\n<parameter=v>x</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=echo>\n<parameter=count>not_an_int</parameter>\n</function>\n</tool_call>",
        ] {
            let mut p = Qwen3Parser::with_tools(false, &tools).unwrap();
            let mut content_stream = String::new();
            let mut tool_deltas = Vec::new();
            for ch in bad.chars() {
                let d = p.push_str(&ch.to_string());
                if let Some(c) = d.content_delta {
                    content_stream.push_str(&c);
                }
                tool_deltas.extend(d.tool_call_deltas);
            }
            let (df, turn) = p.finish();
            if let Some(c) = df.content_delta {
                content_stream.push_str(&c);
            }
            tool_deltas.extend(df.tool_call_deltas);
            assert!(
                tool_deltas.is_empty(),
                "expected no orphan tool deltas for {bad:?}"
            );
            assert!(turn.tool_calls.is_empty());
            assert_eq!(content_stream, bad);
            assert_eq!(turn.content.as_deref(), Some(bad));
        }

        // 2. Post-header abort (duplicate parameter after a valid StreamString parameter):
        // tool call is rejected (turn.tool_calls empty), turn.content retains full fallback text,
        // and content_delta does not dump duplicate raw XML after header_emitted.
        let dup_native = "<tool_call>\n<function=echo>\n<parameter=v>first</parameter>\n<parameter=v>dup</parameter>\n</function>\n</tool_call>";
        let mut p_dup = Qwen3Parser::with_tools(false, &tools).unwrap();
        let mut dup_content_deltas = String::new();
        let mut dup_tool_deltas = Vec::new();
        for ch in dup_native.chars() {
            let d = p_dup.push_str(&ch.to_string());
            if let Some(c) = d.content_delta {
                dup_content_deltas.push_str(&c);
            }
            dup_tool_deltas.extend(d.tool_call_deltas);
        }
        let (df_dup, turn_dup) = p_dup.finish();
        if let Some(c) = df_dup.content_delta {
            dup_content_deltas.push_str(&c);
        }
        assert!(!dup_tool_deltas.is_empty());
        assert!(turn_dup.tool_calls.is_empty());
        assert!(dup_content_deltas.is_empty());
        assert_eq!(turn_dup.content.as_deref(), Some(dup_native));
    }

    #[test]
    fn incremental_tool_call_deltas_newline_and_delimiter_holdback_matrix() {
        use serde_json::json;
        let tools = vec![json!({"function":{"name":"echo","parameters":{
            "type":"object",
            "properties":{
                "v":{"type":"string"},
                "c":{"const":"before</parameter>after"}
            }
        }}})];
        for (raw_body, expected_json) in [
            ("<parameter=v></parameter>", r#"{"v":""}"#),
            ("<parameter=v>\n</parameter>", r#"{"v":""}"#),
            ("<parameter=v>\n\n</parameter>", r#"{"v":""}"#),
            ("<parameter=v>\n\n\n</parameter>", r#"{"v":"\n"}"#),
            (
                "<parameter=v>\n\nParis\n\n</parameter>",
                r#"{"v":"\nParis\n"}"#,
            ),
            (
                "<parameter=v>hello\n</param\nworld\n</parameter>",
                r#"{"v":"hello\n</param\nworld"}"#,
            ),
            (
                "<parameter=v>before</tool_call></function>after</parameter>",
                r#"{"v":"before</tool_call></function>after"}"#,
            ),
            (
                "<parameter=c>before</parameter>after</parameter>",
                r#"{"c":"before</parameter>after"}"#,
            ),
        ] {
            let text =
                format!("<tool_call>\n<function=echo>\n{raw_body}\n</function>\n</tool_call>");
            for s1 in 0..=text.len() {
                for s2 in s1..=text.len() {
                    let mut parser = Qwen3Parser::with_tools(false, &tools).unwrap();
                    let d1 = parser.push_str(&text[..s1]);
                    let d2 = parser.push_str(&text[s1..s2]);
                    let d3 = parser.push_str(&text[s2..]);
                    let (df, turn) = parser.finish();
                    assert_eq!(turn.tool_calls.len(), 1, "{text:?} ({s1}, {s2})");
                    assert_eq!(turn.tool_calls[0].function.arguments, expected_json);
                    let all_deltas: Vec<StreamToolCallDelta> = d1
                        .tool_call_deltas
                        .into_iter()
                        .chain(d2.tool_call_deltas)
                        .chain(d3.tool_call_deltas)
                        .chain(df.tool_call_deltas)
                        .collect();
                    assert_eq!(all_deltas[0].id.as_deref(), Some("call_warp34_0"));
                    assert_eq!(all_deltas[0].name.as_deref(), Some("echo"));
                    let joined: String = all_deltas
                        .iter()
                        .filter_map(|d| d.arguments_delta.as_deref())
                        .collect();
                    assert_eq!(joined, expected_json, "{text:?} ({s1}, {s2})");
                }
            }
        }
    }

    #[test]
    fn test_sampler_repetition_penalties() {
        let params = SamplingParams {
            temperature: 1.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };

        let sampler = SamplerV1::new(params, &[10], None);
        // Candidate 10 is in prompt history, candidate 20 is not.
        // With positive logits, candidate 10's logit should be halved by repetition_penalty=2.0.
        let logits = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 4.0,
        ];
        let probs = sampler.processed_distribution(&logits).unwrap();

        // Token 10 had logit 4.0 / 2.0 = 2.0; token 20 had logit 4.0.
        // Therefore prob[20] should be strictly greater than prob[10].
        assert!(probs[20] > probs[10]);
    }

    #[test]
    fn test_thinking_budget_forcing_and_speculative_rollback() {
        // Token IDs: 1 = <think>, 2 = </think>, 3 = \n, 5 = normal thought/answer token, 9 = <tool_call>
        let cfg = ThinkingBudgetConfig {
            budget: 2,
            start_token_ids: vec![1],
            end_token_ids: vec![2],
            forced_token_ids: vec![2, 3],
            extra_end_token_ids: vec![vec![9]],
        };
        let params = SamplingParams {
            temperature: 0.0,
            thinking_budget: Some(cfg.clone()),
            ..Default::default()
        };
        // Prompt ends with `<think>` -> starts counting immediately
        let mut sampler = SamplerV1::new(params.clone(), &[10, 1], None);
        assert!(!sampler.is_fast_greedy_stateless(0));
        assert_eq!(sampler.forced_thinking_token(), None);

        // Logits favor token 5 strongly
        let mut logits = vec![0.0f32; 16];
        logits[5] = 10.0;

        // 2 thinking tokens allowed before forcing
        assert_eq!(sampler.sample_from_logits(&logits), 5);
        assert_eq!(sampler.forced_thinking_token(), None);
        assert_eq!(sampler.sample_from_logits(&logits), 5);
        assert_eq!(sampler.forced_thinking_token(), Some(2));

        // GPU row bitmask forces ONLY token 2
        let mut bitmask = Vec::new();
        let mut sparse = Vec::new();
        let row = sampler
            .append_gpu_row(16, &[], 0.5, 0.5, -1, false, &mut bitmask, &mut sparse)
            .unwrap();
        assert_eq!(row.has_bitmask, 1);
        assert_eq!(bitmask, vec![1u32 << 2]);

        // Speculative rollback of 1 token restores counting phase (unforced)
        sampler.rollback_tokens(1);
        assert_eq!(sampler.forced_thinking_token(), None);

        // Re-commit token 5 -> forces token 2, then token 3, then returns to normal token 5
        assert_eq!(sampler.sample_from_logits(&logits), 5);
        assert_eq!(sampler.sample_from_logits(&logits), 2);
        assert_eq!(sampler.sample_from_logits(&logits), 3);
        assert_eq!(sampler.forced_thinking_token(), None);
        assert_eq!(sampler.sample_from_logits(&logits), 5);

        // Natural </think> before budget exhaustion deactivates forcing permanently
        // (literal <think> in the subsequent answer must NOT re-arm Done).
        let mut natural = SamplerV1::new(params.clone(), &[10, 1], None);
        natural.commit_token(5);
        natural.commit_token(2); // natural </think>
        assert_eq!(natural.forced_thinking_token(), None);
        natural.commit_token(1); // literal <think> in answer
        natural.commit_token(5);
        natural.commit_token(5);
        assert_eq!(natural.forced_thinking_token(), None);

        // Implicit <tool_call> (extra_end_token_ids) terminates Counting cleanly.
        let mut implicit_tool = SamplerV1::new(params.clone(), &[10, 1], None);
        implicit_tool.commit_token(5);
        implicit_tool.commit_token(9); // <tool_call>
        implicit_tool.commit_token(5);
        implicit_tool.commit_token(5);
        assert_eq!(implicit_tool.forced_thinking_token(), None);

        // Literal <think> far earlier in user prompt does not falsely arm when template
        // ends without <think> in the trailing assistant header.
        let mut far_prompt = vec![10; 24];
        far_prompt[2] = 1; // literal <think> in user text
        let mut user_only = SamplerV1::new(params, &far_prompt, None);
        user_only.commit_token(5);
        user_only.commit_token(5);
        assert_eq!(user_only.forced_thinking_token(), None);

        // budget == 0 forces immediately when prompt ends with <think>
        let immediate = SamplerV1::new(
            SamplingParams {
                thinking_budget: Some(ThinkingBudgetConfig { budget: 0, ..cfg }),
                ..Default::default()
            },
            &[10, 1],
            None,
        );
        assert_eq!(immediate.forced_thinking_token(), Some(2));
    }
}

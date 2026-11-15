use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionMediaTokens {
    pub vision_start_id: i32,
    pub image_pad_id: i32,
    pub vision_end_id: i32,
}

#[derive(Clone, Debug, Default)]
pub struct DecisionVisionContext {
    pub vision_config: Option<crate::vision::VisionConfig>,
    pub has_vision_weights: bool,
    pub allow_remote_images: bool,
}

/// Decodes and preprocesses optional `images` on a `/v1/systemone` or `/v1/decisions` request.
pub(super) fn validate_decision_media_fields(
    obj: &Map<String, Value>,
) -> Result<Option<&[Value]>, DecisionError> {
    if let Some(vids) = obj.get("videos")
        && !vids.is_null()
        && !vids.as_array().is_some_and(Vec::is_empty)
    {
        return Err(DecisionError::bad_request("video inputs are not supported"));
    }
    let Some(imgs_val) = obj.get("images") else {
        return Ok(None);
    };
    if imgs_val.is_null() {
        return Ok(None);
    }
    let Some(arr) = imgs_val.as_array() else {
        return Err(DecisionError::bad_request("images must be an array"));
    };
    if arr.len() > crate::vision::MAX_IMAGES_PER_REQUEST {
        return Err(DecisionError::bad_request(format!(
            "at most {} images are allowed per request",
            crate::vision::MAX_IMAGES_PER_REQUEST
        )));
    }
    if arr.is_empty() {
        return Ok(None);
    }
    Ok(Some(arr.as_slice()))
}

pub fn parse_decision_images(
    body: &Value,
    profile: &DecisionModelProfile,
    ctx: &DecisionVisionContext,
) -> Result<Vec<crate::vision::PreprocessedImage>, DecisionError> {
    let Some(obj) = body.as_object() else {
        return Err(DecisionError::bad_request("request body must be an object"));
    };
    let Some(arr) = validate_decision_media_fields(obj)? else {
        return Ok(Vec::new());
    };
    let (Some(vcfg), true) = (ctx.vision_config.as_ref(), ctx.has_vision_weights) else {
        return Err(DecisionError::bad_request(
            "loaded model does not have vision weights; image inputs are not supported",
        ));
    };
    let (min_pixels, max_pixels) = profile.image_pixel_bounds();
    let deadline = std::time::Instant::now() + crate::remote_images::REMOTE_IMAGE_TIMEOUT;
    let mut out = Vec::with_capacity(arr.len());
    for (idx, item) in arr.iter().enumerate() {
        let url: String = if let Some(s) = item.as_str() {
            s.to_owned()
        } else if let Some(m) = item.as_object() {
            if let Some(u) = m.get("url").and_then(Value::as_str) {
                u.to_owned()
            } else if let Some(u) = m
                .get("image_url")
                .and_then(|v| v.as_str().or_else(|| v.get("url").and_then(Value::as_str)))
            {
                u.to_owned()
            } else if let Some(data) = m.get("data").and_then(Value::as_str) {
                let mt = m
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                format!("data:{mt};base64,{data}")
            } else {
                return Err(DecisionError::bad_request(format!(
                    "images[{idx}] missing 'url' or 'data'"
                )));
            }
        } else {
            return Err(DecisionError::bad_request(format!(
                "images[{idx}] must be an image URL string or object"
            )));
        };
        let (rgb, h, w) =
            crate::vision::parse_image_url_with_deadline(&url, ctx.allow_remote_images, deadline)
                .map_err(|e| DecisionError::bad_request(format!("images[{idx}]: {e}")))?;
        crate::vision::validate_image_aspect_ratio(h, w)
            .map_err(|e| DecisionError::bad_request(format!("images[{idx}]: {e}")))?;
        out.push(crate::vision::preprocess_rgb_image(
            &rgb,
            h,
            w,
            vcfg.patch_size,
            vcfg.temporal_patch_size,
            vcfg.spatial_merge_size,
            min_pixels,
            max_pixels,
        ));
    }
    Ok(out)
}

pub(super) fn build_media_prefix_tokens(
    tokenizer: &Tokenizer,
    media: Option<(DecisionMediaTokens, &[usize])>,
) -> Result<Vec<i32>, DecisionError> {
    let Some((mtoks, counts)) = media else {
        return Ok(Vec::new());
    };
    if counts.is_empty() {
        return Ok(Vec::new());
    }
    let mut prefix = Vec::new();
    for &n in counts {
        prefix.push(mtoks.vision_start_id);
        prefix.resize(prefix.len() + n, mtoks.image_pad_id);
        prefix.push(mtoks.vision_end_id);
    }
    let nl = tokenizer
        .encode("\n", false)
        .map_err(|e| DecisionError::internal(format!("tokenize newline: {e}")))?;
    prefix.extend(nl.get_ids().iter().map(|&x| x as i32));
    Ok(prefix)
}

#[derive(Clone, Debug)]
pub struct CodeReadoutPreparedQuery {
    pub token_ids: Vec<i32>,
    pub positions: Vec<usize>,
    pub prefix_cap: Option<usize>,
    pub prefix_bidir_len: Option<usize>,
}

/// Prepared query plans ready for execution on `EngineWorker`.
#[derive(Clone, Debug)]
pub enum PreparedDecisionBatch {
    CodeReadout {
        head: Arc<CodeReadoutHead>,
        temperature: f64,
        conf_kind: ConfidenceKind,
        fields: Vec<DecisionField>,
        /// Either 1 query (`state_once`) with `fields.len()` positions, or `fields.len()` queries
        /// with 1 position each (`[prompt.len() - 1]`).
        forward_queries: Vec<CodeReadoutPreparedQuery>,
        /// Populated when `order_passes == 2` (same shape as `forward_queries`, with reversed options).
        reverse_queries: Option<Vec<CodeReadoutPreparedQuery>>,
    },
    CandidateLogits {
        family: &'static str,
        temperature: f64,
        conf_kind: ConfidenceKind,
        fields: Vec<DecisionField>,
        prompts: Vec<Vec<i32>>,
        reverse_prompts: Option<Vec<Vec<i32>>>,
        candidates: Vec<Vec<i32>>,
    },
}

pub fn prepare_decision_batch(
    profile: &DecisionModelProfile,
    tokenizer: &Tokenizer,
    max_prompt_tokens: usize,
    body: &Value,
) -> Result<(Option<String>, PreparedDecisionBatch), DecisionError> {
    if let Some(obj) = body.as_object()
        && validate_decision_media_fields(obj)?.is_some()
    {
        return Err(DecisionError::bad_request(
            "image inputs require prepare_decision_batch_with_images on a vision-capable model",
        ));
    }
    prepare_decision_batch_with_images(profile, tokenizer, max_prompt_tokens, body, None, &[])
}

fn parse_order_passes(body: &Value) -> Result<usize, DecisionError> {
    let Some(obj) = body.as_object() else {
        return Ok(1);
    };
    let Some(op) = obj.get("order_passes") else {
        return Ok(1);
    };
    match op.as_u64() {
        Some(1) => Ok(1),
        Some(2) => Ok(2),
        _ => Err(DecisionError::bad_request("order_passes must be 1 or 2")),
    }
}

pub fn prepare_decision_batch_with_images(
    profile: &DecisionModelProfile,
    tokenizer: &Tokenizer,
    max_prompt_tokens: usize,
    body: &Value,
    vision_config: Option<&crate::vision::VisionConfig>,
    images: &[crate::vision::PreprocessedImage],
) -> Result<(Option<String>, PreparedDecisionBatch), DecisionError> {
    let order_passes = parse_order_passes(body)?;
    let merged_counts: Vec<usize>;
    let media = if !images.is_empty() {
        let vcfg = vision_config.ok_or_else(|| {
            DecisionError::bad_request(
                "loaded model does not have vision_config; image inputs are not supported",
            )
        })?;
        merged_counts = images
            .iter()
            .map(|img| img.num_merged_tokens(vcfg.spatial_merge_size))
            .collect();
        Some((
            DecisionMediaTokens {
                vision_start_id: vcfg.vision_start_token_id,
                image_pad_id: vcfg.image_token_id,
                vision_end_id: vcfg.vision_end_token_id,
            },
            merged_counts.as_slice(),
        ))
    } else {
        None
    };

    match profile {
        DecisionModelProfile::CodeReadout { head, config } => {
            let max_choices = config.codes.len().min(MAX_SYSTEMONE_ANSWERS);
            let noul_true_first = config.style == CodeReadoutStyle::Glide;
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices,
                    too_wide_reason: "CodeReadout exceeds available option codes",
                    noul_true_first,
                },
            )?;
            let effective_max = config.max_seq_len.min(max_prompt_tokens);
            let temperature = config.effective_temperature(req.fields.len());
            let conf_kind = match config.style {
                CodeReadoutStyle::Glide => ConfidenceKind::Top1Top2Margin,
                CodeReadoutStyle::D3 => ConfidenceKind::NormalizedEntropy,
            };
            let forward_queries =
                build_code_readout_queries(config, tokenizer, &req, effective_max, media, false)?;
            let reverse_queries = if order_passes == 2 {
                Some(build_code_readout_queries(
                    config,
                    tokenizer,
                    &req,
                    effective_max,
                    media,
                    true,
                )?)
            } else {
                None
            };
            Ok((
                req.model,
                PreparedDecisionBatch::CodeReadout {
                    head: Arc::clone(head),
                    temperature,
                    conf_kind,
                    fields: req.fields,
                    forward_queries,
                    reverse_queries,
                },
            ))
        }
        DecisionModelProfile::CausalLm { temperature } => {
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices: TEV1_MAX_OPTIONS,
                    too_wide_reason: "CausalLm supports 2-24 options labelled A-X",
                    noul_true_first: false,
                },
            )?;
            let fwd_strs = build_tev1_prompts(&req)?;
            let (prompts, candidates) = tokenize_candidate_prompts_with_media(
                tokenizer,
                &req.fields,
                &fwd_strs,
                ANSWER_BOUNDARY,
                b"ABCDEFGHIJKLMNOPQRSTUVWX",
                max_prompt_tokens,
                "causal_lm",
                media,
            )?;
            let reverse_prompts = if order_passes == 2 {
                let mut rev_strs = Vec::with_capacity(req.fields.len());
                for f in &req.fields {
                    let rev_pairs: Vec<(String, String)> = f
                        .keys
                        .iter()
                        .cloned()
                        .zip(f.value_descriptions.iter().cloned())
                        .rev()
                        .collect();
                    rev_strs.push(build_tev1_prompt_for_options(
                        &req.state,
                        &f.description,
                        &rev_pairs,
                    )?);
                }
                let (rev_toks, _) = tokenize_candidate_prompts_with_media(
                    tokenizer,
                    &req.fields,
                    &rev_strs,
                    ANSWER_BOUNDARY,
                    b"ABCDEFGHIJKLMNOPQRSTUVWX",
                    max_prompt_tokens,
                    "causal_lm",
                    media,
                )?;
                Some(rev_toks)
            } else {
                None
            };
            Ok((
                req.model,
                PreparedDecisionBatch::CandidateLogits {
                    family: "causal_lm",
                    temperature: *temperature,
                    conf_kind: ConfidenceKind::NormalizedEntropy,
                    fields: req.fields,
                    prompts,
                    reverse_prompts,
                    candidates,
                },
            ))
        }
    }
}

fn build_code_readout_queries(
    config: &CodeReadoutConfig,
    tokenizer: &Tokenizer,
    req: &CompiledDecisionRequest,
    max_prompt_tokens: usize,
    media: Option<(DecisionMediaTokens, &[usize])>,
    reverse_options: bool,
) -> Result<Vec<CodeReadoutPreparedQuery>, DecisionError> {
    let media_prefix = build_media_prefix_tokens(tokenizer, media)?;

    if config.style == CodeReadoutStyle::D3 && config.state_once {
        let prefix_str = build_d3_state_prefix(&req.context, &req.fields, config.header_v1);
        let prefix_enc = tokenizer
            .encode(prefix_str.as_str(), false)
            .map_err(|e| DecisionError::internal(format!("d3 tokenize state: {e}")))?;
        let mut token_ids = Vec::with_capacity(media_prefix.len() + prefix_enc.len() + 256);
        token_ids.extend_from_slice(&media_prefix);
        token_ids.extend(prefix_enc.get_ids().iter().map(|&x| x as i32));
        let prefix_len = token_ids.len();
        let mut positions = Vec::with_capacity(req.fields.len());
        for f in &req.fields {
            let mut pairs: Vec<(String, String)> = f
                .keys
                .iter()
                .cloned()
                .zip(f.value_descriptions.iter().cloned())
                .collect();
            if reverse_options {
                pairs.reverse();
            }
            let suffix =
                build_d3_question_suffix(&f.field_type, &f.description, &pairs, &config.codes);
            let q_enc = tokenizer
                .encode(suffix.as_str(), false)
                .map_err(|e| DecisionError::internal(format!("d3 tokenize question: {e}")))?;
            if q_enc.is_empty() {
                return Err(DecisionError::internal("d3 question encoded to 0 tokens"));
            }
            token_ids.extend(q_enc.get_ids().iter().map(|&x| x as i32));
            positions.push(token_ids.len() - 1);
        }
        if token_ids.len() > max_prompt_tokens {
            return Err(DecisionError::bad_request(format!(
                "Longest prompt has {} tokens; limit is {max_prompt_tokens}. Nothing was truncated.",
                token_ids.len()
            )));
        }
        let full_noncausal = config.noncausal_full_attention
            || config.midloop.iter().any(|&(_, _, passes)| passes > 1);
        let prefix_bidir_len = if full_noncausal {
            Some(token_ids.len())
        } else {
            config.prefix_bidir.then_some(prefix_len)
        };
        let prefix_cap = if full_noncausal {
            Some(0)
        } else {
            prefix_bidir_len
                .is_none()
                .then(|| positions.first().copied().unwrap_or(prefix_len))
        };
        return Ok(vec![CodeReadoutPreparedQuery {
            token_ids,
            positions,
            prefix_cap,
            prefix_bidir_len,
        }]);
    }

    let mut queries = Vec::with_capacity(req.fields.len());
    let mut longest = 0usize;
    let special_token_strs: Vec<String> = if config.style == CodeReadoutStyle::Glide {
        tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter_map(|t| {
                (t.special && (t.content.contains('<') || t.content.contains('>')))
                    .then_some(t.content)
            })
            .collect()
    } else {
        Vec::new()
    };
    let d3_prefix_ids = if config.style == CodeReadoutStyle::D3 {
        let prefix_str = build_d3_state_prefix(&req.context, &req.fields, config.header_v1);
        let prefix_enc = tokenizer
            .encode(prefix_str.as_str(), false)
            .map_err(|e| DecisionError::internal(format!("d3 tokenize state: {e}")))?;
        let mut ids = Vec::with_capacity(media_prefix.len() + prefix_enc.len());
        ids.extend_from_slice(&media_prefix);
        ids.extend(prefix_enc.get_ids().iter().map(|&x| x as i32));
        Some(ids)
    } else {
        None
    };

    for f in &req.fields {
        let mut pairs: Vec<(String, String)> = f
            .keys
            .iter()
            .cloned()
            .zip(f.value_descriptions.iter().cloned())
            .collect();
        if reverse_options {
            pairs.reverse();
        }
        let (token_ids, prefix_len) = match config.style {
            CodeReadoutStyle::Glide => {
                let raw_prompt = build_glide_prompt(
                    &config.system_prompt,
                    &req.context,
                    &f.name,
                    &f.field_type,
                    &f.description,
                    &pairs,
                    &config.codes,
                );
                let safe_p = escape_added_special_tokens_in_chatml_user_turn(
                    &raw_prompt,
                    &special_token_strs,
                );
                let ids =
                    tokenize_single_prompt_with_media(tokenizer, &safe_p, &media_prefix, "glide")?;
                let len = ids.len();
                (ids, len)
            }
            CodeReadoutStyle::D3 => {
                let prefix = d3_prefix_ids.as_ref().expect("d3_prefix_ids");
                let suffix =
                    build_d3_question_suffix(&f.field_type, &f.description, &pairs, &config.codes);
                let q_enc = tokenizer
                    .encode(suffix.as_str(), false)
                    .map_err(|e| DecisionError::internal(format!("d3 tokenize question: {e}")))?;
                let mut ids = Vec::with_capacity(prefix.len() + q_enc.len());
                ids.extend_from_slice(prefix);
                ids.extend(q_enc.get_ids().iter().map(|&x| x as i32));
                (ids, prefix.len())
            }
        };
        if token_ids.is_empty() {
            return Err(DecisionError::internal(
                "code_readout prompt encoded to 0 tokens",
            ));
        }
        longest = longest.max(token_ids.len());
        let last_pos = token_ids.len() - 1;
        let full_noncausal = config.noncausal_full_attention
            || config.midloop.iter().any(|&(_, _, passes)| passes > 1);
        let prefix_bidir_len = if full_noncausal {
            Some(token_ids.len())
        } else {
            config.prefix_bidir.then_some(prefix_len)
        };
        let prefix_cap = if full_noncausal {
            Some(0)
        } else {
            prefix_bidir_len.is_none().then_some(last_pos)
        };
        queries.push(CodeReadoutPreparedQuery {
            token_ids,
            positions: vec![last_pos],
            prefix_cap,
            prefix_bidir_len,
        });
    }
    if longest > max_prompt_tokens {
        return Err(DecisionError::bad_request(format!(
            "Longest prompt has {longest} tokens; limit is {max_prompt_tokens}. Nothing was truncated."
        )));
    }
    Ok(queries)
}

fn tokenize_single_prompt_with_media(
    tokenizer: &Tokenizer,
    safe_p: &str,
    media_prefix: &[i32],
    model_label: &str,
) -> Result<Vec<i32>, DecisionError> {
    if !media_prefix.is_empty() {
        const USER_OPEN: &str = "<|im_end|>\n<|im_start|>user\n";
        if let Some(u_pos) = safe_p.find(USER_OPEN) {
            let split_at = u_pos + USER_OPEN.len();
            let head_enc = tokenizer
                .encode(&safe_p[..split_at], false)
                .map_err(|e| DecisionError::internal(format!("{model_label}: tokenize: {e}")))?;
            let body_enc = tokenizer
                .encode(&safe_p[split_at..], false)
                .map_err(|e| DecisionError::internal(format!("{model_label}: tokenize: {e}")))?;
            let mut combined =
                Vec::with_capacity(head_enc.len() + media_prefix.len() + body_enc.len());
            combined.extend(head_enc.get_ids().iter().map(|&x| x as i32));
            combined.extend_from_slice(media_prefix);
            combined.extend(body_enc.get_ids().iter().map(|&x| x as i32));
            Ok(combined)
        } else if safe_p.contains("<|im_start|>") {
            Err(DecisionError::internal(format!(
                "{model_label}: ChatML prompt missing user turn boundary"
            )))
        } else {
            let enc = tokenizer
                .encode(safe_p, false)
                .map_err(|e| DecisionError::internal(format!("{model_label}: tokenize: {e}")))?;
            let mut combined = Vec::with_capacity(media_prefix.len() + enc.len());
            combined.extend_from_slice(media_prefix);
            combined.extend(enc.get_ids().iter().map(|&x| x as i32));
            Ok(combined)
        }
    } else {
        let enc = tokenizer
            .encode(safe_p, false)
            .map_err(|e| DecisionError::internal(format!("{model_label}: tokenize: {e}")))?;
        Ok(enc.get_ids().iter().map(|&x| x as i32).collect())
    }
}

type TokenizedCandidates = (Vec<Vec<i32>>, Vec<Vec<i32>>);

#[allow(clippy::too_many_arguments)]
pub(super) fn tokenize_candidate_prompts_with_media(
    tokenizer: &Tokenizer,
    fields: &[DecisionField],
    prompt_strs: &[String],
    boundary: &str,
    alphabet: &[u8],
    max_prompt_tokens: usize,
    model_label: &str,
    media: Option<(DecisionMediaTokens, &[usize])>,
) -> Result<TokenizedCandidates, DecisionError> {
    let media_prefix = build_media_prefix_tokens(tokenizer, media)?;
    let special_token_strs: Vec<String> = tokenizer
        .get_added_tokens_decoder()
        .into_values()
        .filter_map(|t| {
            (t.special && (t.content.contains('<') || t.content.contains('>'))).then_some(t.content)
        })
        .collect();
    let mut prompts = Vec::with_capacity(prompt_strs.len());
    let mut longest = 0usize;
    for p in prompt_strs {
        let safe_p = escape_added_special_tokens_in_chatml_user_turn(p, &special_token_strs);
        let ids =
            tokenize_single_prompt_with_media(tokenizer, &safe_p, &media_prefix, model_label)?;
        longest = longest.max(ids.len());
        prompts.push(ids);
    }
    if longest > max_prompt_tokens {
        return Err(DecisionError::bad_request(format!(
            "Longest prompt has {longest} tokens; limit is {max_prompt_tokens}. Nothing was truncated."
        )));
    }
    let widest = fields.iter().map(|f| f.keys.len()).max().unwrap_or(0);
    let codes = candidate_token_ids(tokenizer, boundary, &alphabet[..widest], model_label)?;
    let candidates = fields
        .iter()
        .map(|f| codes[..f.keys.len()].to_vec())
        .collect();
    Ok((prompts, candidates))
}

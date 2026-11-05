use super::*;

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
    media: Option<(ClefMediaTokens, &[usize])>,
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

/// Prepared query plans ready for execution on `EngineWorker`.
#[derive(Clone, Debug)]
pub enum PreparedDecisionBatch {
    Clef {
        fields: Vec<DecisionField>,
        record: ClefEncodedRecord,
    },
    Kev {
        head: KevPointerHead,
        temperature: f64,
        fields: Vec<DecisionField>,
        encoded: Vec<KevEncodedQuestion>,
    },
    CandidateLogits {
        family: &'static str,
        temperature: f64,
        conf_kind: ConfidenceKind,
        fields: Vec<DecisionField>,
        prompts: Vec<Vec<i32>>,
        candidates: Vec<Vec<i32>>,
    },
    Xor {
        temperature: f64,
        lev_noul_9_scale: bool,
        fields: Vec<DecisionField>,
        /// Pairs of `(forward_prompt_ids, reverse_prompt_ids, candidate_ids)`.
        variants: Vec<(Vec<i32>, Vec<i32>, Vec<i32>)>,
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

pub fn prepare_decision_batch_with_images(
    profile: &DecisionModelProfile,
    tokenizer: &Tokenizer,
    max_prompt_tokens: usize,
    body: &Value,
    vision_config: Option<&crate::vision::VisionConfig>,
    images: &[crate::vision::PreprocessedImage],
) -> Result<(Option<String>, PreparedDecisionBatch), DecisionError> {
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
            ClefMediaTokens {
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
        DecisionModelProfile::Clef { max_length, .. } => {
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices: MAX_SYSTEMONE_ANSWERS,
                    too_wide_reason: "clef supports at most 64 options per question",
                    noul_true_first: true,
                },
            )?;
            let effective_max = (*max_length).min(max_prompt_tokens);
            let record = clef_encode_record(tokenizer, body, effective_max, media)?;
            Ok((
                req.model,
                PreparedDecisionBatch::Clef {
                    fields: req.fields,
                    record,
                },
            ))
        }
        DecisionModelProfile::Kev { head, temperature } => {
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices: MAX_SYSTEMONE_ANSWERS,
                    too_wide_reason: "kev supports at most 64 options per question",
                    noul_true_first: false,
                },
            )?;
            let special =
                KevSpecialTokens::from_tokenizer(tokenizer).map_err(DecisionError::internal)?;
            let enc_text = |s: &str| -> Result<Vec<i32>, DecisionError> {
                let sanitized = kev_sanitize_special_tokens(s);
                tokenizer
                    .encode(sanitized.as_str(), false)
                    .map(|e| e.get_ids().iter().map(|&x| x as i32).collect())
                    .map_err(|e| DecisionError::internal(format!("kev tokenize: {e}")))
            };
            let mut state_tokens = build_media_prefix_tokens(tokenizer, media)?;
            state_tokens.extend(enc_text(&req.context)?);
            let mut encoded = Vec::with_capacity(req.fields.len());
            let mut longest = 0usize;
            for f in &req.fields {
                let instr_tokens = enc_text(&f.description)?;
                let mut opt_tokens = Vec::with_capacity(f.value_descriptions.len());
                for desc in &f.value_descriptions {
                    opt_tokens.push(enc_text(desc)?);
                }
                let q_enc = kev_encode_question(special, &state_tokens, &instr_tokens, &opt_tokens);
                longest = longest.max(q_enc.token_ids.len());
                encoded.push(q_enc);
            }
            if longest > max_prompt_tokens {
                return Err(DecisionError::bad_request(format!(
                    "Longest prompt has {longest} tokens; limit is {max_prompt_tokens}. Nothing was truncated."
                )));
            }
            Ok((
                req.model,
                PreparedDecisionBatch::Kev {
                    head: head.clone(),
                    temperature: *temperature,
                    fields: req.fields,
                    encoded,
                },
            ))
        }
        DecisionModelProfile::Nimble {
            temperature,
            max_length,
        } => {
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices: NIMBLE_MAX_CHOICES,
                    too_wide_reason: "Nimble's one-letter prompt contract (parallel_schema.py) serves at most 26",
                    noul_true_first: false,
                },
            )?;
            let prompt_strs = build_nimble_prompts(&req)?;
            let limit = (*max_length).saturating_sub(1).min(max_prompt_tokens);
            let (prompts, candidates) = tokenize_candidate_prompts_with_media(
                tokenizer,
                &req.fields,
                &prompt_strs,
                ANSWER_BOUNDARY,
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                limit,
                "nimble",
                media,
            )?;
            Ok((
                req.model,
                PreparedDecisionBatch::CandidateLogits {
                    family: "nimble",
                    temperature: *temperature,
                    conf_kind: ConfidenceKind::NormalizedEntropy,
                    fields: req.fields,
                    prompts,
                    candidates,
                },
            ))
        }
        DecisionModelProfile::Tev1 { temperature } => {
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices: TEV1_MAX_OPTIONS,
                    too_wide_reason: "Tev1 is trained on 2-24 options labelled A-X (togethercomputer/tev1 examples/decide.py)",
                    noul_true_first: false,
                },
            )?;
            let prompt_strs = build_tev1_prompts(&req)?;
            let (prompts, candidates) = tokenize_candidate_prompts_with_media(
                tokenizer,
                &req.fields,
                &prompt_strs,
                ANSWER_BOUNDARY,
                b"ABCDEFGHIJKLMNOPQRSTUVWX",
                max_prompt_tokens,
                "tev1",
                media,
            )?;
            Ok((
                req.model,
                PreparedDecisionBatch::CandidateLogits {
                    family: "tev1",
                    temperature: *temperature,
                    conf_kind: ConfidenceKind::NormalizedEntropy,
                    fields: req.fields,
                    prompts,
                    candidates,
                },
            ))
        }
        DecisionModelProfile::OpenJev { temperature } => {
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices: OPENJEV_MAX_OPTIONS,
                    too_wide_reason: "OpenJev supports at most 52 single-letter options (A-Z, a-z)",
                    noul_true_first: true,
                },
            )?;
            let prompt_strs = build_openjev_prompts(&req)?;
            let (prompts, candidates) = tokenize_candidate_prompts_with_media(
                tokenizer,
                &req.fields,
                &prompt_strs,
                ANSWER_BOUNDARY,
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                max_prompt_tokens,
                "openjev",
                media,
            )?;
            Ok((
                req.model,
                PreparedDecisionBatch::CandidateLogits {
                    family: "openjev",
                    temperature: *temperature,
                    conf_kind: ConfidenceKind::MarginAndModeDistance,
                    fields: req.fields,
                    prompts,
                    candidates,
                },
            ))
        }
        DecisionModelProfile::Xor {
            temperature,
            lev_noul_9_scale,
        } => {
            let req = compile_systemone_request(
                body,
                CompileLimits {
                    max_choices: XOR_MAX_OPTIONS,
                    too_wide_reason: "Xor/Lev supports at most 26 single-letter options (A-Z)",
                    noul_true_first: false,
                },
            )?;
            let media_prefix = build_media_prefix_tokens(tokenizer, media)?;
            let mut variants = Vec::with_capacity(req.fields.len());
            let mut longest = 0usize;
            for f in &req.fields {
                let opts: Vec<String> = if *lev_noul_9_scale && f.field_type == "noul" {
                    LEV_NOUL_9_SCALE.iter().map(|s| (*s).to_owned()).collect()
                } else {
                    f.value_descriptions.clone()
                };
                let rev_opts: Vec<String> = opts.iter().rev().cloned().collect();
                let fwd_str = build_xor_prompt(&req.context, &f.field_type, &f.description, &opts);
                let rev_str =
                    build_xor_prompt(&req.context, &f.field_type, &f.description, &rev_opts);
                let mut fwd_ids = media_prefix.clone();
                fwd_ids.extend(
                    tokenizer
                        .encode(fwd_str.as_str(), false)
                        .map_err(|e| DecisionError::internal(format!("xor tokenize: {e}")))?
                        .get_ids()
                        .iter()
                        .map(|&x| x as i32),
                );
                let mut rev_ids = media_prefix.clone();
                rev_ids.extend(
                    tokenizer
                        .encode(rev_str.as_str(), false)
                        .map_err(|e| DecisionError::internal(format!("xor tokenize: {e}")))?
                        .get_ids()
                        .iter()
                        .map(|&x| x as i32),
                );
                longest = longest.max(fwd_ids.len()).max(rev_ids.len());
                let cands = candidate_token_ids(
                    tokenizer,
                    "",
                    &b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"[..opts.len()],
                    "xor",
                )?;
                variants.push((fwd_ids, rev_ids, cands));
            }
            if longest > max_prompt_tokens {
                return Err(DecisionError::bad_request(format!(
                    "Longest prompt has {longest} tokens; limit is {max_prompt_tokens}. Nothing was truncated."
                )));
            }
            Ok((
                req.model,
                PreparedDecisionBatch::Xor {
                    temperature: *temperature,
                    lev_noul_9_scale: *lev_noul_9_scale,
                    fields: req.fields,
                    variants,
                },
            ))
        }
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
    media: Option<(ClefMediaTokens, &[usize])>,
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
        let ids: Vec<i32> = if !media_prefix.is_empty() {
            const USER_OPEN: &str = "<|im_end|>\n<|im_start|>user\n";
            if let Some(u_pos) = safe_p.find(USER_OPEN) {
                let split_at = u_pos + USER_OPEN.len();
                let head_enc = tokenizer.encode(&safe_p[..split_at], false).map_err(|e| {
                    DecisionError::internal(format!("{model_label}: tokenize: {e}"))
                })?;
                let body_enc = tokenizer.encode(&safe_p[split_at..], false).map_err(|e| {
                    DecisionError::internal(format!("{model_label}: tokenize: {e}"))
                })?;
                let mut combined =
                    Vec::with_capacity(head_enc.len() + media_prefix.len() + body_enc.len());
                combined.extend(head_enc.get_ids().iter().map(|&x| x as i32));
                combined.extend_from_slice(&media_prefix);
                combined.extend(body_enc.get_ids().iter().map(|&x| x as i32));
                combined
            } else if safe_p.contains("<|im_start|>") {
                return Err(DecisionError::internal(format!(
                    "{model_label}: ChatML prompt missing user turn boundary"
                )));
            } else {
                // Non-ChatML plain-text prompts (Xor / Lev) begin directly with state at offset 0.
                let enc = tokenizer.encode(safe_p.as_str(), false).map_err(|e| {
                    DecisionError::internal(format!("{model_label}: tokenize: {e}"))
                })?;
                let mut combined = Vec::with_capacity(media_prefix.len() + enc.len());
                combined.extend_from_slice(&media_prefix);
                combined.extend(enc.get_ids().iter().map(|&x| x as i32));
                combined
            }
        } else {
            let enc = tokenizer
                .encode(safe_p.as_str(), false)
                .map_err(|e| DecisionError::internal(format!("{model_label}: tokenize: {e}")))?;
            enc.get_ids().iter().map(|&x| x as i32).collect()
        };
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

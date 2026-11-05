use std::sync::Arc;

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
};
use serde_json::{Value, json};

use crate::engine_worker::GenerationEvent;

use super::http::{
    ApiError, OpenAiServerState, admission_error, bad, cpu, error, read_json, worker_api_err,
};

fn decision_err(err: crate::decision::DecisionError) -> ApiError {
    if err.is_bad_request {
        bad(err.message)
    } else {
        error(StatusCode::INTERNAL_SERVER_ERROR, err.message)
    }
}

fn shared_mamba_branch_hint<'a>(
    first: &[i32],
    others: impl IntoIterator<Item = &'a [i32]>,
    cap: Option<usize>,
) -> Option<usize> {
    let mut lcp = first.len();
    if let Some(c) = cap {
        lcp = lcp.min(c);
    }
    let mut has_other = false;
    for other in others {
        has_other = true;
        let common = first[..lcp]
            .iter()
            .zip(other.iter())
            .take_while(|(a, b)| a == b)
            .count();
        lcp = lcp.min(common);
    }
    if !has_other {
        return None;
    }
    let aligned = (lcp / 64) * 64;
    (aligned >= 64 && aligned < first.len()).then_some(aligned)
}

async fn run_one_candidate_query(
    s: Arc<OpenAiServerState>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    cands: Vec<i32>,
    branch_hint: Option<usize>,
) -> Result<(Vec<f64>, usize), ApiError> {
    let (mut stream, rx) = s
        .worker
        .submit_decision_candidate_logits_with_images(prompt, images, cands, branch_hint)
        .map_err(worker_api_err)?;
    let tap = rx
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "decision worker dropped"))?
        .map_err(worker_api_err)?;
    let mut cached_done = 0usize;
    while let Some(ev) = stream.events.recv().await {
        match ev {
            GenerationEvent::Finished { cached_tokens, .. } => {
                cached_done = cached_tokens;
                break;
            }
            GenerationEvent::Error(err) => {
                return Err(worker_api_err(err));
            }
            GenerationEvent::Token { .. } => {}
        }
    }
    match tap {
        crate::engine_worker::DecisionTapOutput::CandidateLogits {
            logits,
            cached_tokens,
        } => Ok((logits, cached_tokens.max(cached_done))),
        _ => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected decision tap variant",
        )),
    }
}

async fn run_one_hidden_query(
    s: Arc<OpenAiServerState>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    positions: Vec<usize>,
    prefix_cap: Option<usize>,
    branch_hint: Option<usize>,
) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
    let (mut stream, rx) = s
        .worker
        .submit_decision_hidden_positions_with_images(
            prompt,
            images,
            positions,
            prefix_cap,
            branch_hint,
        )
        .map_err(worker_api_err)?;
    let tap = rx
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "decision worker dropped"))?
        .map_err(worker_api_err)?;
    let mut cached_done = 0usize;
    while let Some(ev) = stream.events.recv().await {
        match ev {
            GenerationEvent::Finished { cached_tokens, .. } => {
                cached_done = cached_tokens;
                break;
            }
            GenerationEvent::Error(err) => {
                return Err(worker_api_err(err));
            }
            GenerationEvent::Token { .. } => {}
        }
    }
    match tap {
        crate::engine_worker::DecisionTapOutput::HiddenRows {
            rows,
            cached_tokens,
        } => Ok((rows, cached_tokens.max(cached_done))),
        _ => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected decision tap variant",
        )),
    }
}

async fn run_one_clef_query(
    s: Arc<OpenAiServerState>,
    record: Arc<crate::clef::ClefEncodedRecord>,
    images: Vec<crate::vision::PreprocessedImage>,
) -> Result<(Vec<Vec<f64>>, usize), ApiError> {
    let (mut stream, rx) = s
        .worker
        .submit_decision_clef_with_images(record, images)
        .map_err(worker_api_err)?;
    let tap = rx
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "decision worker dropped"))?
        .map_err(worker_api_err)?;
    let mut cached_done = 0usize;
    while let Some(ev) = stream.events.recv().await {
        match ev {
            GenerationEvent::Finished { cached_tokens, .. } => {
                cached_done = cached_tokens;
                break;
            }
            GenerationEvent::Error(err) => {
                return Err(worker_api_err(err));
            }
            GenerationEvent::Token { .. } => {}
        }
    }
    match tap {
        crate::engine_worker::DecisionTapOutput::ClefLogits {
            logits,
            cached_tokens,
        } => Ok((logits, cached_tokens.max(cached_done))),
        _ => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected decision tap variant",
        )),
    }
}

async fn run_candidate_queries_radix_shared(
    s: &Arc<OpenAiServerState>,
    queries: Vec<(Vec<i32>, Vec<i32>)>,
    images: &[crate::vision::PreprocessedImage],
) -> Result<Vec<(Vec<f64>, usize)>, ApiError> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let mut results = Vec::with_capacity(queries.len());
    let mut iter = queries.into_iter();
    // Run Query 0 first to donate the shared state prefix into MambaRadixCache.
    let (p0, c0) = iter.next().unwrap();
    let rest: Vec<_> = iter.collect();
    let branch0 = shared_mamba_branch_hint(&p0, rest.iter().map(|(p, _)| p.as_slice()), None);
    results.push(run_one_candidate_query(s.clone(), p0, images.to_vec(), c0, branch0).await?);

    for wave in rest.chunks(8) {
        let mut handles = Vec::with_capacity(wave.len());
        for (p, c) in wave.iter().cloned() {
            let st = s.clone();
            let imgs = images.to_vec();
            handles.push(tokio::spawn(async move {
                run_one_candidate_query(st, p, imgs, c, None).await
            }));
        }
        for h in handles {
            let out = h
                .await
                .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))??;
            results.push(out);
        }
    }
    Ok(results)
}

async fn execute_systemone_json(
    s: &Arc<OpenAiServerState>,
    body: Value,
) -> Result<Value, ApiError> {
    let prep_state = s.clone();
    let (req_model, batch, images) = cpu(s, move || {
        Ok::<_, String>((|| {
            let Some(tokenizer) = prep_state.worker.tokenizer.as_ref() else {
                return Err(bad("loaded model has no tokenizer"));
            };
            let vctx = crate::decision::DecisionVisionContext {
                vision_config: prep_state.worker.vision_config.clone(),
                has_vision_weights: prep_state.worker.has_vision_weights,
                allow_remote_images: prep_state.allow_remote_images,
            };
            let images = crate::decision::parse_decision_images(
                &body,
                &prep_state.worker.decision_profile,
                &vctx,
            )
            .map_err(decision_err)?;
            let (req_model, batch) = crate::decision::prepare_decision_batch_with_images(
                &prep_state.worker.decision_profile,
                tokenizer,
                prep_state.worker.kv_token_capacity,
                &body,
                prep_state.worker.vision_config.as_ref(),
                &images,
            )
            .map_err(decision_err)?;
            Ok((req_model, batch, images))
        })())
    })
    .await??;

    let mut answers = serde_json::Map::new();
    let mut input_tokens = 0usize;
    let mut cached_tokens = 0usize;

    match batch {
        crate::decision::PreparedDecisionBatch::Clef { fields, record } => {
            input_tokens = record.input_ids.len();
            let record_arc = Arc::new(record);
            let (all_logits, cached) =
                run_one_clef_query(s.clone(), record_arc.clone(), images).await?;
            cached_tokens += cached;
            for (field, (q_enc, raw_logits)) in fields
                .iter()
                .zip(record_arc.questions.iter().zip(all_logits.iter()))
            {
                let ordered = crate::clef::reorder_clef_question_logits(q_enc, field, raw_logits)
                    .map_err(decision_err)?;
                let ans = crate::decision::answer_from_logits(
                    field,
                    &ordered,
                    1.0,
                    crate::decision::ConfidenceKind::MaxProbability,
                )
                .map_err(decision_err)?;
                answers.insert(field.name.clone(), ans);
            }
        }
        crate::decision::PreparedDecisionBatch::CandidateLogits {
            temperature,
            conf_kind,
            fields,
            prompts,
            candidates,
            ..
        } => {
            for p in &prompts {
                input_tokens += p.len();
            }
            let pairs: Vec<_> = prompts.into_iter().zip(candidates).collect();
            let outs = run_candidate_queries_radix_shared(s, pairs, &images).await?;
            for (field, (logits, cached)) in fields.iter().zip(outs) {
                cached_tokens += cached;
                let ans =
                    crate::decision::answer_from_logits(field, &logits, temperature, conf_kind)
                        .map_err(decision_err)?;
                answers.insert(field.name.clone(), ans);
            }
        }
        crate::decision::PreparedDecisionBatch::Kev {
            head,
            temperature,
            fields,
            encoded,
        } => {
            let mut queries = Vec::with_capacity(encoded.len());
            for q_enc in &encoded {
                input_tokens += q_enc.token_ids.len();
                let mut positions = Vec::with_capacity(1 + q_enc.option_indices.len());
                positions.push(q_enc.decide_index);
                positions.extend_from_slice(&q_enc.option_indices);
                let min_pos = positions.iter().copied().min().unwrap_or(0);
                queries.push((q_enc.token_ids.clone(), positions, Some(min_pos)));
            }
            let mut outs = Vec::with_capacity(queries.len());
            let mut iter = queries.into_iter();
            if let Some((p0, pos0, cap0)) = iter.next() {
                let rest: Vec<_> = iter.collect();
                let branch0 =
                    shared_mamba_branch_hint(&p0, rest.iter().map(|(p, _, _)| p.as_slice()), cap0);
                outs.push(
                    run_one_hidden_query(s.clone(), p0, images.clone(), pos0, cap0, branch0)
                        .await?,
                );
                for wave in rest.chunks(8) {
                    let mut handles = Vec::with_capacity(wave.len());
                    for (p, pos, cap) in wave.iter().cloned() {
                        let st = s.clone();
                        let imgs = images.clone();
                        handles.push(tokio::spawn(async move {
                            run_one_hidden_query(st, p, imgs, pos, cap, None).await
                        }));
                    }
                    for h in handles {
                        let out = h
                            .await
                            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))??;
                        outs.push(out);
                    }
                }
            }
            for (field, (rows, cached)) in fields.iter().zip(outs) {
                cached_tokens += cached;
                let n_opts = field.keys.len();
                let h_decide = &rows[0];
                let h_opts: Vec<f32> = rows[1..].iter().flatten().copied().collect();
                let raw_logits = head
                    .forward(h_decide, &h_opts, n_opts)
                    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
                let logits_f64: Vec<f64> = raw_logits.into_iter().map(|x| x as f64).collect();
                let ans = crate::decision::answer_from_logits(
                    field,
                    &logits_f64,
                    temperature,
                    crate::decision::ConfidenceKind::MarginAndModeDistance,
                )
                .map_err(decision_err)?;
                answers.insert(field.name.clone(), ans);
            }
        }
        crate::decision::PreparedDecisionBatch::Xor {
            temperature,
            lev_noul_9_scale,
            fields,
            variants,
        } => {
            let mut pairs = Vec::with_capacity(variants.len() * 2);
            for (fwd, rev, cands) in variants {
                input_tokens += fwd.len() + rev.len();
                pairs.push((fwd, cands.clone()));
                pairs.push((rev, cands));
            }
            let outs = run_candidate_queries_radix_shared(s, pairs, &images).await?;
            for (i, field) in fields.iter().enumerate() {
                let (fwd_logits, c_fwd) = &outs[2 * i];
                let (rev_logits, c_rev) = &outs[2 * i + 1];
                cached_tokens += *c_fwd + *c_rev;
                let probs = crate::decision::calibrate_forward_reverse_logits(
                    fwd_logits,
                    rev_logits,
                    temperature,
                )
                .map_err(decision_err)?;
                let ans = if lev_noul_9_scale && field.field_type == "noul" {
                    let expected: f64 = probs
                        .iter()
                        .enumerate()
                        .map(|(idx, &p)| p * (idx as f64) / 8.0)
                        .sum();
                    json!({
                        "type": "noul",
                        "noul": expected,
                    })
                } else {
                    crate::decision::answer_from_probabilities(
                        field,
                        &probs,
                        crate::decision::ConfidenceKind::MarginAndModeDistance,
                    )
                    .map_err(decision_err)?
                };
                answers.insert(field.name.clone(), ans);
            }
        }
    }

    Ok(json!({
        "model": req_model.unwrap_or_else(|| s.model_name.clone()),
        "answers": Value::Object(answers),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cached_tokens": cached_tokens,
        }
    }))
}

pub(super) async fn systemone_decide(
    State(s): State<Arc<OpenAiServerState>>,
    request: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(admission_error)?;
    let body = read_json(&s, request).await?;
    let resp = execute_systemone_json(&s, body).await?;
    Ok(Json(resp))
}

pub(super) async fn decisions_adapter(
    State(s): State<Arc<OpenAiServerState>>,
    request: Request,
) -> Result<Json<Value>, ApiError> {
    let _admission = s
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(admission_error)?;
    let body = read_json(&s, request).await?;
    let canonical =
        crate::decision::normalize_decisions_adapter_request(&body).map_err(decision_err)?;
    let resp = execute_systemone_json(&s, canonical).await?;
    Ok(Json(crate::decision::format_decisions_adapter_response(
        &resp,
    )))
}

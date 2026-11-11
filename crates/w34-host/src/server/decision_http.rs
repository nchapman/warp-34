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
    prefix_bidir_len: Option<usize>,
    branch_hint: Option<usize>,
) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
    let (mut stream, rx) = s
        .worker
        .submit_decision_hidden_positions_with_images(
            prompt,
            images,
            positions,
            prefix_cap,
            prefix_bidir_len,
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

async fn run_code_readout_queries(
    s: &Arc<OpenAiServerState>,
    head: &crate::decision::CodeReadoutHead,
    fields: &[crate::decision::DecisionField],
    queries: Vec<crate::decision::CodeReadoutPreparedQuery>,
    images: &[crate::vision::PreprocessedImage],
) -> Result<(Vec<Vec<f64>>, usize, usize), ApiError> {
    if queries.is_empty() {
        return Ok((Vec::new(), 0, 0));
    }
    let mut input_tokens = 0usize;
    let mut cached_tokens = 0usize;
    for q in &queries {
        input_tokens += q.token_ids.len();
    }

    let mut raw_outs = Vec::with_capacity(queries.len());
    let mut iter = queries.into_iter();
    let q0 = iter.next().unwrap();
    let rest: Vec<_> = iter.collect();
    let branch0 = if q0.prefix_bidir_len.is_none() {
        shared_mamba_branch_hint(
            &q0.token_ids,
            rest.iter().map(|q| q.token_ids.as_slice()),
            q0.prefix_cap,
        )
    } else {
        None
    };
    raw_outs.push(
        run_one_hidden_query(
            s.clone(),
            q0.token_ids,
            images.to_vec(),
            q0.positions,
            q0.prefix_cap,
            q0.prefix_bidir_len,
            branch0,
        )
        .await?,
    );
    for wave in rest.chunks(8) {
        let mut handles = Vec::with_capacity(wave.len());
        for q in wave.iter().cloned() {
            let st = s.clone();
            let imgs = images.to_vec();
            handles.push(tokio::spawn(async move {
                run_one_hidden_query(
                    st,
                    q.token_ids,
                    imgs,
                    q.positions,
                    q.prefix_cap,
                    q.prefix_bidir_len,
                    None,
                )
                .await
            }));
        }
        for h in handles {
            let out = h
                .await
                .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))??;
            raw_outs.push(out);
        }
    }

    let mut all_rows = Vec::with_capacity(fields.len());
    for (rows, cached) in raw_outs {
        cached_tokens += cached;
        all_rows.extend(rows);
    }
    if all_rows.len() != fields.len() {
        return Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "code_readout returned {} hidden rows for {} fields",
                all_rows.len(),
                fields.len()
            ),
        ));
    }

    let mut field_logits = Vec::with_capacity(fields.len());
    for (field, h) in fields.iter().zip(all_rows.iter()) {
        let logits = head
            .forward(h, field.keys.len())
            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        field_logits.push(logits);
    }
    Ok((field_logits, input_tokens, cached_tokens))
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
        crate::decision::PreparedDecisionBatch::CodeReadout {
            head,
            temperature,
            conf_kind,
            fields,
            forward_queries,
            reverse_queries,
        } => {
            let (fwd_logits, in_fwd, c_fwd) =
                run_code_readout_queries(s, &head, &fields, forward_queries, &images).await?;
            input_tokens += in_fwd;
            cached_tokens += c_fwd;

            if let Some(rev_queries) = reverse_queries {
                let (rev_logits, in_rev, c_rev) =
                    run_code_readout_queries(s, &head, &fields, rev_queries, &images).await?;
                input_tokens += in_rev;
                cached_tokens += c_rev;
                for (field, (fwd, rev)) in
                    fields.iter().zip(fwd_logits.iter().zip(rev_logits.iter()))
                {
                    let probs =
                        crate::decision::calibrate_forward_reverse_logits(fwd, rev, temperature)
                            .map_err(decision_err)?;
                    let ans = crate::decision::answer_from_probabilities(field, &probs, conf_kind)
                        .map_err(decision_err)?;
                    answers.insert(field.name.clone(), ans);
                }
            } else {
                for (field, logits) in fields.iter().zip(fwd_logits.iter()) {
                    let ans =
                        crate::decision::answer_from_logits(field, logits, temperature, conf_kind)
                            .map_err(decision_err)?;
                    answers.insert(field.name.clone(), ans);
                }
            }
        }
        crate::decision::PreparedDecisionBatch::CandidateLogits {
            temperature,
            conf_kind,
            fields,
            prompts,
            reverse_prompts,
            candidates,
            ..
        } => {
            for p in &prompts {
                input_tokens += p.len();
            }
            if let Some(rev_prompts) = reverse_prompts {
                for p in &rev_prompts {
                    input_tokens += p.len();
                }
                let mut pairs = Vec::with_capacity(fields.len() * 2);
                for ((fwd, rev), cands) in prompts.into_iter().zip(rev_prompts).zip(candidates) {
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
                    let ans = crate::decision::answer_from_probabilities(field, &probs, conf_kind)
                        .map_err(decision_err)?;
                    answers.insert(field.name.clone(), ans);
                }
            } else {
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

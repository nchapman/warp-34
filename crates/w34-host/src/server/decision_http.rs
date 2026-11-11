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

type SubmittedDecisionQuery = (
    crate::engine_worker::GenerationStream,
    tokio::sync::oneshot::Receiver<
        Result<crate::engine_worker::DecisionTapOutput, crate::engine_worker::WorkerError>,
    >,
);

async fn await_candidate_query(
    (mut stream, rx): SubmittedDecisionQuery,
) -> Result<(Vec<f64>, usize), ApiError> {
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

async fn run_one_candidate_query(
    s: Arc<OpenAiServerState>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    cands: Vec<i32>,
    branch_hint: Option<usize>,
) -> Result<(Vec<f64>, usize), ApiError> {
    let sub = s
        .worker
        .submit_decision_candidate_logits_with_images(prompt, images, cands, branch_hint)
        .map_err(worker_api_err)?;
    await_candidate_query(sub).await
}

async fn await_hidden_query(
    (mut stream, rx): SubmittedDecisionQuery,
) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
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

async fn run_one_hidden_query(
    s: Arc<OpenAiServerState>,
    prompt: Vec<i32>,
    images: Vec<crate::vision::PreprocessedImage>,
    positions: Vec<usize>,
    prefix_cap: Option<usize>,
    prefix_bidir_len: Option<usize>,
    branch_hint: Option<usize>,
) -> Result<(Vec<Vec<f32>>, usize), ApiError> {
    let sub = s
        .worker
        .submit_decision_hidden_positions_with_images(
            prompt,
            images,
            positions,
            prefix_cap,
            branch_hint,
            prefix_bidir_len,
        )
        .map_err(worker_api_err)?;
    await_hidden_query(sub).await
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
        let mut submitted = Vec::with_capacity(wave.len());
        for (p, c) in wave.iter().cloned() {
            submitted.push(
                s.worker
                    .submit_decision_candidate_logits_with_images(p, images.to_vec(), c, None)
                    .map_err(worker_api_err)?,
            );
        }
        for sub in submitted {
            results.push(await_candidate_query(sub).await?);
        }
    }
    Ok(results)
}

async fn run_code_readout_queries_raw(
    s: &Arc<OpenAiServerState>,
    queries: Vec<crate::decision::CodeReadoutPreparedQuery>,
    images: &[crate::vision::PreprocessedImage],
) -> Result<(Vec<Vec<f32>>, usize, usize), ApiError> {
    if queries.is_empty() {
        return Ok((Vec::new(), 0, 0));
    }
    let mut input_tokens = 0usize;
    let mut cached_tokens = 0usize;
    let mut total_positions = 0usize;
    for q in &queries {
        input_tokens += q.token_ids.len();
        total_positions += q.positions.len();
    }

    let mut raw_outs = Vec::with_capacity(queries.len());
    let all_bidir = queries.iter().all(|q| q.prefix_bidir_len.is_some());
    if all_bidir {
        // Bidirectional prefixes (`d3` `prefix_bidir = true`) do not donate into
        // `MambaRadixCache`; submit all queries synchronously into the worker queue
        // so `order_passes == 2` forward + reverse passes batch together in lockstep.
        for wave in queries.chunks(8) {
            let mut submitted = Vec::with_capacity(wave.len());
            for q in wave.iter().cloned() {
                submitted.push(
                    s.worker
                        .submit_decision_hidden_positions_with_images(
                            q.token_ids,
                            images.to_vec(),
                            q.positions,
                            q.prefix_cap,
                            None,
                            q.prefix_bidir_len,
                        )
                        .map_err(worker_api_err)?,
                );
            }
            for sub in submitted {
                raw_outs.push(await_hidden_query(sub).await?);
            }
        }
    } else {
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
            let mut submitted = Vec::with_capacity(wave.len());
            for q in wave.iter().cloned() {
                submitted.push(
                    s.worker
                        .submit_decision_hidden_positions_with_images(
                            q.token_ids,
                            images.to_vec(),
                            q.positions,
                            q.prefix_cap,
                            None,
                            q.prefix_bidir_len,
                        )
                        .map_err(worker_api_err)?,
                );
            }
            for sub in submitted {
                raw_outs.push(await_hidden_query(sub).await?);
            }
        }
    }

    let mut all_rows = Vec::with_capacity(total_positions);
    for (rows, cached) in raw_outs {
        cached_tokens += cached;
        all_rows.extend(rows);
    }
    Ok((all_rows, input_tokens, cached_tokens))
}

fn project_code_readout_rows(
    head: &crate::decision::CodeReadoutHead,
    fields: &[crate::decision::DecisionField],
    rows: &[Vec<f32>],
) -> Result<Vec<Vec<f64>>, ApiError> {
    if rows.len() != fields.len() {
        return Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "code_readout returned {} hidden rows for {} fields",
                rows.len(),
                fields.len()
            ),
        ));
    }
    let mut field_logits = Vec::with_capacity(fields.len());
    for (field, h) in fields.iter().zip(rows.iter()) {
        let logits = head
            .forward(h, field.keys.len())
            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        field_logits.push(logits);
    }
    Ok(field_logits)
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
            if let Some(rev_queries) = reverse_queries {
                let mut combined = Vec::with_capacity(forward_queries.len() + rev_queries.len());
                combined.extend(forward_queries);
                combined.extend(rev_queries);
                let (all_rows, in_toks, c_toks) =
                    run_code_readout_queries_raw(s, combined, &images).await?;
                input_tokens += in_toks;
                cached_tokens += c_toks;
                let n = fields.len();
                if all_rows.len() != 2 * n {
                    return Err(error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(
                            "code_readout order_passes=2 returned {} hidden rows for {} fields",
                            all_rows.len(),
                            n
                        ),
                    ));
                }
                let fwd_logits = project_code_readout_rows(&head, &fields, &all_rows[..n])?;
                let rev_logits = project_code_readout_rows(&head, &fields, &all_rows[n..])?;
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
                let (all_rows, in_toks, c_toks) =
                    run_code_readout_queries_raw(s, forward_queries, &images).await?;
                input_tokens += in_toks;
                cached_tokens += c_toks;
                let fwd_logits = project_code_readout_rows(&head, &fields, &all_rows)?;
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

//! Focused CPU-only sampling API tests.
use w34_host::parser::{SamplerV1, SamplingParams, TokenBitmaskConstraint};

#[test]
fn native_http_sampling_parameters_are_validated() {
    use serde_json::json;
    use w34_host::server::native_sampling_params;
    let params = native_sampling_params(
        &json!({"temperature":0.8,"top_k":12,"top_p":0.9,"min_p":0.02,
        "repetition_penalty":1.2,"frequency_penalty":0.1,"presence_penalty":0.2,"seed":72}),
        8,
    )
    .unwrap();
    assert_eq!(params.max_tokens, 8);
    assert_eq!(params.seed, 72);
    assert_eq!(params.top_k, 12);
    assert_eq!(params.temperature, 0.8);
    assert_eq!(params.top_p, 0.9);
    for bad in [
        json!({"temperature":-1}),
        json!({"temperature":"0.8"}),
        json!({"top_p":0}),
        json!({"top_k":1e20}),
        json!({"seed":-1}),
        json!({"repetition_penalty":0}),
        json!({"stop":"END"}),
    ] {
        assert!(native_sampling_params(&bad, 8).is_err(), "{bad}");
    }
}

fn stochastic() -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        ..Default::default()
    }
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
    }
    assert!((actual.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn dense_normalized_and_sorted_seed_cdf_without_mutation() {
    let mut s = SamplerV1::new(stochastic(), &[], None);
    let logits = [0.0, 3.0f32.ln(), 2.0f32.ln()];
    let before = format!("{s:?}");
    close(
        &s.processed_distribution(&logits).unwrap(),
        &[1.0 / 6.0, 0.5, 1.0 / 3.0],
    );
    assert_eq!(format!("{s:?}"), before);
    let u = SamplerV1::uniform_f32(s.params.seed, 0);
    let expected = if u <= 0.5 {
        1
    } else if u <= 5.0 / 6.0 {
        2
    } else {
        0
    };
    assert_eq!(s.sample_from_logits(&logits), expected);
    assert_eq!(s.output_history, vec![expected]);
    assert_eq!(u, SamplerV1::uniform_f32(42, 0));
    assert!((0.0..1.0).contains(&u));
}

#[test]
fn greedy_constraint_ties_and_grammar_rollback() {
    let c = TokenBitmaskConstraint::json_object(3, vec![1], vec![2]);
    let mut s = SamplerV1::new(SamplingParams::default(), &[], Some(c));
    let before = format!("{s:?}");
    close(
        &s.processed_distribution(&[9.0, 1.0, 9.0]).unwrap(),
        &[0.0, 1.0, 0.0],
    );
    assert_eq!(before, format!("{s:?}"));
    s.commit_token(1);
    close(
        &s.processed_distribution(&[9.0, 1.0, 9.0]).unwrap(),
        &[1.0, 0.0, 0.0],
    );
    s.rollback_tokens(1);
    assert_eq!(before, format!("{s:?}"));
}

#[test]
fn prompt_output_penalties_and_rollback() {
    let params = SamplingParams {
        repetition_penalty: 2.0,
        frequency_penalty: 0.5,
        presence_penalty: 0.25,
        ..stochastic()
    };
    let mut s = SamplerV1::new(params, &[0, 0, 2], None);
    let logits = [4.0, 2.0, -1.0];
    let original = s.processed_distribution(&logits).unwrap();
    s.commit_token(1);
    s.commit_token(1);
    let weights = [2.0f32.exp(), (-0.25f32).exp(), (-2.0f32).exp()];
    let sum: f32 = weights.iter().sum();
    close(
        &s.processed_distribution(&logits).unwrap(),
        &weights.map(|w| w / sum),
    );
    s.rollback_tokens(2);
    assert_eq!(s.processed_distribution(&logits).unwrap(), original);
}

#[test]
fn topk_minp_topp_in_policy_order() {
    let logits = [8.0f32.ln(), 4.0f32.ln(), 2.0f32.ln(), 0.0];
    for (k, min_p, top_p, expected) in [
        (2, 0.0, 1.0, [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]),
        (-1, 0.3, 1.0, [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]),
        (3, 0.3, 0.6, [1.0, 0.0, 0.0, 0.0]),
        (-1, 0.0, 0.7, [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]),
    ] {
        let s = SamplerV1::new(
            SamplingParams {
                top_k: k,
                min_p,
                top_p,
                ..stochastic()
            },
            &[],
            None,
        );
        close(&s.processed_distribution(&logits).unwrap(), &expected);
    }
}

#[test]
fn errors_are_nonmutating_and_masked_nonfinite_is_ignored() {
    let mut s = SamplerV1::new(stochastic(), &[], None);
    for logits in [
        vec![],
        vec![f32::NAN],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
    ] {
        let before = format!("{s:?}");
        assert!(s.processed_distribution(&logits).is_err());
        assert_eq!(before, format!("{s:?}"));
    }
    assert_eq!(s.sample_from_logits(&[]), 0);
    for params in [
        SamplingParams {
            temperature: -1.0,
            ..stochastic()
        },
        SamplingParams {
            temperature: f32::NAN,
            ..stochastic()
        },
        SamplingParams {
            top_k: -2,
            ..stochastic()
        },
        SamplingParams {
            top_p: 0.0,
            ..stochastic()
        },
        SamplingParams {
            min_p: 1.1,
            ..stochastic()
        },
        SamplingParams {
            repetition_penalty: 0.0,
            ..stochastic()
        },
        SamplingParams {
            frequency_penalty: f32::INFINITY,
            ..stochastic()
        },
        SamplingParams {
            presence_penalty: f32::NAN,
            ..stochastic()
        },
    ] {
        s.params = params;
        assert!(s.processed_distribution(&[0.0]).is_err());
    }
    s.params = stochastic();
    s.constraint = Some(TokenBitmaskConstraint::from_allowed_tokens(2, vec![1]));
    close(
        &s.processed_distribution(&[f32::NAN, 0.0]).unwrap(),
        &[0.0, 1.0],
    );
    s.constraint = Some(TokenBitmaskConstraint::from_allowed_tokens(2, vec![]));
    assert!(s.processed_distribution(&[0.0, 1.0]).is_err());
    let before = format!("{s:?}");
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || s.sample_from_logits(&[0.0, 1.0])
        ))
        .is_err()
    );
    assert_eq!(before, format!("{s:?}"));
}

//! Host rejection sampler and transactional rollback unit/integration tests.
use w34_host::cache::{StateTx, StateTxPhase};
use w34_host::parser::{SamplerV1, SamplingParams, TokenBitmaskConstraint};
use w34_host::speculative::RejectionSampler;

#[test]
fn greedy_commits_only_consecutive_matches_at_every_rejection_position() {
    let drafts = [10, 11, 12];
    for accepted in 0..=drafts.len() {
        let mut target = vec![10, 11, 12, 99];
        target[accepted] = 99;
        let result = RejectionSampler::greedy(&drafts, &target).unwrap();
        assert_eq!(result.accepted_drafts(), accepted);
        assert_eq!(result.processed_rows(), accepted + 1);
        let mut expected = drafts[..accepted].to_vec();
        expected.push(99);
        assert_eq!(result.emitted_tokens(), expected);
    }
}

#[test]
fn empty_draft_samples_target_bonus() {
    assert_eq!(
        RejectionSampler::greedy(&[], &[7])
            .unwrap()
            .emitted_tokens(),
        &[7]
    );
    let result = RejectionSampler::stochastic(&[], &[vec![0.0, 1.0]], &[], &[], 0.0).unwrap();
    assert_eq!(result.emitted_tokens(), &[1]);
    assert_eq!(result.processed_rows(), 1);
}

#[test]
fn rejection_uses_positive_residual_not_target_distribution() {
    let result = RejectionSampler::stochastic(
        &[0, 1],
        &[vec![0.25, 0.75], vec![0.5, 0.5], vec![1.0, 0.0]],
        &[vec![0.75, 0.25], vec![0.5, 0.5]],
        &[0.9, 0.0],
        0.0,
    )
    .unwrap();
    assert_eq!(result.accepted_drafts(), 0);
    assert_eq!(result.emitted_tokens(), &[1]);
}

#[test]
fn zero_target_mass_rejects_even_at_zero_uniform() {
    let result = RejectionSampler::stochastic(
        &[0],
        &[vec![0.0, 1.0], vec![0.5, 0.5]],
        &[vec![1.0, 0.0]],
        &[0.0],
        0.0,
    )
    .unwrap();
    assert_eq!(result.accepted_drafts(), 0);
    assert_eq!(result.emitted_tokens(), &[1]);
}

#[test]
fn identical_distributions_accept_all_and_sample_bonus() {
    let result = RejectionSampler::stochastic(
        &[0, 1],
        &[vec![0.5, 0.5], vec![0.25, 0.75], vec![0.0, 1.0]],
        &[vec![0.5, 0.5], vec![0.25, 0.75]],
        &[0.999, 0.999],
        0.0,
    )
    .unwrap();
    assert_eq!(result.accepted_drafts(), 2);
    assert_eq!(result.emitted_tokens(), &[0, 1, 1]);
}

#[test]
fn enumerated_proposal_and_acceptance_draws_recover_target_marginal() {
    // Exact stratified quadrature: q=(.75,.25), p=(.25,.75).
    // The first emitted token must follow p, not q or p sampled on rejection.
    let mut counts = [0usize; 2];
    for proposal_draw in 0..4 {
        let draft = if proposal_draw < 3 { 0 } else { 1 };
        for accept_draw in 0..12 {
            let result = RejectionSampler::stochastic(
                &[draft],
                &[vec![0.25, 0.75], vec![0.5, 0.5]],
                &[vec![0.75, 0.25]],
                &[(accept_draw as f64 + 0.5) / 12.0],
                0.5,
            )
            .unwrap();
            counts[result.emitted_tokens()[0] as usize] += 1;
        }
    }
    assert_eq!(counts, [12, 36]);
}

#[test]
fn malformed_probabilities_draws_and_ids_fail_without_panics() {
    let target = [vec![0.5, 0.5], vec![0.5, 0.5]];
    for bad in [
        vec![0.0, 0.0],
        vec![f32::NAN, 1.0],
        vec![-0.5, 1.5],
        vec![1.0],
    ] {
        assert!(RejectionSampler::stochastic(&[0], &target, &[bad], &[0.5], 0.5).is_err());
    }
    for bad in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
        assert!(
            RejectionSampler::stochastic(&[0], &target, &[vec![0.5, 0.5]], &[bad], 0.5).is_err()
        );
        assert!(
            RejectionSampler::stochastic(&[0], &target, &[vec![0.5, 0.5]], &[0.5], bad).is_err()
        );
    }
    for bad in [-1, 2] {
        assert!(
            RejectionSampler::stochastic(&[bad], &target, &[vec![0.5, 0.5]], &[0.5], 0.5).is_err()
        );
    }
    assert!(RejectionSampler::stochastic(&[0], &target, &[vec![0.0, 1.0]], &[0.5], 0.5).is_err());
    assert!(RejectionSampler::greedy(&[1], &[1]).is_err());
    assert!(RejectionSampler::greedy(&[], &[-1]).is_err());
}

#[test]
fn transaction_rolls_back_tentative_sampler_and_grammar_and_selects_state() {
    let mut tx = StateTx::new(3);
    tx.commit_prefill(63);
    tx.begin_spec_verify(vec![4, 5, 6]);
    let constraint = TokenBitmaskConstraint::from_allowed_tokens(100, vec![7, 10, 11, 12, 99]);
    let mut sampler = SamplerV1::new(SamplingParams::default(), &[], Some(constraint));
    sampler.commit_token(7); // input anchor, already emitted
    for token in [10, 11, 12] {
        sampler.commit_token(token);
    }
    let result = RejectionSampler::greedy(&[10, 11, 12], &[10, 99, 12, 99]).unwrap();
    assert_eq!(
        result.commit(&mut tx, &mut sampler, 3).unwrap(),
        vec![3, 5, 6]
    );
    assert_eq!(tx.committed_slot, 4);
    assert_eq!(tx.kv_seq_len_before, 65); // anchor+10, NOT recovered token 99
    assert_eq!(tx.phase, StateTxPhase::RolledBack);
    assert_eq!(sampler.output_history, [7, 10, 99]);
    assert_eq!(sampler.constraint.as_ref().unwrap().committed_len(), 3);
    assert!(!sampler.output_counts.contains_key(&11));
    assert!(!sampler.output_counts.contains_key(&12));
}

#[test]
fn commit_replays_accepted_tokens_when_proposer_used_private_sampler_state() {
    for provisional in 0..=3 {
        let mut tx = StateTx::new(3);
        tx.commit_prefill(63);
        tx.begin_spec_verify(vec![4, 5, 6]);
        let constraint = TokenBitmaskConstraint::from_allowed_tokens(100, vec![7, 10, 11, 12, 99]);
        let mut sampler = SamplerV1::new(SamplingParams::default(), &[], Some(constraint));
        sampler.commit_token(7);
        for &token in &[10, 11, 12][..provisional] {
            sampler.commit_token(token);
        }
        let result = RejectionSampler::greedy(&[10, 11, 12], &[10, 11, 12, 99]).unwrap();
        result.commit(&mut tx, &mut sampler, provisional).unwrap();
        assert_eq!(sampler.output_history, [7, 10, 11, 12, 99]);
        assert_eq!(sampler.constraint.as_ref().unwrap().committed_len(), 5);
        assert_eq!(sampler.output_counts[&10], 1);
        assert_eq!(tx.committed_slot, 6);
        assert_eq!(tx.kv_seq_len_before, 67);
    }
}

#[test]
fn invalid_commit_does_not_mutate_state_or_sampler() {
    let mut tx = StateTx::new(3);
    tx.commit_prefill(63);
    tx.begin_spec_verify(vec![4]);
    let constraint = TokenBitmaskConstraint::from_allowed_tokens(100, vec![7, 10]);
    let mut sampler = SamplerV1::new(SamplingParams::default(), &[], Some(constraint));
    sampler.commit_token(7);
    sampler.commit_token(10);
    let result = RejectionSampler::greedy(&[10], &[99, 99]).unwrap();
    assert!(result.commit(&mut tx, &mut sampler, 1).is_err());
    assert_eq!(tx.committed_slot, 3);
    assert_eq!(tx.kv_seq_len_before, 63);
    assert_eq!(tx.spec_slots, [4]);
    assert_eq!(tx.phase, StateTxPhase::SpecVerifying);
    assert_eq!(sampler.output_history, [7, 10]);
    assert_eq!(sampler.constraint.as_ref().unwrap().committed_len(), 2);
}

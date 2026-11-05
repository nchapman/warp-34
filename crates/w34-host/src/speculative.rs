//! Host reference for canonical speculative rejection sampling.
//!
//! Oracle: vLLM `v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
//! (standard token-wise verification, not optional block verification).
//! Inputs are *processed*, normalized target/proposal distributions, including
//! temperature, penalties and grammar masks at each hypothetical prefix. This
//! module does not yet connect model logits or proposers to the serving loop.

use crate::cache::{StateTx, StateTxPhase};
use crate::parser::SamplerV1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RejectionError(pub &'static str);

impl std::fmt::Display for RejectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for RejectionError {}

type Result<T> = std::result::Result<T, RejectionError>;

/// Accepted drafts followed by exactly one recovery/bonus token. The latter
/// has not been processed by the target and is the next round's anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verification {
    drafts: Vec<i32>,
    accepted: usize,
    emitted: Vec<i32>,
}

impl Verification {
    pub fn accepted_drafts(&self) -> usize {
        self.accepted
    }

    /// Target rows committed: the input anchor plus accepted drafts.
    pub fn processed_rows(&self) -> usize {
        self.accepted + 1
    }

    /// Newly visible tokens; excludes the already-emitted input anchor.
    pub fn emitted_tokens(&self) -> &[i32] {
        &self.emitted
    }

    /// Resolve target state and sampler/grammar together. `sampler` must contain
    /// exactly `provisional_drafts` tentative proposals after its committed
    /// history (the input anchor is already committed). This count may be zero:
    /// a proposer can use a private sampler clone and leave the target sampler
    /// untouched. Accepted tokens need not have been tentatively appended here;
    /// every emitted token is replayed against the committed grammar below.
    /// On error neither object
    /// changes. Returned losing state slots may be freed after GPU completion.
    /// KV truncation and committed-only radix donation remain the caller's job.
    pub fn commit(
        &self,
        tx: &mut StateTx,
        sampler: &mut SamplerV1,
        provisional_drafts: usize,
    ) -> Result<Vec<i32>> {
        if tx.phase != StateTxPhase::SpecVerifying
            || tx.spec_slots.len() != self.drafts.len()
            || provisional_drafts > self.drafts.len()
            || provisional_drafts > sampler.output_history.len()
        {
            return Err(RejectionError("invalid speculative transaction geometry"));
        }
        let suffix_start = sampler.output_history.len() - provisional_drafts;
        if sampler.output_history[suffix_start..] != self.drafts[..provisional_drafts] {
            return Err(RejectionError(
                "sampler tentative suffix differs from proposals",
            ));
        }
        let schedule = tx.verify_slot_schedule();
        if schedule
            .iter()
            .enumerate()
            .any(|(i, slot)| *slot < 0 || schedule[..i].contains(slot))
        {
            return Err(RejectionError(
                "speculative state slots must be distinct and nonnegative",
            ));
        }
        tx.kv_seq_len_before
            .checked_add(self.processed_rows())
            .ok_or(RejectionError("processed length overflow"))?;

        let mut committed_sampler = sampler.clone();
        committed_sampler
            .try_rollback_tokens(provisional_drafts)
            .map_err(|_| RejectionError("grammar rollback failed"))?;
        for &token in &self.emitted {
            if committed_sampler
                .constraint
                .as_ref()
                .is_some_and(|c| !c.is_token_allowed(token))
            {
                return Err(RejectionError(
                    "verified token violates committed grammar state",
                ));
            }
            committed_sampler
                .try_commit_token(token)
                .map_err(|_| RejectionError("grammar commit failed"))?;
        }
        let freed = tx.resolve_spec_verify(self.processed_rows());
        *sampler = committed_sampler;
        Ok(freed)
    }
}

pub struct RejectionSampler;

impl RejectionSampler {
    /// `target_ids[i]` predicts the token following verify input row i.
    /// Greedy verification stops at the first mismatch; later matches do not count.
    pub fn greedy(drafts: &[i32], target_ids: &[i32]) -> Result<Verification> {
        if target_ids.len() != drafts.len() + 1 || drafts.iter().chain(target_ids).any(|&id| id < 0)
        {
            return Err(RejectionError(
                "expected k nonnegative drafts and k+1 target ids",
            ));
        }
        let accepted = drafts
            .iter()
            .zip(target_ids)
            .take_while(|(d, t)| d == t)
            .count();
        Ok(Self::result(drafts, accepted, target_ids[accepted]))
    }

    /// Accept draft d with probability min(1, p(d)/q(d)); at the first
    /// rejection sample from normalized max(p-q, 0). On full acceptance sample
    /// from target row k. Explicit independent uniforms make oracle replay and
    /// RNG rollback testable; each draw must be in [0, 1).
    pub fn stochastic(
        drafts: &[i32],
        target: &[Vec<f32>],
        proposal: &[Vec<f32>],
        accept_uniforms: &[f64],
        recovery_uniform: f64,
    ) -> Result<Verification> {
        if target.len() != drafts.len() + 1
            || proposal.len() != drafts.len()
            || accept_uniforms.len() != drafts.len()
        {
            return Err(RejectionError(
                "expected k proposal rows/draws and k+1 target rows",
            ));
        }
        if !valid_uniform(recovery_uniform) || accept_uniforms.iter().any(|&u| !valid_uniform(u)) {
            return Err(RejectionError("uniform draws must be finite and in [0, 1)"));
        }
        let vocab = target[0].len();
        if vocab == 0 || vocab > i32::MAX as usize {
            return Err(RejectionError("invalid vocabulary size"));
        }
        // Validate the entire batch before selection, including unused suffixes.
        let p: Vec<Vec<f64>> = target
            .iter()
            .map(|row| distribution(row, vocab))
            .collect::<Result<_>>()?;
        let q: Vec<Vec<f64>> = proposal
            .iter()
            .map(|row| distribution(row, vocab))
            .collect::<Result<_>>()?;
        for (i, &id) in drafts.iter().enumerate() {
            if id < 0 || id as usize >= vocab || q[i][id as usize] == 0.0 {
                return Err(RejectionError(
                    "draft token must have positive proposal probability",
                ));
            }
        }
        for (i, &id) in drafts.iter().enumerate() {
            let id = id as usize;
            if accept_uniforms[i] < (p[i][id] / q[i][id]).min(1.0) {
                continue;
            }
            let residual: Vec<f64> = p[i]
                .iter()
                .zip(&q[i])
                .map(|(p, q)| (p - q).max(0.0))
                .collect();
            let recovered = categorical(&residual, recovery_uniform)?;
            return Ok(Self::result(drafts, i, recovered));
        }
        Ok(Self::result(
            drafts,
            drafts.len(),
            categorical(&p[drafts.len()], recovery_uniform)?,
        ))
    }

    /// Constructs a `Verification` directly from GPU-resident rejection sampling outputs.
    pub fn from_gpu_verified(
        drafts: &[i32],
        accepted: usize,
        next_anchor: i32,
    ) -> Result<Verification> {
        if accepted > drafts.len() || next_anchor < 0 || drafts.iter().any(|&id| id < 0) {
            return Err(RejectionError("invalid GPU rejection verification result"));
        }
        Ok(Self::result(drafts, accepted, next_anchor))
    }

    fn result(drafts: &[i32], accepted: usize, next_anchor: i32) -> Verification {
        let mut emitted = drafts[..accepted].to_vec();
        emitted.push(next_anchor);
        Verification {
            drafts: drafts.to_vec(),
            accepted,
            emitted,
        }
    }
}

fn valid_uniform(u: f64) -> bool {
    u.is_finite() && (0.0..1.0).contains(&u)
}

fn distribution(row: &[f32], vocab: usize) -> Result<Vec<f64>> {
    if row.len() != vocab || row.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Err(RejectionError(
            "probability rows must have matching width and finite nonnegative entries",
        ));
    }
    let sum: f64 = row.iter().map(|&p| f64::from(p)).sum();
    if (sum - 1.0).abs() > 1e-5 {
        return Err(RejectionError("probability rows must sum to one"));
    }
    Ok(row.iter().map(|&p| f64::from(p) / sum).collect())
}

fn categorical(weights: &[f64], uniform: f64) -> Result<i32> {
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(RejectionError("recovery distribution has no positive mass"));
    }
    let threshold = uniform * total;
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (id, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            last_positive = Some(id as i32);
            cumulative += weight;
            if threshold < cumulative {
                return Ok(id as i32);
            }
        }
    }
    Ok(last_positive.expect("positive total implies positive entry"))
}

/// Searches `history` (most-recent match first) for a suffix of `tail` of length in
/// `min_ngram..=max_ngram` and returns the `take` tokens immediately following the match.
pub fn lookup_suffix_continuation(
    history: &[i32],
    tail: &[i32],
    min_ngram: usize,
    max_ngram: usize,
    take: usize,
) -> Option<Vec<i32>> {
    if min_ngram == 0
        || min_ngram > max_ngram
        || take == 0
        || history.len() < min_ngram + take
        || tail.len() < min_ngram
    {
        return None;
    }
    let searchable_end = history.len() - take;
    let hi = max_ngram.min(tail.len()).min(searchable_end);
    for ngram in (min_ngram..=hi).rev() {
        let needle = &tail[tail.len() - ngram..];
        let first = needle[0];
        let max_start = searchable_end - ngram;
        for start in (0..=max_start).rev() {
            if history[start] == first && &history[start..start + ngram] == needle {
                return Some(history[start + ngram..start + ngram + take].to_vec());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_rejection_all_accepted() {
        let drafts = vec![10, 20, 30];
        let target_ids = vec![10, 20, 30, 40]; // 4 target IDs for 3 drafts + 1 bonus

        let v = RejectionSampler::greedy(&drafts, &target_ids).unwrap();
        assert_eq!(v.accepted_drafts(), 3);
        assert_eq!(v.processed_rows(), 4);
        assert_eq!(v.emitted_tokens(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_greedy_rejection_partial_accepted() {
        let drafts = vec![10, 20, 30];
        let target_ids = vec![10, 99, 30, 40]; // mismatch at index 1

        let v = RejectionSampler::greedy(&drafts, &target_ids).unwrap();
        assert_eq!(v.accepted_drafts(), 1);
        assert_eq!(v.processed_rows(), 2);
        assert_eq!(v.emitted_tokens(), &[10, 99]);
    }

    #[test]
    fn test_greedy_rejection_first_mismatch() {
        let drafts = vec![10, 20, 30];
        let target_ids = vec![99, 20, 30, 40]; // immediate mismatch at index 0

        let v = RejectionSampler::greedy(&drafts, &target_ids).unwrap();
        assert_eq!(v.accepted_drafts(), 0);
        assert_eq!(v.processed_rows(), 1);
        assert_eq!(v.emitted_tokens(), &[99]);
    }

    #[test]
    fn test_greedy_rejection_empty_drafts() {
        let drafts = vec![];
        let target_ids = vec![42]; // only the bonus token

        let v = RejectionSampler::greedy(&drafts, &target_ids).unwrap();
        assert_eq!(v.accepted_drafts(), 0);
        assert_eq!(v.processed_rows(), 1);
        assert_eq!(v.emitted_tokens(), &[42]);
    }

    #[test]
    fn test_greedy_rejection_geometry_validation() {
        // Mismatched length (must be drafts.len() + 1)
        assert!(RejectionSampler::greedy(&[10, 20], &[10, 20]).is_err());
        assert!(RejectionSampler::greedy(&[10, 20], &[10, 20, 30, 40]).is_err());

        // Negative token IDs
        assert!(RejectionSampler::greedy(&[-1, 20], &[10, 20, 30]).is_err());
        assert!(RejectionSampler::greedy(&[10, 20], &[10, -1, 30]).is_err());
    }

    #[test]
    fn test_stochastic_rejection_identical_distributions() {
        let p = vec![0.1f32, 0.7f32, 0.1f32, 0.1f32];
        let drafts = vec![1]; // draft predicted token 1 (highest prob)
        let target_probs = vec![p.clone(), p.clone()];
        let draft_probs = vec![p.clone()];
        let accept_uniforms = vec![0.2f64];
        let recovery_uniform = 0.5f64;

        let v = RejectionSampler::stochastic(
            &drafts,
            &target_probs,
            &draft_probs,
            &accept_uniforms,
            recovery_uniform,
        )
        .unwrap();
        // Since target == draft, p(1)/q(1) = 1.0, uniform 0.2 <= 1.0, accepted!
        assert_eq!(v.accepted_drafts(), 1);
        assert_eq!(v.emitted_tokens().len(), 2); // 1 accepted draft + 1 bonus token
        assert_eq!(v.emitted_tokens()[0], 1);
    }

    #[test]
    fn test_lookup_suffix_continuation_prefers_longer_and_recent_matches() {
        let history = vec![
            1, 10, 20, 30, 40, 100, 101, 102, // older 5-gram [1, 10, 20, 30, 40]
            2, 10, 20, 30, 40, 200, 201, 202, // newer 4-gram [10, 20, 30, 40]
            9, 8, 7, 10, 20, 30,
        ];
        // Tail ends with [1, 10, 20, 30, 40]: 5-gram match beats newer 4-gram match.
        assert_eq!(
            lookup_suffix_continuation(&history, &[1, 10, 20, 30, 40], 4, 6, 3),
            Some(vec![100, 101, 102])
        );
        // Tail ends with [99, 10, 20, 30, 40]: falls back to 4-gram and picks most recent.
        assert_eq!(
            lookup_suffix_continuation(&history, &[99, 10, 20, 30, 40], 4, 6, 2),
            Some(vec![200, 201])
        );
        assert_eq!(
            lookup_suffix_continuation(&history, &[55, 66, 77, 88], 4, 6, 2),
            None
        );
    }
}

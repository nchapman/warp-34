//! Host-side transaction boundary for collective-bearing multi-rank graph replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplayPhase {
    Upload,
    Capture,
    Launch,
}

#[derive(Debug)]
pub(crate) struct ReplayError<E> {
    pub source: E,
    /// At least one peer may already be waiting in a device collective. Returning
    /// to normal inference or freeing its buffers is unsafe in this state.
    pub partial_launch: bool,
}

pub(crate) fn replay_ranks<E>(
    ranks: usize,
    mut step: impl FnMut(ReplayPhase, usize) -> Result<(), E>,
) -> Result<(), ReplayError<E>> {
    for phase in [
        ReplayPhase::Upload,
        ReplayPhase::Capture,
        ReplayPhase::Launch,
    ] {
        for rank in 0..ranks {
            if let Err(source) = step(phase, rank) {
                return Err(ReplayError {
                    source,
                    partial_launch: phase == ReplayPhase::Launch && rank > 0,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_rank_launches_if_either_rank_fails_upload_or_capture() {
        for failure in [
            (ReplayPhase::Upload, 0),
            (ReplayPhase::Upload, 1),
            (ReplayPhase::Capture, 0),
            (ReplayPhase::Capture, 1),
        ] {
            let mut calls = Vec::new();
            let error = replay_ranks(2, |phase, rank| {
                calls.push((phase, rank));
                if (phase, rank) == failure {
                    Err("injected failure")
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
            assert_eq!(error.source, "injected failure");
            assert!(!error.partial_launch);
            assert!(!calls.iter().any(|(phase, _)| *phase == ReplayPhase::Launch));
        }
    }
    #[test]
    fn launch_failures_distinguish_safe_return_from_fatal_partial_replay() {
        for fail_rank in 0..2 {
            let mut calls = Vec::new();
            let error = replay_ranks(2, |phase, rank| {
                calls.push((phase, rank));
                if phase == ReplayPhase::Launch && rank == fail_rank {
                    Err(())
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
            assert_eq!(error.partial_launch, fail_rank == 1);
            assert_eq!(
                &calls[..4],
                &[
                    (ReplayPhase::Upload, 0),
                    (ReplayPhase::Upload, 1),
                    (ReplayPhase::Capture, 0),
                    (ReplayPhase::Capture, 1)
                ]
            );
        }
    }
}

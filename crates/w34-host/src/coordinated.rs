//! Host-side transaction boundary for collective-bearing multi-rank graph replay
//! and persistent eager multi-rank worker dispatch.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering},
};
use std::thread::JoinHandle;

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

struct ErasedCall {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

struct SharedPoolState {
    epoch: AtomicU32,
    sleeping: AtomicU32,
    stop: AtomicBool,
    peer_pending: Vec<AtomicU32>,
    remaining: AtomicUsize,
    job: AtomicPtr<ErasedCall>,
    done_lock: Mutex<()>,
    done_cvar: Condvar,
}

impl SharedPoolState {
    fn new(num_peers: usize) -> Self {
        Self {
            epoch: AtomicU32::new(0),
            sleeping: AtomicU32::new(0),
            stop: AtomicBool::new(false),
            peer_pending: (0..num_peers).map(|_| AtomicU32::new(0)).collect(),
            remaining: AtomicUsize::new(0),
            job: AtomicPtr::new(core::ptr::null_mut()),
            done_lock: Mutex::new(()),
            done_cvar: Condvar::new(),
        }
    }

    fn notify_all(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        if self.sleeping.load(Ordering::SeqCst) == 0 {
            return;
        }
        #[cfg(target_os = "linux")]
        // SAFETY: `self.epoch` is a valid 4-byte-aligned `AtomicU32` that outlives all peer threads.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                self.epoch.as_ptr(),
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                i32::MAX,
            );
        }
    }

    fn wait_epoch(&self, seen: u32) {
        for _ in 0..(1 << 14) {
            if self.epoch.load(Ordering::Acquire) != seen {
                return;
            }
            std::hint::spin_loop();
        }
        self.sleeping.fetch_add(1, Ordering::SeqCst);
        if self.epoch.load(Ordering::SeqCst) == seen {
            #[cfg(target_os = "linux")]
            // SAFETY: futex atomically compares `epoch == seen` before sleeping; any concurrent
            // `notify_all` increments `epoch` before checking `sleeping`, preventing lost wakes.
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    self.epoch.as_ptr(),
                    libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                    seen,
                    core::ptr::null::<libc::timespec>(),
                );
            }
            #[cfg(not(target_os = "linux"))]
            std::thread::sleep(std::time::Duration::from_micros(25));
        }
        self.sleeping.fetch_sub(1, Ordering::SeqCst);
    }

    fn wait_remaining_zero(&self) {
        for _ in 0..(1 << 14) {
            if self.remaining.load(Ordering::Acquire) == 0 {
                self.job.store(core::ptr::null_mut(), Ordering::Release);
                return;
            }
            std::hint::spin_loop();
        }
        let guard = self.done_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = self
            .done_cvar
            .wait_while(guard, |_| self.remaining.load(Ordering::Acquire) != 0)
            .unwrap_or_else(|e| e.into_inner());
        self.job.store(core::ptr::null_mut(), Ordering::Release);
    }
}

struct PeerWaitGuard<'a> {
    state: &'a SharedPoolState,
    completed: bool,
}

impl PeerWaitGuard<'_> {
    fn wait(&mut self) {
        if !self.completed {
            self.state.wait_remaining_zero();
            self.completed = true;
        }
    }
}

impl Drop for PeerWaitGuard<'_> {
    fn drop(&mut self) {
        self.wait();
    }
}

/// Persistent peer-rank worker pool (`tp_size - 1` background threads; rank 0 runs inline
/// on the calling thread) eliminating per-step OS thread spawn/join overhead on eager TP paths.
pub struct TpWorkerPool {
    state: Arc<SharedPoolState>,
    threads: Vec<JoinHandle<()>>,
    caller_lock: Mutex<()>,
}

impl core::fmt::Debug for TpWorkerPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TpWorkerPool")
            .field("num_peers", &self.threads.len())
            .finish()
    }
}

impl TpWorkerPool {
    pub fn new(num_peers: usize) -> std::io::Result<Self> {
        let state = Arc::new(SharedPoolState::new(num_peers));
        let mut pool = Self {
            state,
            threads: Vec::with_capacity(num_peers),
            caller_lock: Mutex::new(()),
        };
        for peer_idx in 0..num_peers {
            let st = pool.state.clone();
            let handle = std::thread::Builder::new()
                .name(format!("w34-tp-peer-{}", peer_idx + 1))
                .spawn(move || {
                    let mut seen = 0u32;
                    loop {
                        if st.peer_pending[peer_idx].swap(0, Ordering::AcqRel) == 1 {
                            let job_ptr = st.job.load(Ordering::Acquire);
                            if !job_ptr.is_null() {
                                // SAFETY: `job_ptr` points to the caller's stack `ErasedCall`,
                                // kept alive by `PeerWaitGuard` until `remaining` reaches 0.
                                unsafe {
                                    let job = &*job_ptr;
                                    (job.call)(job.data, peer_idx);
                                }
                            }
                            if st.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                                let _g = st.done_lock.lock().unwrap_or_else(|e| e.into_inner());
                                st.done_cvar.notify_all();
                            }
                            continue;
                        }
                        if st.stop.load(Ordering::Acquire) {
                            break;
                        }
                        let cur = st.epoch.load(Ordering::Acquire);
                        if cur != seen {
                            seen = cur;
                            continue;
                        }
                        st.wait_epoch(seen);
                        seen = st.epoch.load(Ordering::Acquire);
                    }
                })?;
            pool.threads.push(handle);
        }
        Ok(pool)
    }

    #[cfg(test)]
    #[inline]
    pub fn num_peers(&self) -> usize {
        self.threads.len()
    }

    /// Splits `items` into `(&mut items[0], &mut items[1..])`, running `f0(&mut items[0])` inline
    /// on the calling thread while running `fp(peer_idx, &mut items[peer_idx + 1])` concurrently
    /// on persistent peer worker threads.
    pub fn run_split<T, R0, RP, F0, FP>(&self, items: &mut [T], f0: F0, fp: FP) -> (R0, Vec<RP>)
    where
        T: Send,
        R0: Send,
        RP: Send,
        F0: FnOnce(&mut T) -> R0,
        FP: Fn(usize, &mut T) -> RP + Sync,
    {
        let (first, peers) = items
            .split_first_mut()
            .expect("TpWorkerPool::run_split requires non-empty items");
        let active_peers = peers.len();
        if active_peers == 0 {
            return (f0(first), Vec::new());
        }
        assert!(
            active_peers <= self.threads.len(),
            "TpWorkerPool has {} peers, need {active_peers}",
            self.threads.len()
        );
        let _lock = self.caller_lock.lock().unwrap_or_else(|e| e.into_inner());

        struct SplitCtx<'a, T, RP, FP> {
            peers_base: *mut T,
            fp: &'a FP,
            slots_base: *mut Option<std::thread::Result<RP>>,
        }
        // SAFETY: each peer `p in 0..active_peers` accesses exclusively `peers_base.add(p)` and
        // `slots_base.add(p)`, `T: Send`, `RP: Send`, and `FP: Sync`.
        unsafe impl<T: Send, RP: Send, FP: Sync> Sync for SplitCtx<'_, T, RP, FP> {}

        unsafe fn invoke_split<T, RP, FP>(data: *const (), peer_idx: usize)
        where
            T: Send,
            RP: Send,
            FP: Fn(usize, &mut T) -> RP + Sync,
        {
            // SAFETY: `data` points to `SplitCtx` on the caller stack; `peer_idx < active_peers`
            // owns disjoint elements `peers_base.add(peer_idx)` and `slots_base.add(peer_idx)`.
            unsafe {
                let ctx = &*data.cast::<SplitCtx<'_, T, RP, FP>>();
                let item = &mut *ctx.peers_base.add(peer_idx);
                let res = catch_unwind(AssertUnwindSafe(|| (ctx.fp)(peer_idx, item)));
                *ctx.slots_base.add(peer_idx) = Some(res);
            }
        }

        let mut peer_slots: Vec<Option<std::thread::Result<RP>>> =
            (0..active_peers).map(|_| None).collect();
        let ctx = SplitCtx {
            peers_base: peers.as_mut_ptr(),
            fp: &fp,
            slots_base: peer_slots.as_mut_ptr(),
        };
        let erased = ErasedCall {
            data: (&raw const ctx).cast::<()>(),
            call: invoke_split::<T, RP, FP>,
        };

        self.state
            .job
            .store((&raw const erased).cast_mut(), Ordering::Release);
        self.state.remaining.store(active_peers, Ordering::Release);
        let mut wait_guard = PeerWaitGuard {
            state: &self.state,
            completed: false,
        };
        for p in 0..active_peers {
            self.state.peer_pending[p].store(1, Ordering::Release);
        }
        self.state.notify_all();

        let res0 = catch_unwind(AssertUnwindSafe(|| f0(first)));
        wait_guard.wait();

        let mut out_peers = Vec::with_capacity(active_peers);
        let mut first_panic: Option<Box<dyn Any + Send + 'static>> = None;
        let r0_val = match res0 {
            Ok(v) => Some(v),
            Err(p) => {
                first_panic = Some(p);
                None
            }
        };
        for slot in peer_slots {
            match slot.expect("peer worker finished without writing result") {
                Ok(v) => out_peers.push(v),
                Err(p) => {
                    if first_panic.is_none() {
                        first_panic = Some(p);
                    }
                }
            }
        }
        if let Some(p) = first_panic {
            resume_unwind(p);
        }
        (r0_val.unwrap(), out_peers)
    }

    /// Executes an iterator of per-rank closures (`len >= 1`), running closure `0` inline on
    /// the calling thread and closures `1..len` on persistent peer worker threads.
    pub fn run_closures<F, R>(&self, closures: impl IntoIterator<Item = F>) -> Vec<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let mut fns: Vec<Option<F>> = closures.into_iter().map(Some).collect();
        if fns.is_empty() {
            return Vec::new();
        }
        if fns.len() == 1 {
            return vec![(fns[0].take().unwrap())()];
        }
        let (r0, r_peers) = self.run_split(
            fns.as_mut_slice(),
            |slot0| (slot0.take().expect("rank 0 closure missing"))(),
            |_, slot_p| (slot_p.take().expect("peer closure missing"))(),
        );
        let mut out = Vec::with_capacity(1 + r_peers.len());
        out.push(r0);
        out.extend(r_peers);
        out
    }
}

pub(crate) fn run_tp_closures<F, R>(
    pool: Option<&TpWorkerPool>,
    closures: impl IntoIterator<Item = F>,
) -> Vec<R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if let Some(p) = pool {
        return p.run_closures(closures);
    }
    let mut iter = closures.into_iter();
    let Some(f0) = iter.next() else {
        return Vec::new();
    };
    let rest: Vec<F> = iter.collect();
    if rest.is_empty() {
        return vec![f0()];
    }
    std::thread::scope(|sc| {
        let handles: Vec<_> = rest.into_iter().map(|f| sc.spawn(f)).collect();
        let r0 = f0();
        let mut out = Vec::with_capacity(1 + handles.len());
        out.push(r0);
        for h in handles {
            out.push(h.join().expect("TP rank worker panicked"));
        }
        out
    })
}

impl Drop for TpWorkerPool {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Release);
        self.state.notify_all();
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
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

    #[test]
    fn tp_worker_pool_reuses_persistent_peers_and_contains_panics() {
        let pool = TpWorkerPool::new(3).unwrap();
        let mut ranks = [10i32, 20, 30, 40];
        // Partial peer count (`TP=2` on a 3-peer pool) followed by full `TP=4` dispatch.
        for step in 0..8 {
            let (r0, rp) = pool.run_split(
                &mut ranks[..2],
                |x| {
                    *x += 1;
                    (*x, std::thread::current().id())
                },
                |p, x| {
                    *x += (p as i32) + 2;
                    (*x, std::thread::current().id())
                },
            );
            assert_eq!(r0.0, 10 + step + 1);
            assert_eq!(rp.len(), 1);
            assert_ne!(r0.1, rp[0].1);

            assert_eq!(pool.num_peers(), 3);
            let out = pool.run_closures((0..4).map(|r| move || r * 100 + step));
            assert_eq!(out, vec![step, 100 + step, 200 + step, 300 + step]);
        }

        // Rank-0 panic waits for peer completion before unwinding, leaving pool usable.
        let mut vals = [1i32, 2];
        let caught = catch_unwind(AssertUnwindSafe(|| {
            pool.run_split(
                &mut vals,
                |_| -> i32 { panic!("rank 0 boom") },
                |_, v| {
                    *v += 99;
                    *v
                },
            )
        }));
        assert!(caught.is_err());
        assert_eq!(vals[1], 101);

        // Peer panic propagates cleanly and pool stays healthy for subsequent dispatches.
        let caught_peer = catch_unwind(AssertUnwindSafe(|| {
            pool.run_split(
                &mut vals,
                |v| {
                    *v += 5;
                    *v
                },
                |_, _| -> i32 { panic!("peer boom") },
            )
        }));
        assert!(caught_peer.is_err());
        let (ok0, okp) = pool.run_split(&mut vals, |v| *v, |_, v| *v);
        assert_eq!((ok0, okp), (6, vec![101]));
    }
}

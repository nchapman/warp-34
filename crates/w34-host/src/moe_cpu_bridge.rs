mod eval;
// Owned CPU responders. Mailboxes are retained until GPU synchronization and joined shutdown.
use crate::cuda::{CudaDevice, MappedHostBuffer};
use crate::moe_offload::HostExpertStore;
pub(crate) use eval::{has_avx512, supported};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU32, AtomicUsize, Ordering},
};
use std::thread::JoinHandle;

pub(crate) const MAX_DECODE_TOKENS: usize =
    crate::moe_offload::TIERED_MOE_CPU_BRIDGE_MAX_DECODE_TOKENS;
pub(crate) const MAX_TOP_K: usize = crate::moe_offload::TIERED_MOE_CPU_BRIDGE_MAX_TOP_K;
pub(crate) const MAX_PAIRS: usize = MAX_DECODE_TOKENS * MAX_TOP_K;
pub(crate) const MAX_HIDDEN: usize = crate::moe_offload::TIERED_MOE_CPU_BRIDGE_MAX_HIDDEN;
pub(crate) const MAX_ELEMENTS: usize = MAX_DECODE_TOKENS * MAX_HIDDEN;
const _: () = assert!(eval::MAX_SLICE_TOKENS == MAX_DECODE_TOKENS);

#[repr(C, align(64))]
struct Request {
    seq: AtomicI64,
    tokens: i32,
    top_k: i32,
    _reserved0: i32,
    pairs: i32,
    hidden: i32,
    intermediate: i32,
    experts: i32,
    pad: [i32; 7],
    ids: [i32; MAX_PAIRS],
    weights: [f32; MAX_PAIRS],
    x: [u16; MAX_ELEMENTS],
}
#[repr(C, align(64))]
struct Response {
    seq: AtomicI64,
    status: i32,
    pad: [i32; 13],
    y: [f32; MAX_ELEMENTS],
}
fn bridge_error(detail: impl ToString) -> crate::cuda::CudaError {
    crate::cuda::CudaError::with_detail(-1, "CPU MoE bridge", detail.to_string())
}
#[derive(Default)]
struct Scratch {
    xp: Vec<f32>,
    xg: Vec<f32>,
    weights: Vec<f32>,
    eval: Vec<f32>,
}
#[derive(Default)]
struct Job {
    expert: usize,
    rows: Vec<(usize, f32)>,
}
// Workers contend on the claim counter, not the immutable job descriptors or
// the completion counter polled by the responder. Keep these on separate lines.
#[repr(align(64))]
struct Counter(AtomicUsize);
impl Counter {
    fn new(value: usize) -> Self {
        Self(AtomicUsize::new(value))
    }
}
impl std::ops::Deref for Counter {
    type Target = AtomicUsize;
    fn deref(&self) -> &AtomicUsize {
        &self.0
    }
}
struct Batch {
    store: HostExpertStore,
    xp: Vec<f32>,
    xg: Vec<f32>,
    jobs: Vec<Job>,
    outputs: Vec<Mutex<Vec<f32>>>,
    slices: usize,
    tokens: usize,
    next: Counter,
    failed: AtomicBool,
    completion: Arc<Completion>,
}
struct Completion {
    remaining: Counter,
    lock: Mutex<()>,
    done: Condvar,
}
impl Completion {
    fn new(n: usize) -> Arc<Self> {
        Arc::new(Self {
            remaining: Counter::new(n),
            lock: Mutex::new(()),
            done: Condvar::new(),
        })
    }
}
// Linux futex epoch: no queue nodes, per-job closures, or channel sends.
#[derive(Default)]
struct Wake {
    epoch: AtomicU32,
    sleeping: AtomicU32,
    batch: AtomicPtr<Batch>,
    stop: AtomicBool,
}
impl Wake {
    fn notify_all(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        if self.sleeping.load(Ordering::SeqCst) == 0 {
            return;
        }
        #[cfg(target_os = "linux")]
        // SAFETY: aligned AtomicU32 remains alive through all worker joins.
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                self.epoch.as_ptr(),
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                i32::MAX,
            );
        }
    }
    fn wait(&self, seen: u32) {
        // Spin across the ~8-10ms GPU lm_head + MTP-3 draft window between Target
        // Verify rounds so worker cores stay out of kernel futex / C-states during
        // active generation, then sleep at 0% CPU once idle.
        for _ in 0..(1 << 18) {
            if self.epoch.load(Ordering::Acquire) != seen {
                return;
            }
            std::hint::spin_loop();
        }
        self.sleeping.fetch_add(1, Ordering::SeqCst);
        if self.epoch.load(Ordering::SeqCst) == seen {
            #[cfg(target_os = "linux")]
            // SAFETY: futex checks the epoch atomically before sleeping; a concurrent
            // publication yields EAGAIN, so a wake cannot be lost. Spurious/EINTR
            // returns are handled by the caller's epoch loop.
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    self.epoch.as_ptr(),
                    libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                    seen,
                    std::ptr::null::<libc::timespec>(),
                );
            }
            #[cfg(not(target_os = "linux"))]
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        self.sleeping.fetch_sub(1, Ordering::SeqCst);
    }
}
#[derive(Default)]
struct Dispatch {
    batch: Option<Arc<Batch>>,
}
#[derive(Default)]
struct RoutingScratch {
    tok_to_compact: Vec<u32>,
    active_toks: Vec<usize>,
    token_routes: Vec<Vec<usize>>,
    expert_to_job: Vec<u32>,
}
struct Pool {
    state: Arc<(Mutex<Dispatch>, Wake)>,
    cpus: Vec<usize>,
    threads: Vec<JoinHandle<()>>,
    // Serializes callers, but never grants workers borrowed caller buffers.
    submit: Mutex<()>,
    spare: Mutex<Vec<Job>>,
    spare_outputs: Mutex<Vec<Mutex<Vec<f32>>>>,
    spare_inputs: Mutex<(Vec<f32>, Vec<f32>)>,
    spare_routing: Mutex<RoutingScratch>,
    caller_scratch: Mutex<Scratch>,
}
impl Pool {
    fn new() -> crate::cuda::Result<Self> {
        Self::new_with_cpus(detect_worker_cpus())
    }
    fn new_with_cpus(cpus: Vec<usize>) -> crate::cuda::Result<Self> {
        let cpus = if cpus.is_empty() { vec![0] } else { cpus };
        let mut pool = Self {
            state: Arc::new((Mutex::new(Dispatch::default()), Wake::default())),
            cpus,
            threads: Vec::new(),
            submit: Mutex::new(()),
            spare: Mutex::new(Vec::new()),
            spare_outputs: Mutex::new(Vec::new()),
            spare_inputs: Mutex::new((Vec::new(), Vec::new())),
            spare_routing: Mutex::new(RoutingScratch::default()),
            caller_scratch: Mutex::new(Scratch::default()),
        };
        // The submitting responder is worker zero on `cpus[0]`. Background workers
        // cover the remaining physical cores (`cpus[1..]`) without SMT oversubscription.
        for (i, &cpu) in pool.cpus.iter().enumerate().skip(1) {
            let state = pool.state.clone();
            // On failure, Pool::drop wakes and joins all already-created workers.
            let thread = std::thread::Builder::new()
                .name(format!("moe-cpu-{i}"))
                .spawn(move || {
                    pin_current_thread(cpu);
                    let mut scratch = Scratch::default();
                    let mut epoch = 0;
                    loop {
                        let wake = &state.1;
                        wake.wait(epoch);
                        if wake.stop.load(Ordering::Acquire) {
                            return;
                        }
                        let current = wake.epoch.load(Ordering::Acquire);
                        if current == epoch {
                            continue; // EINTR or a spurious futex wake.
                        }
                        epoch = current;
                        let completion = {
                            // SAFETY: publication stores an owned Arc in Dispatch
                            // before releasing the epoch. Every worker observes each
                            // epoch and holds off replacement via `remaining` until
                            // ALL its Batch references end below. Timed-out batches
                            // stay owned; Drop joins workers before releasing Dispatch.
                            let batch = unsafe { &*wake.batch.load(Ordering::Acquire) };
                            work_batch(batch, &mut scratch, None);
                            batch.completion.clone()
                        };
                        if completion.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                            let _guard = completion.lock.lock().unwrap();
                            completion.done.notify_all();
                        }
                    }
                })
                .map_err(bridge_error)?;
            pool.threads.push(thread);
        }
        Ok(pool)
    }
    fn run(&self, batch: &Arc<Batch>, deadline: std::time::Instant) -> crate::cuda::Result<()> {
        let (lock, wake) = &*self.state;
        {
            let mut state = lock.lock().unwrap();
            if state
                .batch
                .as_ref()
                .is_some_and(|previous| previous.completion.remaining.load(Ordering::Acquire) != 0)
            {
                return Err(bridge_error("previous batch still running"));
            }
            state.batch = Some(batch.clone());
            wake.batch
                .store(Arc::as_ptr(batch).cast_mut(), Ordering::Release);
            wake.notify_all();
        }
        work_batch(
            batch,
            &mut self.caller_scratch.lock().unwrap(),
            Some(deadline),
        );
        for _ in 0..(1 << 16) {
            if batch.completion.remaining.load(Ordering::Acquire) == 0 {
                break;
            }
            std::hint::spin_loop();
        }
        let mut remaining = batch.completion.lock.lock().unwrap();
        while batch.completion.remaining.load(Ordering::Acquire) != 0 {
            let now = std::time::Instant::now();
            if now >= deadline {
                return Err(bridge_error(
                    "30s worker deadline exceeded; batch buffers retained",
                ));
            }
            remaining = batch
                .completion
                .done
                .wait_timeout(remaining, deadline - now)
                .unwrap()
                .0;
        }
        drop(remaining);
        self.state
            .1
            .batch
            .store(std::ptr::null_mut(), Ordering::Release);
        self.state.0.lock().unwrap().batch = None;
        if batch.failed.load(Ordering::Relaxed) {
            return Err(bridge_error("expert worker failed"));
        }
        Ok(())
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        let wake = &self.state.1;
        wake.stop.store(true, Ordering::Release);
        wake.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
pub(crate) struct Mailbox {
    req: MappedHostBuffer,
    resp: MappedHostBuffer,
    store: HostExpertStore,
}
impl Mailbox {
    pub fn request_ptr(&self) -> *mut core::ffi::c_void {
        self.req.dev_ptr
    }
    pub fn response_ptr(&self) -> *const core::ffi::c_void {
        self.resp.dev_ptr
    }
}
pub(crate) struct Runtime {
    pool: Arc<Pool>,
    layers: Arc<Mutex<Vec<Arc<Mailbox>>>>,
    layer_count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    prefill_active: Arc<AtomicBool>,
    prefill_park: Arc<(Mutex<()>, Condvar)>,
    server: Option<JoinHandle<()>>,
    device: i32,
}
impl Runtime {
    pub fn new(stream: &crate::cuda::CudaStream) -> crate::cuda::Result<Arc<Self>> {
        let device = stream.device_id;
        let failure = stream.host_failures.register();
        let pool = Arc::new(Pool::new()?);
        let layers = Arc::new(Mutex::new(Vec::<Arc<Mailbox>>::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let prefill_active = Arc::new(AtomicBool::new(false));
        let prefill_park = Arc::new((Mutex::new(()), Condvar::new()));
        let layer_count = Arc::new(AtomicUsize::new(0));
        let count = layer_count.clone();
        let (p, l, s, pa, pp) = (
            pool.clone(),
            layers.clone(),
            stop.clone(),
            prefill_active.clone(),
            prefill_park.clone(),
        );
        let server = std::thread::Builder::new()
            .name("moe-responder".into())
            .spawn(move || {
                pin_current_thread(p.cpus[0]);
                let mut mailboxes = Vec::<Arc<Mailbox>>::new();
                let mut seen = Vec::new();
                while !s.load(Ordering::Acquire) {
                    if pa.load(Ordering::Acquire) {
                        let mut guard = pp.0.lock().unwrap();
                        while pa.load(Ordering::Acquire) && !s.load(Ordering::Acquire) {
                            guard = pp.1.wait(guard).unwrap();
                        }
                        continue;
                    }
                    // Registration is append-only. Once loaded, polling needs no
                    // registry mutex or response-page reads: this is its sole writer.
                    if count.load(Ordering::Acquire) != mailboxes.len() {
                        mailboxes.clone_from(&*l.lock().unwrap());
                        seen.resize(mailboxes.len(), 0);
                    }
                    for (index, m) in mailboxes.iter().enumerate() {
                        // SAFETY: stable pinned allocations; release/acquire sequence transfers exclusive
                        // payload ownership between the single GPU stream and this responder.
                        unsafe {
                            let rq = m.req.host_ptr.cast::<Request>();
                            let rs = m.resp.host_ptr.cast::<Response>();
                            let seq = (*rq).seq.load(Ordering::Acquire);
                            if seq == seen[index] {
                                continue;
                            }
                            seen[index] = seq;
                            // Once a graph has used a failed expert result, its KV/state
                            // is no longer trustworthy. Drain subsequent mailboxes, but
                            // keep the owning stream poisoned until the model is reloaded.
                            if failure.get().is_some() {
                                (*rs).status = 1;
                                (*rs).seq.store(seq, Ordering::Release);
                                continue;
                            }
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let r = &*rq;
                                    assert!(
                                        (1..=MAX_DECODE_TOKENS as i32).contains(&r.tokens)
                                            && (1..=MAX_TOP_K as i32).contains(&r.top_k)
                                    );
                                    let h = m.store.layout().hidden;
                                    assert_eq!(r.hidden as usize, h);
                                    let total = r.tokens as usize * r.top_k as usize;
                                    let mut tasks = [(0usize, 0usize, 0.0f32); MAX_PAIRS];
                                    let mut n_tasks = 0;
                                    for i in 0..total {
                                        if r.ids[i] >= 0 {
                                            tasks[n_tasks] = (
                                                i / r.top_k as usize,
                                                r.ids[i] as usize,
                                                r.weights[i],
                                            );
                                            n_tasks += 1;
                                        }
                                    }
                                    let elems = r.tokens as usize * h;
                                    evaluate_into(
                                        &p,
                                        &m.store,
                                        Input::F16(&r.x[..elems]),
                                        r.tokens as usize,
                                        &tasks[..n_tasks],
                                        Output::F32(&mut (&mut (*rs).y)[..elems]),
                                    )
                                }));
                            match result {
                                Ok(Ok(())) => {
                                    (*rs).status = 0;
                                }
                                error => {
                                    let _ = failure.set(bridge_error(format!(
                                        "responder failed: {error:?}; reload model before reuse"
                                    )));
                                    (*rs).status = 1;
                                }
                            }
                            (*rs).seq.store(seq, Ordering::Release);
                        }
                    }
                    std::hint::spin_loop();
                }
            })
            .map_err(bridge_error)?;
        Ok(Arc::new(Self {
            pool,
            layers,
            layer_count,
            stop,
            prefill_active,
            prefill_park,
            server: Some(server),
            device,
        }))
    }
    /// Creates a prefill-only worker pool for `EP >= 2` rank `ep_rank` without spawning
    /// a spinning decode responder thread or allocating mapped decode mailboxes.
    /// Physical worker cores are partitioned disjointly across `ep_size` ranks so
    /// concurrent `TP=2 / EP=2` prefill ranks never contend on the same CPU cores.
    pub fn new_prefill_only(
        device: i32,
        ep_rank: usize,
        ep_size: usize,
    ) -> crate::cuda::Result<Arc<Self>> {
        let all_cpus = detect_worker_cpus();
        let ep_size = ep_size.max(1);
        let ep_rank = ep_rank.min(ep_size - 1);
        let rank_cpus = if all_cpus.len() >= ep_size {
            let base = all_cpus.len() / ep_size;
            let rem = all_cpus.len() % ep_size;
            let start = ep_rank * base + ep_rank.min(rem);
            let len = base + usize::from(ep_rank < rem);
            all_cpus[start..start + len].to_vec()
        } else {
            vec![all_cpus[ep_rank % all_cpus.len()]]
        };
        let pool = Arc::new(Pool::new_with_cpus(rank_cpus)?);
        Ok(Arc::new(Self {
            pool,
            layers: Arc::new(Mutex::new(Vec::new())),
            layer_count: Arc::new(AtomicUsize::new(0)),
            stop: Arc::new(AtomicBool::new(false)),
            prefill_active: Arc::new(AtomicBool::new(false)),
            prefill_park: Arc::new((Mutex::new(()), Condvar::new())),
            server: None,
            device,
        }))
    }
    pub fn register(&self, store: &HostExpertStore) -> crate::cuda::Result<Arc<Mailbox>> {
        let layout = store.layout();
        if layout.hidden > 4096
            || layout.intermediate > 2048
            || !matches!(layout.group_size, 64 | 128)
            || !matches!(layout.down_hadamard_block_size, 0 | 64)
        {
            return Err(crate::cuda::CudaError::with_detail(
                -1,
                "CPU bridge geometry",
                "requires hidden <= 4096, intermediate <= 2048, group size 64 or 128, down Hadamard 0 or 64",
            ));
        }
        let req = MappedHostBuffer::alloc(std::mem::size_of::<Request>())?;
        let resp = MappedHostBuffer::alloc(std::mem::size_of::<Response>())?;
        // SAFETY: fresh allocations, before publication to CPU or GPU.
        unsafe {
            std::ptr::write_bytes(req.host_ptr, 0, req.len_bytes);
            std::ptr::write_bytes(resp.host_ptr, 0, resp.len_bytes);
        }
        let m = Arc::new(Mailbox {
            req,
            resp,
            store: store.clone(),
        });
        let mut layers = self.layers.lock().unwrap();
        layers.push(m.clone());
        self.layer_count.store(layers.len(), Ordering::Release);
        Ok(m)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        store: &HostExpertStore,
        x: &[u16],
        tokens: usize,
        experts: &[i32],
        counts: &[i32],
        toks: &[i32],
        weights: &[f32],
        out: &mut [u16],
    ) -> crate::cuda::Result<()> {
        let h = store.layout().hidden;
        if x.len() != tokens.saturating_mul(h)
            || out.len() != x.len()
            || experts.len() != counts.len()
            || experts
                .len()
                .checked_mul(96)
                .is_none_or(|n| toks.len() < n || weights.len() < n)
            || counts.iter().any(|&n| !(0..=96).contains(&n))
            || experts
                .iter()
                .any(|&e| e < 0 || e as usize >= store.expert_ids().len())
        {
            return Err(bridge_error("invalid prefill buffers or routes"));
        }
        let tasks = experts
            .iter()
            .enumerate()
            .flat_map(|(j, &e)| {
                (0..counts[j] as usize)
                    .map(move |s| (toks[j * 96 + s] as usize, e as usize, weights[j * 96 + s]))
            })
            .collect::<Vec<_>>();
        let _guard = PrefillGuard::enter(self);
        evaluate_into(
            &self.pool,
            store,
            Input::F16(x),
            tokens,
            &tasks,
            Output::F16(out),
        )
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        // Responders must remain alive while outstanding graph launches finish.
        if CudaDevice::set_current(self.device)
            .and_then(|device| device.synchronize())
            .is_err()
        {
            // Cannot prove GPU has released mapped pointers; never free live mailboxes.
            std::process::abort();
        }
        self.stop.store(true, Ordering::Release);
        {
            let _guard = self.prefill_park.0.lock().unwrap();
            self.prefill_park.1.notify_all();
        }
        if let Some(t) = self.server.take() {
            let _ = t.join();
        }
    }
}
struct PrefillGuard<'a> {
    runtime: &'a Runtime,
    #[cfg(target_os = "linux")]
    saved_mask: Option<libc::cpu_set_t>,
}
impl<'a> PrefillGuard<'a> {
    fn enter(runtime: &'a Runtime) -> Self {
        runtime.prefill_active.store(true, Ordering::Release);
        // Wait for any in-flight responder batch on `cpus[0]` to finish before pinning.
        drop(runtime.pool.submit.lock().unwrap());
        #[cfg(target_os = "linux")]
        // SAFETY: `cpu_set_t` is plain old data; `sched_getaffinity(0, ...)` queries the current thread.
        let saved_mask = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
                pin_current_thread(runtime.pool.cpus[0]);
                Some(set)
            } else {
                None
            }
        };
        Self {
            runtime,
            #[cfg(target_os = "linux")]
            saved_mask,
        }
    }
}
impl Drop for PrefillGuard<'_> {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(ref set) = self.saved_mask {
            // SAFETY: `set` was captured from `sched_getaffinity` on this thread in `enter`.
            unsafe {
                let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set);
            }
        }
        let _guard = self.runtime.prefill_park.0.lock().unwrap();
        self.runtime.prefill_active.store(false, Ordering::Release);
        self.runtime.prefill_park.1.notify_all();
    }
}
const MAX_WORKERS: usize = 32;

/// Selects up to `max_workers` OS CPUs, keeping at most one hardware thread per
/// physical `(package_id, core_id)` when topology metadata is available.
fn select_physical_cpus(
    candidates: impl IntoIterator<Item = (usize, Option<(i32, i32)>)>,
    max_workers: usize,
) -> Vec<usize> {
    let max_workers = max_workers.max(1);
    let all = candidates.into_iter().collect::<Vec<_>>();
    if all.is_empty() {
        return vec![0];
    }
    if all.iter().all(|(_, topo)| topo.is_some()) {
        let mut seen_cores = Vec::<(i32, i32)>::new();
        let mut selected = Vec::new();
        for (cpu, topo) in all {
            let key = topo.unwrap();
            if !seen_cores.contains(&key) {
                seen_cores.push(key);
                selected.push(cpu);
                if selected.len() == max_workers {
                    break;
                }
            }
        }
        return selected;
    }
    let count = if all.len() >= 32 {
        (all.len() / 2).min(max_workers)
    } else {
        all.len().min(max_workers)
    };
    all.into_iter().take(count.max(1)).map(|(c, _)| c).collect()
}

fn parse_canonical_core_key(
    pkg: i32,
    die: Option<i32>,
    core: i32,
    thread_siblings_list: Option<&str>,
) -> (i32, i32) {
    if let Some(list) = thread_siblings_list
        && let Some(first) = list
            .trim()
            .split([',', '-'])
            .next()
            .and_then(|s| s.trim().parse::<i32>().ok())
    {
        return (pkg, first);
    }
    let die_bits = die.unwrap_or(0).max(0) & 0x7fff;
    (pkg, (die_bits << 16) ^ (core & 0xffff))
}

fn read_cpu_core_key(cpu: usize) -> Option<(i32, i32)> {
    let base = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
    let pkg = std::fs::read_to_string(format!("{base}/physical_package_id"))
        .ok()?
        .trim()
        .parse::<i32>()
        .ok()?;
    let core = std::fs::read_to_string(format!("{base}/core_id"))
        .ok()?
        .trim()
        .parse::<i32>()
        .ok()?;
    let die = std::fs::read_to_string(format!("{base}/die_id"))
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok());
    let siblings = std::fs::read_to_string(format!("{base}/thread_siblings_list")).ok();
    Some(parse_canonical_core_key(
        pkg,
        die,
        core,
        siblings.as_deref(),
    ))
}

fn detect_worker_cpus() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    // SAFETY: initialized cpu_set_t, valid current-thread affinity query.
    unsafe {
        let mut allowed: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of_val(&allowed), &mut allowed) == 0 {
            let candidates = (0..libc::CPU_SETSIZE as usize)
                .filter(|&c| libc::CPU_ISSET(c, &allowed))
                .map(|c| (c, read_cpu_core_key(c)));
            return select_physical_cpus(candidates, MAX_WORKERS);
        }
    }
    let n = std::thread::available_parallelism().map_or(1, |n| n.get());
    select_physical_cpus((0..n).map(|c| (c, None)), MAX_WORKERS)
}

fn pin_current_thread(cpu: usize) {
    #[cfg(target_os = "linux")]
    // SAFETY: initialized cpu_set_t, valid current-thread affinity call.
    unsafe {
        if cpu < libc::CPU_SETSIZE as usize {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}
fn work_batch(batch: &Batch, scratch: &mut Scratch, deadline: Option<std::time::Instant>) {
    loop {
        // Only the caller has a deadline. Workers retain and drain timed-out
        // batches. A claimed job is bounded CPU arithmetic and always finishes
        // before its scratch is reused; no caller-owned input is borrowed.
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            break;
        }
        let index = batch.next.fetch_add(1, Ordering::Relaxed);
        if index >= batch.jobs.len() * batch.slices {
            break;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job(batch, index, scratch);
        }));
        if result.is_err() {
            batch.failed.store(true, Ordering::Relaxed);
        }
    }
}
fn run_job(batch: &Batch, index: usize, scratch: &mut Scratch) {
    let job = &batch.jobs[index / batch.slices];
    let store = &batch.store;
    let expert = job.expert;
    let layout = store.layout();
    let h = layout.hidden;
    let gs = layout.group_size;
    let groups = layout.intermediate / 128;
    let n = job.rows.len();
    scratch.xp.clear();
    scratch.xg.clear();
    scratch.weights.clear();
    for &(tok, weight) in &job.rows {
        scratch
            .xp
            .extend_from_slice(&batch.xp[tok * h..(tok + 1) * h]);
        scratch
            .xg
            .extend_from_slice(&batch.xg[tok * (h / gs)..(tok + 1) * (h / gs)]);
        scratch.weights.push(weight);
    }
    let mut output = batch.outputs[index].lock().unwrap();
    output.resize(n * h, 0.);
    let pack = store
        .expert_pack(store.expert_ids().start + expert)
        .unwrap();
    // SAFETY: canonical pack regions are aligned to u32/u16, initialized, and
    // immutable for the lifetime of this cloned store (mapped or Vec allocation).
    let d = unsafe {
        let words = |r: std::ops::Range<usize>| {
            assert_eq!((pack.as_ptr().add(r.start) as usize) % 4, 0);
            std::slice::from_raw_parts(pack.as_ptr().add(r.start).cast::<u32>(), r.len() / 4)
        };
        let scales = |r: std::ops::Range<usize>| {
            assert_eq!((pack.as_ptr().add(r.start) as usize) % 2, 0);
            std::slice::from_raw_parts(pack.as_ptr().add(r.start).cast::<u16>(), r.len() / 2)
        };
        eval::ExpertLayout {
            hidden: h,
            intermediate: layout.intermediate,
            group_size: gs,
            down_hadamard: layout.down_hadamard_block_size,
            gate_up_words: words(layout.gate_up_packed.clone()),
            gate_up_scales: scales(layout.gate_up_scales.clone()),
            down_words: words(layout.down_packed.clone()),
            down_scales: scales(layout.down_scales.clone()),
        }
    };
    {
        let slice = index % batch.slices;
        let range = if batch.slices == 2 {
            if slice == 0 {
                0..groups / 2
            } else {
                groups / 2..groups
            }
        } else {
            0..groups
        };
        scratch
            .eval
            .resize(eval::scratch_len(n, range.len()).unwrap(), 0.);
        let y = &mut output[..];
        if batch.tokens <= MAX_DECODE_TOKENS {
            eval::eval_expert_slice(
                d,
                range,
                &scratch.xp,
                &scratch.xg,
                &scratch.weights,
                y,
                &mut scratch.eval,
            )
            .unwrap();
        } else {
            eval::eval_expert_prefill(
                d,
                &scratch.xp,
                &scratch.xg,
                &scratch.weights,
                y,
                &mut scratch.eval,
            )
            .unwrap();
        }
    }
}
enum Input<'a> {
    #[cfg(test)]
    F32(&'a [f32]),
    F16(&'a [u16]),
}
impl Input<'_> {
    fn len(&self) -> usize {
        match self {
            #[cfg(test)]
            Self::F32(x) => x.len(),
            Self::F16(x) => x.len(),
        }
    }
    fn prepare_row(
        &self,
        tok: usize,
        h: usize,
        group_size: usize,
        xp: &mut [f32],
        xg: &mut [f32],
    ) -> Result<(), eval::EvalError> {
        let range = tok * h..(tok + 1) * h;
        match self {
            #[cfg(test)]
            Self::F32(x) => eval::prepare_x_perm_grouped(&x[range], group_size, xp, xg),
            Self::F16(x) => eval::prepare_x_perm_f16_grouped(&x[range], group_size, xp, xg),
        }
    }
}
enum Output<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [u16]),
}
#[cfg(test)]
fn evaluate(
    pool: &Pool,
    store: &HostExpertStore,
    x: &[f32],
    tokens: usize,
    tasks: &[(usize, usize, f32)],
) -> crate::cuda::Result<Vec<f32>> {
    let mut y = vec![0.; tokens * store.layout().hidden];
    evaluate_into(
        pool,
        store,
        Input::F32(x),
        tokens,
        tasks,
        Output::F32(&mut y),
    )?;
    Ok(y)
}
fn evaluate_into(
    pool: &Pool,
    store: &HostExpertStore,
    x: Input<'_>,
    tokens: usize,
    tasks: &[(usize, usize, f32)],
    out: Output<'_>,
) -> crate::cuda::Result<()> {
    let h = store.layout().hidden;
    if x.len() != tokens.saturating_mul(h) {
        return Err(bridge_error("invalid input shape"));
    }
    let num_experts = store.expert_ids().len();
    for &(tok, expert, _) in tasks {
        if tok >= tokens || expert >= num_experts {
            return Err(bridge_error("invalid expert route"));
        }
    }
    let lock_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    // A timed-out batch is still owned by the pool/workers. Do not publish another
    // epoch until it finishes: every worker must observe each submitted epoch.
    let _submit = loop {
        if let Ok(guard) = pool.submit.try_lock() {
            break guard;
        }
        if std::time::Instant::now() >= lock_deadline {
            return Err(bridge_error("pool busy for 30s"));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    {
        let state = pool.state.0.lock().unwrap();
        if let Some(previous) = &state.batch
            && previous.completion.remaining.load(Ordering::Acquire) != 0
        {
            return Err(bridge_error("previous timed-out batch still running"));
        }
    }
    let mut jobs = std::mem::take(&mut *pool.spare.lock().unwrap());
    for job in &mut jobs {
        job.rows.clear();
    }
    let mut routing = std::mem::take(&mut *pool.spare_routing.lock().unwrap());
    // Compact only tokens that actually appear in `tasks` (`n_act <= tokens`) so prefill
    // never transposes or reduces non-routed prompt rows.
    routing.tok_to_compact.clear();
    routing.tok_to_compact.resize(tokens, u32::MAX);
    routing.active_toks.clear();
    for routes in &mut routing.token_routes {
        routes.clear();
    }
    routing.expert_to_job.clear();
    routing.expert_to_job.resize(num_experts, u32::MAX);
    let mut used = 0usize;
    for &(tok, expert, weight) in tasks {
        let compact_tok = if routing.tok_to_compact[tok] == u32::MAX {
            let c = routing.active_toks.len();
            routing.tok_to_compact[tok] = c as u32;
            routing.active_toks.push(tok);
            if c == routing.token_routes.len() {
                routing.token_routes.push(Vec::with_capacity(10));
            }
            c
        } else {
            routing.tok_to_compact[tok] as usize
        };
        if !routing.token_routes[compact_tok].contains(&expert) {
            routing.token_routes[compact_tok].push(expert);
        }
        let j = if routing.expert_to_job[expert] == u32::MAX {
            if used == jobs.len() {
                jobs.push(Job::default());
            }
            jobs[used].expert = expert;
            routing.expert_to_job[expert] = used as u32;
            used += 1;
            used - 1
        } else {
            routing.expert_to_job[expert] as usize
        };
        let rows = &mut jobs[j].rows;
        if let Some(row) = rows.iter_mut().find(|r| r.0 == compact_tok) {
            row.1 += weight;
        } else {
            rows.push((compact_tok, weight));
        }
    }
    jobs.truncate(used);
    jobs.sort_by_key(|j| std::cmp::Reverse(j.rows.len()));
    for (j, job) in jobs.iter().enumerate() {
        routing.expert_to_job[job.expert] = j as u32;
    }
    let slices = if tokens <= MAX_DECODE_TOKENS && store.layout().intermediate / 128 >= 2 {
        2
    } else {
        1
    };
    let n_act = routing.active_toks.len();
    let gs = store.layout().group_size;
    let h_groups = h / gs;
    let (mut xp, mut xg) = std::mem::take(&mut *pool.spare_inputs.lock().unwrap());
    xp.resize(n_act * h, 0.);
    xg.resize(n_act * h_groups, 0.);
    for (c, &tok) in routing.active_toks.iter().enumerate() {
        if let Err(e) = x.prepare_row(
            tok,
            h,
            gs,
            &mut xp[c * h..(c + 1) * h],
            &mut xg[c * h_groups..(c + 1) * h_groups],
        ) {
            *pool.spare.lock().unwrap() = jobs;
            *pool.spare_inputs.lock().unwrap() = (xp, xg);
            *pool.spare_routing.lock().unwrap() = routing;
            return Err(bridge_error(format!("{e:?}")));
        }
    }
    let mut outputs = std::mem::take(&mut *pool.spare_outputs.lock().unwrap());
    outputs.resize_with(jobs.len() * slices, || Mutex::new(Vec::new()));
    let batch = Arc::new(Batch {
        outputs,
        store: store.clone(),
        xp,
        xg,
        jobs,
        slices,
        tokens,
        next: Counter::new(0),
        failed: AtomicBool::new(false),
        completion: Completion::new(pool.threads.len()),
    });
    let work_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let run_result = pool.run(&batch, work_deadline);
    if run_result.is_ok() {
        reduce_batch_outputs(&batch, &routing, h, slices, out);
    }
    *pool.spare_routing.lock().unwrap() = routing;
    if let Ok(batch) = Arc::try_unwrap(batch) {
        *pool.spare.lock().unwrap() = batch.jobs;
        *pool.spare_outputs.lock().unwrap() = batch.outputs;
        *pool.spare_inputs.lock().unwrap() = (batch.xp, batch.xg);
    }
    run_result
}
fn reduce_batch_outputs(
    batch: &Batch,
    routing: &RoutingScratch,
    h: usize,
    slices: usize,
    out: Output<'_>,
) {
    match out {
        Output::F32(y) => {
            for (tok, &compact) in routing.tok_to_compact.iter().enumerate() {
                if compact == u32::MAX {
                    y[tok * h..(tok + 1) * h].fill(0.);
                }
            }
            for (c, &tok) in routing.active_toks.iter().enumerate() {
                let dst = &mut y[tok * h..(tok + 1) * h];
                let mut first = true;
                for &expert in &routing.token_routes[c] {
                    let j = routing.expert_to_job[expert] as usize;
                    let row = batch.jobs[j].rows.iter().position(|r| r.0 == c).unwrap();
                    let start = row * h;
                    for slice in 0..slices {
                        let output = batch.outputs[j * slices + slice].lock().unwrap();
                        let src = &output[start..start + h];
                        if first {
                            dst.copy_from_slice(src);
                            first = false;
                        } else {
                            eval::add_assign_slice(dst, src);
                        }
                    }
                }
            }
        }
        Output::F16(out_f16) => {
            for (tok, &compact) in routing.tok_to_compact.iter().enumerate() {
                if compact == u32::MAX {
                    out_f16[tok * h..(tok + 1) * h].fill(0);
                }
            }
            let mut acc = [0.0f32; 4096];
            for (c, &tok) in routing.active_toks.iter().enumerate() {
                let dst_f32 = &mut acc[..h];
                let mut first = true;
                for &expert in &routing.token_routes[c] {
                    let j = routing.expert_to_job[expert] as usize;
                    let row = batch.jobs[j].rows.iter().position(|r| r.0 == c).unwrap();
                    let start = row * h;
                    for slice in 0..slices {
                        let output = batch.outputs[j * slices + slice].lock().unwrap();
                        let src = &output[start..start + h];
                        if first {
                            dst_f32.copy_from_slice(src);
                            first = false;
                        } else {
                            eval::add_assign_slice(dst_f32, src);
                        }
                    }
                }
                if !first {
                    eval::f32_to_f16_slice(dst_f32, &mut out_f16[tok * h..(tok + 1) * h]);
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mailbox_abi() {
        assert_eq!(std::mem::offset_of!(Request, ids), 64);
        assert_eq!(std::mem::offset_of!(Request, x), 1088);
        assert_eq!(std::mem::size_of::<Request>(), 66624);
        assert_eq!(std::mem::offset_of!(Response, status), 8);
        assert_eq!(std::mem::offset_of!(Response, y), 64);
        assert_eq!(std::mem::size_of::<Response>(), 131136);
    }
    #[test]
    fn new_prefill_only_partitions_ep_cpus_and_matches_prefill_output() {
        let _device = CudaDevice::set_current(0).unwrap();
        let r0 = Runtime::new_prefill_only(0, 0, 2).unwrap();
        let r1 = Runtime::new_prefill_only(0, 1, 2).unwrap();
        assert!(r0.server.is_none());
        assert!(r1.server.is_none());
        assert!(!r0.pool.cpus.is_empty());
        assert!(!r1.pool.cpus.is_empty());
        if detect_worker_cpus().len() >= 2 {
            for c in &r0.pool.cpus {
                assert!(
                    !r1.pool.cpus.contains(c),
                    "EP=2 prefill-only worker pools must use disjoint physical CPU sets"
                );
            }
        }
        let store = zero_store();
        let x = vec![0x3c00u16; 2 * 128];
        let mut out0 = vec![0u16; 2 * 128];
        let mut out1 = vec![0u16; 2 * 128];
        let mut toks = vec![0i32; 96];
        let mut wts = vec![0.0f32; 96];
        toks[0] = 0;
        toks[1] = 1;
        wts[0] = 0.5;
        wts[1] = 0.5;
        r0.prefill(&store, &x, 2, &[0], &[2], &toks, &wts, &mut out0)
            .unwrap();
        r1.prefill(&store, &x, 2, &[0], &[2], &toks, &wts, &mut out1)
            .unwrap();
        assert_eq!(out0, out1);
    }
    #[test]
    fn mailbox_retains_store_and_responds_until_joined_shutdown() {
        let _device = CudaDevice::set_current(0).unwrap();
        let layout = crate::moe_offload::ExpertPackLayout::new(128, 128, 128).unwrap();
        let store = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..1,
            &vec![0; layout.stride],
            layout.stride,
        )
        .unwrap();
        let stream = crate::cuda::CudaStream::new(0).unwrap();
        let runtime = Runtime::new(&stream).unwrap();
        let mailbox = runtime.register(&store).unwrap();
        let other = runtime.register(&store).unwrap();
        drop(store);
        assert!(mailbox.store.expert_pack(0).is_some());
        for seq in 1..=3 {
            // SAFETY: no GPU producer in this test; CPU takes ownership after seq publish.
            unsafe {
                let req = mailbox.req.host_ptr.cast::<Request>();
                (*req).tokens = 1;
                (*req).top_k = 1;
                (*req).hidden = 128;
                (*req).ids.fill(-1);
                (*req).seq.store(seq, Ordering::Release);
                let resp = mailbox.resp.host_ptr.cast::<Response>();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while (*resp).seq.load(Ordering::Acquire) != seq {
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::yield_now();
                }
                assert!((&(*resp).y)[..128].iter().all(|&v| v == 0.));
            }
        }
        drop(other);
        drop(mailbox);
        // Last owner synchronizes first, then joins responder and all fifteen background workers.
        drop(runtime);
    }
    #[test]
    fn responder_failure_releases_mailbox_and_poisoned_stream_rejects_outputs() {
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = crate::cuda::CudaStream::new(0).unwrap();
        let event = crate::cuda::CudaEvent::new(0, false).unwrap();
        let dependent = crate::cuda::CudaStream::new(0).unwrap();
        let runtime = Runtime::new(&stream).unwrap();
        let mailbox = runtime.register(&zero_store()).unwrap();
        // Record before the failure: events must observe later asynchronous errors.
        event.record(&stream).unwrap();
        dependent.wait_event(&event).unwrap();
        for seq in 1..=2 {
            // SAFETY: test is the only producer; responder acknowledges each sequence
            // before the next write. An invalid route exercises the actual error path.
            unsafe {
                let req = mailbox.req.host_ptr.cast::<Request>();
                (*req).tokens = 1;
                (*req).top_k = 1;
                (*req).hidden = 128;
                (*req).ids.fill(-1);
                (*req).ids[0] = 999;
                (*req).seq.store(seq, Ordering::Release);
                let resp = mailbox.resp.host_ptr.cast::<Response>();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while (*resp).seq.load(Ordering::Acquire) != seq {
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::yield_now();
                }
            }
            assert!(stream.synchronize().is_err());
            assert!(event.synchronize().is_err());
            assert!(dependent.synchronize().is_err());
        }
        // Exercise the actual GPU failed-response path: it must leave the hot
        // output untouched and finish, not consume stale y or spin forever.
        let out = crate::cuda::DeviceBuffer::alloc(0, 128 * 2).unwrap();
        let misses = crate::cuda::DeviceBuffer::alloc(0, 8).unwrap();
        let expected = [0x3c00u16; 128];
        out.copy_from_host(&expected, &stream).unwrap();
        misses.copy_from_host(&[0i32, 1], &stream).unwrap();
        let ex = stream.exec(0, 0);
        // SAFETY: the request was acknowledged above; mailbox/out/misses are live
        // until device synchronization, and the output has the advertised shape.
        crate::cuda::check_rc(
            unsafe {
                w34_kernels::ffi::w34_moe_hybrid_wait_and_add(
                    &ex,
                    mailbox.request_ptr(),
                    mailbox.response_ptr(),
                    out.addr() as *mut core::ffi::c_void,
                    misses.addr() as *const i32,
                    1,
                    128,
                )
            },
            "test failed mailbox",
        )
        .unwrap();
        assert!(stream.synchronize().is_err());
        // Inspect raw memory for this test only; production callers may not
        // consume outputs from the poisoned stream, even if CUDA itself finished.
        _device.synchronize().unwrap();
        let inspect = crate::cuda::CudaStream::new(0).unwrap();
        let mut actual = [0u16; 128];
        out.copy_to_host(&mut actual, &inspect).unwrap();
        inspect.synchronize().unwrap();
        assert_eq!(actual, expected);
        // Teardown must still drain the GPU and join workers despite host failure.
        drop(runtime);
    }

    fn zero_store() -> HostExpertStore {
        let layout = crate::moe_offload::ExpertPackLayout::new(256, 128, 128).unwrap();
        HostExpertStore::from_raw_packs(
            layout.clone(),
            0..2,
            &vec![0; 2 * layout.stride],
            2 * layout.stride,
        )
        .unwrap()
    }
    #[test]
    fn pool_reuses_buffers_and_reduces_deterministically() {
        let pool = Pool::new().unwrap();
        let store = zero_store();
        let tasks = [(0, 1, 0.2), (0, 0, 0.3), (0, 1, 0.5), (1, 0, 0.7)];
        let x = vec![0.; 256];
        let first = evaluate(&pool, &store, &x, 2, &tasks).unwrap();
        for _ in 0..2000 {
            assert_eq!(evaluate(&pool, &store, &x, 2, &tasks).unwrap(), first);
        }
        assert_eq!(pool.spare_outputs.lock().unwrap().len(), 4);
        assert!(evaluate(&pool, &store, &x, 2, &[(2, 0, 1.)]).is_err());
    }
    #[test]
    fn timeout_retains_owned_batch_and_rejects_overlap() {
        let pool = Pool {
            state: Arc::new((Mutex::new(Dispatch::default()), Wake::default())),
            cpus: vec![0],
            threads: Vec::new(),
            submit: Mutex::new(()),
            spare: Mutex::new(Vec::new()),
            spare_outputs: Mutex::new(Vec::new()),
            spare_inputs: Mutex::new((Vec::new(), Vec::new())),
            spare_routing: Mutex::new(RoutingScratch::default()),
            caller_scratch: Mutex::new(Scratch::default()),
        };
        let batch = Arc::new(Batch {
            store: zero_store(),
            xp: vec![1.; 128],
            xg: vec![1.],
            jobs: Vec::new(),
            outputs: Vec::new(),
            slices: 1,
            tokens: 1,
            next: Counter::new(0),
            failed: AtomicBool::new(false),
            completion: Completion::new(1),
        });
        let weak = Arc::downgrade(&batch);
        assert!(pool.run(&batch, std::time::Instant::now()).is_err());
        drop(batch);
        let retained = weak.upgrade().unwrap();
        assert_eq!(retained.xp[0], 1.);
        assert!(evaluate(&pool, &retained.store, &[0.; 128], 1, &[]).is_err());
        retained.completion.remaining.store(0, Ordering::Release);
        retained.failed.store(true, Ordering::Relaxed);
        assert!(pool.run(&retained, std::time::Instant::now()).is_err());
    }
    #[test]
    fn worker_panic_is_reported_and_pool_survives() {
        let pool = Pool::new().unwrap();
        let batch = Arc::new(Batch {
            store: zero_store(),
            xp: vec![0.; 128],
            xg: vec![0.],
            jobs: vec![Job {
                expert: 99,
                rows: vec![(0, 1.)],
            }],
            outputs: vec![Mutex::new(Vec::new())],
            slices: 1,
            tokens: 1,
            next: Counter::new(0),
            failed: AtomicBool::new(false),
            completion: Completion::new(pool.threads.len()),
        });
        assert!(
            pool.run(
                &batch,
                std::time::Instant::now() + std::time::Duration::from_secs(5)
            )
            .is_err()
        );
        assert!(evaluate(&pool, &batch.store, &[0.; 128], 1, &[]).is_ok());
    }
    /// CPU-only dispatch/evaluator probe at Flash Next geometry. Run explicitly
    /// with --ignored --nocapture --test-threads=1 on an otherwise idle host.
    #[test]
    #[ignore]
    fn benchmark_flash_next_pool() {
        assert!(eval::supported());
        let layout = crate::moe_offload::ExpertPackLayout::new(640, 2560, 128).unwrap();
        let mut packs = vec![0x89u8; 32 * layout.stride];
        for pack in packs.chunks_exact_mut(layout.stride) {
            for range in [layout.gate_up_scales.clone(), layout.down_scales.clone()] {
                for scale in pack[range].as_chunks_mut::<2>().0 {
                    scale.copy_from_slice(&0x2000u16.to_le_bytes());
                }
            }
        }
        let store =
            HostExpertStore::from_raw_packs(layout.clone(), 0..32, &packs, packs.len()).unwrap();
        #[cfg(target_os = "linux")]
        // SAFETY: `cpu_set_t` is plain old data; `sched_getaffinity` reads the current thread's affinity mask.
        let saved_mask = unsafe {
            let mut m: libc::cpu_set_t = std::mem::zeroed();
            let _ = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut m);
            m
        };
        let pool = Pool::new().unwrap();
        pin_current_thread(pool.cpus[0]);
        for tokens in [1, 4, 16] {
            let x = vec![0.125; tokens * layout.hidden];
            let mut samples = Vec::new();
            for step in 0..220 {
                let routes = (0..tokens)
                    .flat_map(|t| (0..8).map(move |k| (t, (step + k + t * 3) % 32, 0.125)))
                    .collect::<Vec<_>>();
                let start = std::time::Instant::now();
                let y = evaluate(&pool, &store, &x, tokens, &routes).unwrap();
                if step >= 20 {
                    samples.push(start.elapsed().as_secs_f64() * 1e6);
                }
                assert!(y.iter().all(|v| v.is_finite() && *v != 0.));
            }
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "pool tokens={tokens} median={:.1}us p90={:.1}us",
                samples[100], samples[180]
            );
        }
        drop(pool);
        // Restore original process CPU affinity before spawning Runtime's worker pool.
        #[cfg(target_os = "linux")]
        // SAFETY: `saved_mask` was captured by `sched_getaffinity` before `pin_current_thread`.
        unsafe {
            let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &saved_mask);
        }
        let _device = CudaDevice::set_current(0).unwrap();
        let stream = crate::cuda::CudaStream::new(0).unwrap();
        let runtime = Runtime::new(&stream).unwrap();
        let prefill_tokens = 6144;
        let x_f16 = vec![0x3000u16; prefill_tokens * layout.hidden];
        let mut out_f16 = vec![0u16; prefill_tokens * layout.hidden];
        let experts = (0..16).collect::<Vec<i32>>();
        let counts = vec![12i32; 16];
        let mut toks = vec![0i32; 16 * 96];
        let mut wts = vec![0.0f32; 16 * 96];
        for j in 0..16 {
            for s in 0..12 {
                toks[j * 96 + s] = ((j * 37 + s * 191) % prefill_tokens) as i32;
                wts[j * 96 + s] = 0.125;
            }
        }
        let mut prefill_samples = Vec::new();
        for step in 0..25 {
            let start = std::time::Instant::now();
            runtime
                .prefill(
                    &store,
                    &x_f16,
                    prefill_tokens,
                    &experts,
                    &counts,
                    &toks,
                    &wts,
                    &mut out_f16,
                )
                .unwrap();
            if step >= 5 {
                prefill_samples.push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
        prefill_samples.sort_by(f64::total_cmp);
        eprintln!(
            "prefill tokens={prefill_tokens} (16x12 active) median={:.1}us p90={:.1}us",
            prefill_samples[10], prefill_samples[18]
        );
        let heavy_jobs = 220usize;
        let heavy_toks_per_job = 5usize;
        let heavy_packs = vec![0x89u8; 256 * layout.stride];
        let heavy_store = HostExpertStore::from_raw_packs(
            layout.clone(),
            0..256,
            &heavy_packs,
            heavy_packs.len(),
        )
        .unwrap();
        let heavy_experts = (0..heavy_jobs as i32).collect::<Vec<i32>>();
        let heavy_counts = vec![heavy_toks_per_job as i32; heavy_jobs];
        let mut heavy_toks = vec![0i32; heavy_jobs * 96];
        let mut heavy_wts = vec![0.0f32; heavy_jobs * 96];
        for j in 0..heavy_jobs {
            for s in 0..heavy_toks_per_job {
                heavy_toks[j * 96 + s] = ((j * 29 + s * 131) % prefill_tokens) as i32;
                heavy_wts[j * 96 + s] = 0.125;
            }
        }
        let mut heavy_samples = Vec::new();
        for step in 0..15 {
            let start = std::time::Instant::now();
            runtime
                .prefill(
                    &heavy_store,
                    &x_f16,
                    prefill_tokens,
                    &heavy_experts,
                    &heavy_counts,
                    &heavy_toks,
                    &heavy_wts,
                    &mut out_f16,
                )
                .unwrap();
            if step >= 3 {
                heavy_samples.push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
        heavy_samples.sort_by(f64::total_cmp);
        eprintln!(
            "prefill tokens={prefill_tokens} ({heavy_jobs}x{heavy_toks_per_job} active) median={:.1}us p90={:.1}us",
            heavy_samples[6], heavy_samples[10]
        );
    }

    #[test]
    fn physical_cpu_selection_deduplicates_smt_siblings_and_caps_workers() {
        // 8 physical cores with 2 SMT threads each (0..8 and 8..16).
        let smt_8c16t = (0..16).map(|c| (c, Some((0, (c % 8) as i32))));
        assert_eq!(
            select_physical_cpus(smt_8c16t, MAX_WORKERS),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        // Non-contiguous container cpuset where CPUs 2 and 18 share physical core 2.
        let cpuset = [
            (2, Some((0, 2))),
            (4, Some((0, 4))),
            (18, Some((0, 2))),
            (22, Some((0, 6))),
        ];
        assert_eq!(select_physical_cpus(cpuset, MAX_WORKERS), vec![2, 4, 22]);
        // Fallback without sysfs topology on a 32-thread system halves SMT threads.
        assert_eq!(
            select_physical_cpus((0..32).map(|c| (c, None)), MAX_WORKERS).len(),
            16
        );
        // Multi-CCD / multi-die topology where `core_id` repeats `0..4` per die:
        // `parse_canonical_core_key` keeps dies distinct via `thread_siblings_list` or `die_id`.
        let multi_ccd = (0..16).map(|c| {
            let physical = c % 8;
            let die = (physical / 4) as i32;
            let core_in_die = (physical % 4) as i32;
            let siblings = format!("{physical},{}", physical + 8);
            (
                c,
                Some(parse_canonical_core_key(
                    0,
                    Some(die),
                    core_in_die,
                    Some(&siblings),
                )),
            )
        });
        assert_eq!(
            select_physical_cpus(multi_ccd, MAX_WORKERS),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        let multi_die_no_siblings = (0..8).map(|c| {
            let die = (c / 4) as i32;
            let core_in_die = (c % 4) as i32;
            (
                c,
                Some(parse_canonical_core_key(0, Some(die), core_in_die, None)),
            )
        });
        assert_eq!(
            select_physical_cpus(multi_die_no_siblings, MAX_WORKERS),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            select_physical_cpus(std::iter::empty(), MAX_WORKERS),
            vec![0]
        );
    }

    #[test]
    fn pool_joins_and_completes() {
        let pool = Pool::new().unwrap();
        assert!(!pool.cpus.is_empty() && pool.cpus.len() <= MAX_WORKERS);
        assert_eq!(pool.threads.len(), pool.cpus.len() - 1);
        drop(pool);
    }
}

//! CLI-facing, native Hugging Face model acquisition. No Python or remote code.
//!
//! Online resolution pins a commit before fetching metadata and weights. Offline
//! resolution uses the SDK's cache-only lookup and validates checkpoint completeness.
//! The loader's older cache-only lookup remains available to fixture/test callers.
mod files;
mod progress;
use hf_hub::{
    HFClient, HFClientBuilder, HFClientSync, HFError, HFRepositorySync, repository::RepoTypeModel,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Default)]
pub struct HubOptions {
    /// Branch, tag, or full commit SHA. Defaults to main for Hub IDs only.
    pub revision: Option<String>,
    pub offline: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelRole {
    Target,
    Draft,
    Lora,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    commit: String,
    files: Vec<String>,
}

/// Resolve a local model directory or acquire a supported Hub checkpoint.
/// Downloads and progress messages happen before CUDA initialization.
pub fn resolve_model(spec: &str, options: &HubOptions, role: ModelRole) -> io::Result<PathBuf> {
    if role == ModelRole::Lora {
        let mut lora_spec = crate::lora::LoraModuleSpec {
            name: spec.to_string(),
            path: spec.to_string(),
            revision: options.revision.clone(),
            resolved_dir: None,
        };
        return resolve_lora_adapter(&mut lora_spec, options.offline);
    }
    if let Some(path) = local_model(spec, options)? {
        return Ok(path);
    }
    let options = HubOptions {
        offline: options.offline || env_enabled("HF_HUB_OFFLINE"),
        ..options.clone()
    };
    resolve_remote(
        spec,
        &options,
        role,
        HFClient::builder(),
        !env_enabled("HF_HUB_DISABLE_PROGRESS_BARS"),
    )
}

/// Resolve a local PEFT LoRA adapter directory or acquire one from Hugging Face Hub
/// (supporting both root repositories `owner/repo` and subdirectories `owner/repo/sub/dir`).
pub fn resolve_lora_adapter(
    spec: &mut crate::lora::LoraModuleSpec,
    offline: bool,
) -> io::Result<PathBuf> {
    let opts = HubOptions {
        revision: spec.revision.clone(),
        offline: offline || env_enabled("HF_HUB_OFFLINE"),
    };
    if let Some(dir) = local_model(&spec.path, &opts)? {
        validate_lora_dir(&dir)?;
        spec.resolved_dir = Some(dir.clone());
        return Ok(dir);
    }
    resolve_lora_remote(
        spec,
        opts.offline,
        HFClient::builder(),
        !env_enabled("HF_HUB_DISABLE_PROGRESS_BARS"),
    )
}

fn resolve_lora_remote(
    spec: &mut crate::lora::LoraModuleSpec,
    offline: bool,
    builder: HFClientBuilder,
    progress: bool,
) -> io::Result<PathBuf> {
    let (id, subfolder) = crate::lora::split_hf_repo_and_subfolder(&spec.path)?;
    let revision = spec.revision.as_deref().unwrap_or("main");
    safe_revision(revision)?;
    let normalized_sha = is_commit(revision).then(|| revision.to_ascii_lowercase());
    let revision = normalized_sha.as_deref().unwrap_or(revision);
    let client = builder.build().map_err(hub_error)?;
    let repo_dir = client
        .cache_dir()
        .join(format!("models--{}", id.replace('/', "--")));
    let client = HFClientSync::from_inner(client).map_err(hub_error)?;
    let (owner, name) = id.rsplit_once('/').unwrap_or(("", &id));
    let repo = client.model(owner, name);
    if offline {
        if spec.revision.is_none()
            && !repo_dir.join("refs/main").is_file()
            && let Ok(dir) = crate::lora::resolve_cached_or_local_lora_dir(&spec.path, None)
        {
            validate_lora_dir(&dir)?;
            eprintln!(
                "Using cached LoRA adapter '{}': {}",
                spec.name,
                dir.display()
            );
            spec.resolved_dir = Some(dir.clone());
            return Ok(dir);
        }
        let snap_dir = match repo
            .snapshot_download()
            .revision(revision)
            .local_files_only(true)
            .send()
        {
            Ok(dir) => dir,
            Err(HFError::Io(ref e))
                if matches!(
                    e.kind(),
                    io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
                ) =>
            {
                resolve_readonly_cached_snapshot(&repo_dir, revision)?
            }
            Err(err) => return Err(hub_error(err)),
        };
        let adapter_dir =
            validate_cached_lora_snapshot(&repo_dir, &snap_dir, subfolder.as_deref())?;
        eprintln!(
            "Using cached LoRA adapter '{}': {}",
            spec.name,
            adapter_dir.display()
        );
        spec.resolved_dir = Some(adapter_dir.clone());
        return Ok(adapter_dir);
    }
    let lock = repo_download_lock(&repo_dir);
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let lora_subdir_key = subfolder.as_deref().unwrap_or("");
    if is_commit(revision) && lora_manifest_path(&repo_dir, revision, lora_subdir_key).is_file() {
        let snap_dir = repo_dir.join("snapshots").join(revision);
        if let Ok(adapter_dir) =
            validate_cached_lora_snapshot(&repo_dir, &snap_dir, subfolder.as_deref())
        {
            eprintln!(
                "Using cached LoRA adapter '{}': {}",
                spec.name,
                adapter_dir.display()
            );
            spec.resolved_dir = Some(adapter_dir.clone());
            return Ok(adapter_dir);
        }
    }
    eprintln!("Resolving Hugging Face LoRA {}@{revision}…", spec.path);
    let info = match repo.info().revision(revision).send() {
        Ok(info) => info,
        Err(err) => {
            let transient = is_transient_hub_failure(&err);
            let io_err = hub_error(err);
            if transient
                && let Some(dir) = try_cached_lora_fallback(
                    &repo,
                    &repo_dir,
                    revision,
                    subfolder.as_deref(),
                    &spec.name,
                    &io_err,
                )
            {
                spec.resolved_dir = Some(dir.clone());
                return Ok(dir);
            }
            return Err(io_err);
        }
    };
    let commit = info
        .sha
        .filter(|s| is_commit(s))
        .map(|s| s.to_ascii_lowercase())
        .ok_or_else(|| invalid("Hub did not return a full commit SHA"))?;
    if is_commit(revision) && !commit.eq_ignore_ascii_case(revision) {
        return Err(invalid("Hub returned a different commit than requested"));
    }
    let available: BTreeSet<_> = info
        .siblings
        .ok_or_else(|| invalid("Hub returned no file listing for LoRA repository"))?
        .into_iter()
        .map(|s| s.rfilename)
        .collect();
    let required: Vec<String> = match &subfolder {
        Some(sub) => vec![
            format!("{sub}/adapter_config.json"),
            format!("{sub}/adapter_model.safetensors"),
        ],
        None => vec![
            "adapter_config.json".to_string(),
            "adapter_model.safetensors".to_string(),
        ],
    };
    for req in &required {
        safe_path(req)?;
        if !available.contains(req) {
            return Err(invalid(format!(
                "LoRA repository '{}' is missing required file '{req}'",
                spec.path
            )));
        }
    }
    let optional_candidates: Vec<String> = match &subfolder {
        Some(sub) => vec![
            format!("{sub}/decision_config.json"),
            format!("{sub}/readout.safetensors"),
        ],
        None => vec![
            "decision_config.json".to_string(),
            "readout.safetensors".to_string(),
        ],
    };
    let mut optional_files = Vec::new();
    for opt in optional_candidates {
        safe_path(&opt)?;
        if available.contains(&opt) {
            optional_files.push(opt);
        }
    }
    let mut selected = required.clone();
    selected.extend(optional_files.iter().cloned());
    let cached_snap = repo_dir.join("snapshots").join(&commit);
    let cached_adapter_dir = match &subfolder {
        Some(sub) => cached_snap.join(sub),
        None => cached_snap.clone(),
    };
    if cached_snap.is_dir()
        && files::validate_files(&cached_snap, &selected).is_ok()
        && validate_lora_dir(&cached_adapter_dir).is_ok()
    {
        let mpath = lora_manifest_path(&repo_dir, &commit, lora_subdir_key);
        let manifest_ok = fs::read(&mpath)
            .ok()
            .and_then(|b| serde_json::from_slice::<Manifest>(&b).ok())
            .is_some_and(|m| m.version == 1 && m.commit == commit && m.files == selected);
        if !manifest_ok {
            let manifest = Manifest {
                version: 1,
                commit: commit.clone(),
                files: selected,
            };
            atomic_write(&mpath, &serde_json::to_vec(&manifest)?)?;
        }
        if !is_commit(revision) {
            let ref_path = repo_dir.join("refs").join(revision);
            if fs::read_to_string(&ref_path).ok().as_deref().map(str::trim) != Some(&commit) {
                atomic_write(&ref_path, commit.as_bytes())?;
            }
        }
        eprintln!(
            "LoRA ready: {} ({}@{} -> {})",
            spec.name,
            id,
            commit,
            cached_adapter_dir.display()
        );
        spec.resolved_dir = Some(cached_adapter_dir.clone());
        return Ok(cached_adapter_dir);
    }
    let manifest = Manifest {
        version: 1,
        commit: commit.clone(),
        files: selected.clone(),
    };
    atomic_write(
        &lora_manifest_path(&repo_dir, &commit, lora_subdir_key),
        &serde_json::to_vec(&manifest)?,
    )?;
    let reporter = progress::Reporter::new(progress)?;
    let snap_dir = match fetch_files(&repo, &commit, &selected, &reporter) {
        Ok(dir) => dir,
        Err((transient, io_err)) => {
            if transient
                && !is_commit(revision)
                && let Some(dir) = try_cached_lora_fallback(
                    &repo,
                    &repo_dir,
                    revision,
                    subfolder.as_deref(),
                    &spec.name,
                    &io_err,
                )
            {
                spec.resolved_dir = Some(dir.clone());
                return Ok(dir);
            }
            return Err(io_err);
        }
    };
    files::validate_files(&snap_dir, &selected)?;
    let adapter_dir = match &subfolder {
        Some(sub) => snap_dir.join(sub),
        None => snap_dir,
    };
    validate_lora_dir(&adapter_dir)?;
    if !is_commit(revision) {
        atomic_write(&repo_dir.join("refs").join(revision), commit.as_bytes())?;
    }
    eprintln!(
        "LoRA ready: {} ({}@{} -> {})",
        spec.name,
        id,
        commit,
        adapter_dir.display()
    );
    spec.resolved_dir = Some(adapter_dir.clone());
    Ok(adapter_dir)
}

fn validate_lora_dir(dir: &Path) -> io::Result<()> {
    crate::lora::PeftConfig::from_dir(dir)?;
    let st_path = dir.join("adapter_model.safetensors");
    if !fs::metadata(&st_path).is_ok_and(|m| m.is_file() && m.len() > 0) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing or empty {}", st_path.display()),
        ));
    }
    crate::loader::SafetensorsArchive::open(&st_path)?;
    Ok(())
}

pub(crate) fn cache_dir() -> PathBuf {
    hf_hub::resolve_cache_dir()
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|s| matches!(s.to_ascii_uppercase().as_str(), "1" | "ON" | "YES" | "TRUE"))
}

fn local_model(spec: &str, options: &HubOptions) -> io::Result<Option<PathBuf>> {
    let path = Path::new(spec);
    if path.exists() {
        if options.revision.is_some() {
            return Err(invalid(
                "--revision/--draft-revision cannot be used with a local model path",
            ));
        }
        if !path.is_dir() {
            return Err(invalid("model path must be a checkpoint directory"));
        }
        return Ok(Some(path.to_path_buf()));
    }
    if path.is_absolute() || spec.starts_with('.') || spec.starts_with('~') || spec.contains('\\') {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "local model directory does not exist",
        ));
    }
    Ok(None)
}

fn repo_id(spec: &str) -> io::Result<String> {
    let id = spec
        .strip_prefix("models--")
        .map_or_else(|| spec.into(), |s| s.replace("--", "/"));
    let parts: Vec<_> = id.split('/').collect();
    if parts.is_empty()
        || parts.len() > 2
        || parts.iter().any(|p| {
            p.is_empty()
                || p.starts_with(['.', '-'])
                || p.ends_with(['.', '-'])
                || p.contains("..")
                || p.contains("--")
                || !p
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        })
    {
        return Err(invalid(
            "expected a local model directory or Hugging Face model ID (username/modelname)",
        ));
    }
    Ok(id)
}

fn safe_path(path: &str) -> io::Result<()> {
    if path.is_empty()
        || path
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
        || !path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-'))
    {
        return Err(invalid(
            "unsafe or unsupported Hub filename/revision; expected a relative path without traversal or glob characters",
        ));
    }
    Ok(())
}

fn safe_revision(revision: &str) -> io::Result<()> {
    if revision.is_empty()
        || revision.split('/').any(|part| {
            part.is_empty()
                || part.starts_with('.')
                || part.ends_with('.')
                || part.ends_with(".lock")
        })
        || revision.contains("..")
        || revision.contains("@{")
        || revision
            .chars()
            .any(|c| c.is_control() || c.is_whitespace() || "~^:?*[\\".contains(c))
    {
        return Err(invalid(
            "invalid Hub revision; expected a branch, tag, or full commit SHA",
        ));
    }
    Ok(())
}

fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
fn read_json(path: &Path) -> io::Result<serde_json::Value> {
    serde_json::from_slice(&fs::read(path)?).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid {}: {e}",
                path.file_name().unwrap_or_default().to_string_lossy()
            ),
        )
    })
}

fn resolve_remote(
    spec: &str,
    options: &HubOptions,
    role: ModelRole,
    builder: HFClientBuilder,
    progress: bool,
) -> io::Result<PathBuf> {
    let id = repo_id(spec)?;
    let revision = options.revision.as_deref().unwrap_or("main");
    safe_revision(revision)?;
    let normalized_sha = is_commit(revision).then(|| revision.to_ascii_lowercase());
    let revision = normalized_sha.as_deref().unwrap_or(revision);
    // The SDK owns authentication, endpoint, cache configuration, and transport.
    let client = builder.build().map_err(hub_error)?;
    let repo_dir = client
        .cache_dir()
        .join(format!("models--{}", id.replace('/', "--")));
    let client = HFClientSync::from_inner(client).map_err(hub_error)?;
    let (owner, name) = id.rsplit_once('/').unwrap_or(("", &id));
    let repo = client.model(owner, name);
    if options.offline {
        let dir = match repo
            .snapshot_download()
            .revision(revision)
            .local_files_only(true)
            .send()
        {
            Ok(dir) => dir,
            Err(HFError::Io(ref e))
                if matches!(
                    e.kind(),
                    io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
                ) =>
            {
                resolve_readonly_cached_snapshot(&repo_dir, revision)?
            }
            Err(err) => return Err(hub_error(err)),
        };
        return validate_offline_snapshot(&repo_dir, &dir, role);
    }
    eprintln!("Resolving Hugging Face {id}@{revision}…");
    let result = download(&repo, &id, revision, &repo_dir, role, progress);
    result.map_err(|e| io::Error::new(e.kind(), format!("Hugging Face {id}@{revision}: {e}")))
}

fn repo_download_lock(repo_dir: &Path) -> std::sync::Arc<std::sync::Mutex<()>> {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex, OnceLock},
    };
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let map = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    map.lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(repo_dir.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn download(
    repo: &HFRepositorySync<RepoTypeModel>,
    id: &str,
    revision: &str,
    repo_dir: &Path,
    role: ModelRole,
    progress: bool,
) -> io::Result<PathBuf> {
    let lock = repo_download_lock(repo_dir);
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    if is_commit(revision) && manifest_path(repo_dir, revision).is_file() {
        let pinned_dir = repo_dir.join("snapshots").join(revision);
        if let Ok(dir) = validate_offline_snapshot(repo_dir, &pinned_dir, role) {
            return Ok(dir);
        }
    }
    let info = match repo.info().revision(revision).send() {
        Ok(info) => info,
        Err(err) => {
            let transient = is_transient_hub_failure(&err);
            let io_err = hub_error(err);
            if transient
                && let Some(cached) =
                    try_cached_model_fallback(repo, repo_dir, revision, role, &io_err)
            {
                return Ok(cached);
            }
            return Err(io_err);
        }
    };
    let commit = info
        .sha
        .filter(|s| is_commit(s))
        .map(|s| s.to_ascii_lowercase())
        .ok_or_else(|| invalid("Hub did not return a full commit SHA"))?;
    if is_commit(revision) && !commit.eq_ignore_ascii_case(revision) {
        return Err(invalid("Hub returned a different commit than requested"));
    }
    let available: BTreeSet<_> = info
        .siblings
        .ok_or_else(|| invalid("Hub returned no checkpoint file listing"))?
        .into_iter()
        .map(|s| s.rfilename)
        .collect();
    let metadata = files::metadata_files(&available)?;
    if role == ModelRole::Target && !available.contains("tokenizer.json") {
        return Err(invalid(
            "checkpoint has no tokenizer.json required by warp34",
        ));
    }
    let cached_dir = repo_dir.join("snapshots").join(&commit);
    if cached_dir.is_dir()
        && files::validate_files(&cached_dir, &metadata).is_ok()
        && files::validate_config(&cached_dir, role).is_ok()
        && let Ok(weights) = files::weight_files(&cached_dir, &available)
    {
        let mut selected = metadata.clone();
        selected.extend(weights);
        if files::validate_files(&cached_dir, &selected).is_ok()
            && files::validate_checkpoint(&cached_dir, role, &available).is_ok()
        {
            let mpath = manifest_path(repo_dir, &commit);
            let manifest_ok = fs::read(&mpath)
                .ok()
                .and_then(|b| serde_json::from_slice::<Manifest>(&b).ok())
                .is_some_and(|m| m.version == 1 && m.commit == commit && m.files == selected);
            if !manifest_ok {
                let manifest = Manifest {
                    version: 1,
                    commit: commit.clone(),
                    files: selected,
                };
                atomic_write(&mpath, &serde_json::to_vec(&manifest)?)?;
            }
            if !is_commit(revision) {
                let ref_path = repo_dir.join("refs").join(revision);
                if fs::read_to_string(&ref_path).ok().as_deref().map(str::trim) != Some(&commit) {
                    atomic_write(&ref_path, commit.as_bytes())?;
                }
            }
            eprintln!("Model ready: {id}@{commit} ({})", cached_dir.display());
            return Ok(cached_dir);
        }
    }
    let reporter = progress::Reporter::new(progress)?;
    let dir = match fetch_files(repo, &commit, &metadata, &reporter) {
        Ok(dir) => dir,
        Err((transient, io_err)) => {
            if transient
                && !is_commit(revision)
                && let Some(cached) =
                    try_cached_model_fallback(repo, repo_dir, revision, role, &io_err)
            {
                return Ok(cached);
            }
            return Err(io_err);
        }
    };
    files::validate_config(&dir, role)?;
    let weights = files::weight_files(&dir, &available)?;
    let mut selected = metadata;
    selected.extend(weights.iter().cloned());
    // Persist the required file set before downloading weights. A partially
    // completed acquisition must not become an apparently complete offline hit.
    let manifest = Manifest {
        version: 1,
        commit: commit.clone(),
        files: selected.clone(),
    };
    atomic_write(
        &manifest_path(repo_dir, &commit),
        &serde_json::to_vec(&manifest)?,
    )?;
    if let Err((transient, io_err)) = fetch_files(repo, &commit, &weights, &reporter) {
        if transient
            && !is_commit(revision)
            && let Some(cached) = try_cached_model_fallback(repo, repo_dir, revision, role, &io_err)
        {
            return Ok(cached);
        }
        return Err(io_err);
    }
    files::validate_files(&dir, &selected)?;
    files::validate_checkpoint(&dir, role, &available)?;
    // The client fetched by immutable SHA, so publish the caller's mutable ref
    // only after the entire selected snapshot is ready (atomic across readers).
    if !is_commit(revision) {
        atomic_write(&repo_dir.join("refs").join(revision), commit.as_bytes())?;
    }
    eprintln!("Model ready: {id}@{commit} ({})", dir.display());
    Ok(dir)
}

fn fetch_files(
    repo: &HFRepositorySync<RepoTypeModel>,
    commit: &str,
    names: &[String],
    reporter: &progress::Reporter,
) -> Result<PathBuf, (bool, io::Error)> {
    let dir = repo
        .snapshot_download()
        .revision(commit)
        .allow_patterns(names.to_vec())
        .max_workers(4)
        .progress(reporter.handler())
        .send()
        .map_err(|err| (is_transient_hub_failure(&err), hub_error(err)))?;
    // hf-hub deliberately skips per-file 404s. Required checkpoint files may
    // never be silently skipped, even when the overall snapshot call succeeds.
    files::validate_files(&dir, names).map_err(|e| (false, e))?;
    Ok(dir)
}

fn try_cached_model_fallback(
    repo: &HFRepositorySync<RepoTypeModel>,
    repo_dir: &Path,
    revision: &str,
    role: ModelRole,
    cause: &io::Error,
) -> Option<PathBuf> {
    let dir = repo
        .snapshot_download()
        .revision(revision)
        .local_files_only(true)
        .send()
        .ok()?;
    let validated = validate_offline_snapshot(repo_dir, &dir, role).ok()?;
    eprintln!(
        "Warning: could not reach Hugging Face ({cause}); falling back to verified cached snapshot {}",
        validated.display()
    );
    Some(validated)
}

fn lora_manifest_path(repo: &Path, commit: &str, subdir: &str) -> PathBuf {
    if subdir.is_empty() {
        repo.join(".warp34").join(format!("lora-{commit}.json"))
    } else {
        let slug = subdir.replace('/', "__");
        repo.join(".warp34")
            .join(format!("lora-{commit}-{slug}.json"))
    }
}

fn validate_cached_lora_snapshot(
    repo_dir: &Path,
    snap_dir: &Path,
    subfolder: Option<&str>,
) -> io::Result<PathBuf> {
    let commit = snap_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| invalid("invalid cached LoRA snapshot path"))?;
    if !is_commit(commit) {
        return Err(invalid(
            "cached Hugging Face LoRA ref does not contain a valid commit SHA",
        ));
    }
    let mpath = lora_manifest_path(repo_dir, commit, subfolder.unwrap_or(""));
    if let Ok(bytes) = fs::read(&mpath) {
        let manifest: Manifest = serde_json::from_slice(&bytes)
            .map_err(|_| invalid("invalid warp34 LoRA snapshot manifest"))?;
        if manifest.version != 1 || manifest.commit != commit {
            return Err(invalid("LoRA snapshot manifest version/commit mismatch"));
        }
        files::validate_files(snap_dir, &manifest.files)?;
    }
    let adapter_dir = match subfolder {
        Some(sub) => snap_dir.join(sub),
        None => snap_dir.to_path_buf(),
    };
    validate_lora_dir(&adapter_dir)?;
    Ok(adapter_dir)
}

fn try_cached_lora_fallback(
    repo: &HFRepositorySync<RepoTypeModel>,
    repo_dir: &Path,
    revision: &str,
    subfolder: Option<&str>,
    adapter_name: &str,
    cause: &io::Error,
) -> Option<PathBuf> {
    let snap_dir = repo
        .snapshot_download()
        .revision(revision)
        .local_files_only(true)
        .send()
        .ok()?;
    let validated = validate_cached_lora_snapshot(repo_dir, &snap_dir, subfolder).ok()?;
    eprintln!(
        "Warning: could not reach Hugging Face ({cause}); falling back to verified cached LoRA '{adapter_name}': {}",
        validated.display()
    );
    Some(validated)
}

fn is_transient_hub_failure(error: &HFError) -> bool {
    match error {
        HFError::Request { .. } | HFError::RateLimited { .. } => true,
        HFError::Http { context } => {
            context.status.is_server_error() || context.status.as_u16() == 429
        }
        _ => false,
    }
}

fn manifest_path(repo: &Path, commit: &str) -> PathBuf {
    repo.join(".warp34").join(format!("{commit}.json"))
}

fn resolve_readonly_cached_snapshot(repo_dir: &Path, revision: &str) -> io::Result<PathBuf> {
    let commit = if is_commit(revision) {
        revision.to_owned()
    } else {
        let ref_path = repo_dir.join("refs").join(revision);
        fs::read_to_string(&ref_path)
            .map_err(|e| {
                io::Error::new(
                    e.kind(),
                    "requested revision is not cached; offline mode forbids downloads",
                )
            })?
            .trim()
            .to_owned()
    };
    Ok(repo_dir.join("snapshots").join(commit))
}

fn validate_offline_snapshot(repo_dir: &Path, dir: &Path, role: ModelRole) -> io::Result<PathBuf> {
    let commit = dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| invalid("invalid cached snapshot path"))?;
    if !is_commit(commit) {
        return Err(invalid(
            "cached Hugging Face ref does not contain a valid commit SHA",
        ));
    }
    let names = match fs::read(manifest_path(repo_dir, commit)) {
        Ok(bytes) => {
            let manifest: Manifest = serde_json::from_slice(&bytes)
                .map_err(|_| invalid("invalid warp34 snapshot manifest"))?;
            if manifest.version != 1 || manifest.commit != commit {
                return Err(invalid("snapshot manifest version/commit mismatch"));
            }
            files::validate_files(dir, &manifest.files)?;
            manifest.files.into_iter().collect()
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => cached_files(dir)?,
        Err(e) => return Err(e),
    };
    files::validate_checkpoint(dir, role, &names)?;
    eprintln!("Using cached model: {}", dir.display());
    Ok(dir.to_path_buf())
}

fn cached_files(dir: &Path) -> io::Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(dir).map_err(|e| {
        io::Error::new(
            e.kind(),
            "model snapshot is not cached; offline mode forbids downloads",
        )
    })? {
        let entry = entry?;
        if entry.path().is_file() {
            names.insert(entry.file_name().to_string_lossy().into_owned());
        }
    }
    // Indexed shards can live below the root. Validate their paths before
    // checking files; never follow '..' entries from an untrusted index.
    if names.contains("model.safetensors.index.json") {
        let index = read_json(&dir.join("model.safetensors.index.json"))?;
        if let Some(map) = index["weight_map"].as_object() {
            for name in map.values().filter_map(|v| v.as_str()) {
                safe_path(name)?;
                if dir.join(name).is_file() {
                    names.insert(name.into());
                }
            }
        }
    }
    Ok(names)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| invalid("invalid cache metadata path"))?;
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(bytes)?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}

fn hub_error(error: HFError) -> io::Error {
    // Do not expose raw HTTP bodies, signed download URLs, or authorization
    // headers in diagnostics. Keep actionable categories instead.
    let message = match error {
        HFError::AuthRequired { .. } | HFError::Forbidden { .. } => "access denied; set HF_TOKEN or run `hf auth login`, and accept any gated-model license on Hugging Face".into(),
        HFError::RepoNotFound { .. } => "repository not found or private; check the model ID and HF_TOKEN / `hf auth login` credentials".into(),
        HFError::RevisionNotFound { .. } => "revision not found; check --revision / --draft-revision".into(),
        HFError::LocalEntryNotFound { .. } => return io::Error::new(io::ErrorKind::NotFound, "requested revision is not cached; offline mode forbids downloads"),
        HFError::EntryNotFound { .. } => "required checkpoint file not found at the selected revision".into(),
        HFError::RateLimited { .. } => "Hub rate limit reached; authenticate with HF_TOKEN or retry later".into(),
        HFError::Request { source, .. } if source.is_timeout() => "Hub request timed out; retry, or use --offline with a complete cached snapshot".into(),
        HFError::Request { .. } => "cannot connect to Hub; check network, TLS certificates, and HF_ENDPOINT, or use --offline with a complete cached snapshot".into(),
        HFError::Io(e) => return io::Error::new(e.kind(), format!("Hub cache I/O failed: {e}; check disk space and permissions")),
        HFError::CacheLockTimeout { .. } => "Hub cache is locked by another download; retry after it finishes".into(),
        HFError::Xet { .. } => "Hub Xet transfer failed; check network and disk space, then retry to reuse completed cache files".into(),
        HFError::Http { context } if matches!(context.status.as_u16(), 401 | 403) => "access denied; set HF_TOKEN or run `hf auth login`, and accept any gated-model license on Hugging Face".into(),
        HFError::Http { context } => format!("Hub returned HTTP {}; retry or check repository access", context.status),
        _ => "Hub client could not complete the download; check endpoint/configuration and retry".into(),
    };
    io::Error::other(message)
}

#[cfg(test)]
mod tests;

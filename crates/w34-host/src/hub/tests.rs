mod environment;

use super::*;
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    thread,
};

const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const OTHER_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

struct Fixture {
    cache: PathBuf,
    endpoint: String,
    token: String,
    offline: bool,
}

fn settings(root: &Path, extra: &[(&str, &str)]) -> Fixture {
    let get = |name| {
        extra
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    };
    Fixture {
        cache: root.join("hub"),
        endpoint: get("HF_ENDPOINT").unwrap_or("http://127.0.0.1:1").into(),
        token: get("HF_TOKEN").unwrap_or("fixture-token").into(),
        offline: get("HF_HUB_OFFLINE") == Some("1"),
    }
}

// Fixture requests use explicit SDK options, never the developer's credentials.
// Production environment handling is exercised separately in child processes.
fn resolve_remote(
    spec: &str,
    options: &HubOptions,
    role: ModelRole,
    fixture: &Fixture,
) -> io::Result<PathBuf> {
    super::resolve_remote(
        spec,
        &HubOptions {
            offline: options.offline || fixture.offline,
            ..options.clone()
        },
        role,
        HFClient::builder()
            .cache_dir(&fixture.cache)
            .endpoint(&fixture.endpoint)
            .token(&fixture.token),
        false,
    )
}

fn resolve_offline(repo: &Path, revision: &str, role: ModelRole) -> io::Result<PathBuf> {
    super::resolve_remote(
        "test/model",
        &HubOptions {
            revision: Some(revision.into()),
            offline: true,
        },
        role,
        HFClient::builder()
            .cache_dir(repo.parent().unwrap())
            .endpoint("http://127.0.0.1:1")
            .token("fixture-token"),
        false,
    )
}

fn config() -> Vec<u8> {
    serde_json::to_vec(&json!({"model_type":"qwen3_5", "hidden_size":2560,
        "num_hidden_layers":32, "num_attention_heads":20, "num_key_value_heads":4,
        "head_dim":128, "vocab_size":152064, "rms_norm_eps":1e-6,
        "layer_types":["linear_attention", "full_attention"]}))
    .unwrap()
}
fn weights() -> Vec<u8> {
    let header = br#"{"weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0, 0]);
    bytes
}
fn fixture_files() -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        ("config.json".into(), config()),
        ("tokenizer.json".into(), b"{}".to_vec()),
        ("model.safetensors".into(), weights()),
        ("head.safetensors".into(), weights()),
        ("preprocessor_config.json".into(), b"{}".to_vec()),
        ("chat_template.jinja".into(), b"{{ messages }}".to_vec()),
        ("model.bin".into(), b"do not download".to_vec()),
        (
            "alternative.safetensors".into(),
            b"do not download".to_vec(),
        ),
        (
            "modeling_custom.py".into(),
            b"raise RuntimeError('must not execute')".to_vec(),
        ),
        (
            "original/weights.safetensors".into(),
            b"do not download".to_vec(),
        ),
    ])
}
fn cached(root: &Path, sha: &str) -> PathBuf {
    let dir = root.join("hub/models--test--model/snapshots").join(sha);
    fs::create_dir_all(&dir).unwrap();
    for (name, data) in fixture_files() {
        if !name.contains('/') {
            fs::write(dir.join(name), data).unwrap();
        }
    }
    dir
}

#[test]
fn uppercase_sha_selects_the_same_offline_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = cached(tmp.path(), SHA);
    let s = settings(tmp.path(), &[]);
    let options = HubOptions {
        revision: Some(SHA.to_ascii_uppercase()),
        offline: true,
    };
    assert_eq!(
        resolve_remote("test/model", &options, ModelRole::Target, &s).unwrap(),
        dir
    );
}

#[test]
fn local_paths_and_invalid_ids_never_need_hub_configuration() {
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(
        resolve_model(
            tmp.path().to_str().unwrap(),
            &HubOptions::default(),
            ModelRole::Target
        )
        .unwrap(),
        tmp.path()
    );
    assert!(
        resolve_model(
            tmp.path().to_str().unwrap(),
            &HubOptions {
                revision: Some("main".into()),
                offline: false
            },
            ModelRole::Target
        )
        .is_err()
    );
    assert!(local_model("/missing/w34-model-directory", &HubOptions::default()).is_err());
    for id in [
        "../model",
        "user/../model",
        "https://huggingface.co/user/model",
        "user/model?token=secret",
        "user//model",
        "user/model\n",
    ] {
        assert!(repo_id(id).is_err(), "{id}");
    }
    assert_eq!(repo_id("models--test--model").unwrap(), "test/model");
    for name in [
        "../secret",
        "/etc/passwd",
        "model/../../secret",
        "a\\b",
        "*",
        "refs//main",
    ] {
        assert!(safe_path(name).is_err());
    }
    assert!(safe_path("refs/pr/12").is_ok());
    assert!(safe_revision("release/résumé+2").is_ok());
    for revision in [
        "../escape",
        "a/../../escape",
        "/absolute",
        "refs//main",
        "a\\b",
        "a.lock",
        "a..b",
    ] {
        assert!(safe_revision(revision).is_err());
    }
}

#[test]
fn offline_uses_exact_revision_not_an_arbitrary_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = cached(tmp.path(), SHA);
    cached(tmp.path(), OTHER_SHA);
    let s = settings(
        tmp.path(),
        &[
            ("HF_HUB_OFFLINE", "1"),
            ("HF_ENDPOINT", "http://127.0.0.1:1"),
        ],
    );
    assert!(
        resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s)
            .unwrap_err()
            .to_string()
            .contains("not cached")
    );
    let repo = s.cache.join("models--test--model");
    atomic_write(&repo.join("refs/main"), SHA.as_bytes()).unwrap();
    assert_eq!(
        resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s).unwrap(),
        dir
    );
    let options = HubOptions {
        revision: Some(OTHER_SHA.into()),
        offline: true,
    };
    assert_eq!(
        resolve_remote("test/model", &options, ModelRole::Target, &s)
            .unwrap()
            .file_name()
            .unwrap(),
        OTHER_SHA
    );
    atomic_write(&repo.join("refs/main"), b"../../secret").unwrap();
    assert!(resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s).is_err());
}

#[test]
fn offline_rejects_missing_indexed_shards_and_manifest_assets() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = cached(tmp.path(), SHA);
    let repo = tmp.path().join("hub/models--test--model");
    atomic_write(&repo.join("refs/main"), SHA.as_bytes()).unwrap();
    fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"w":"shards/missing.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        resolve_offline(&repo, "main", ModelRole::Target)
            .unwrap_err()
            .to_string()
            .contains("missing indexed")
    );
    fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
    let manifest = Manifest {
        version: 1,
        commit: SHA.into(),
        files: vec!["missing-template.jinja".into()],
    };
    atomic_write(
        &manifest_path(&repo, SHA),
        &serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(
        resolve_offline(&repo, "main", ModelRole::Target)
            .unwrap_err()
            .to_string()
            .contains("incomplete")
    );
}

type RecordedRequest = (String, String, Option<String>);

#[derive(Clone)]
struct MockState {
    files: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    missing: Arc<Mutex<BTreeSet<String>>>,
    denied: Arc<Mutex<bool>>,
    forbidden: Arc<Mutex<BTreeSet<String>>>,
}
struct HubServer {
    endpoint: String,
    state: MockState,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    worker: Option<thread::JoinHandle<()>>,
}
impl HubServer {
    fn new(files: BTreeMap<String, Vec<u8>>) -> Self {
        let state = MockState {
            files: Arc::new(Mutex::new(files)),
            requests: Arc::default(),
            missing: Arc::default(),
            denied: Arc::default(),
            forbidden: Arc::default(),
        };
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let app = Router::new()
            .fallback(any(handle))
            .with_state(state.clone());
        let worker = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                tokio::select! {
                    result = axum::serve(tokio::net::TcpListener::from_std(listener).unwrap(), app).into_future() => result.unwrap(),
                    _ = stopped => {},
                }
            });
        });
        Self {
            endpoint,
            state,
            stop: Some(stop),
            worker: Some(worker),
        }
    }
    fn gets(&self, suffix: &str) -> usize {
        self.state
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(method, path, _)| method == "GET" && path.ends_with(suffix))
            .count()
    }
}
impl Drop for HubServer {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
async fn handle(State(state): State<MockState>, request: Request<Body>) -> Response {
    let path = request.uri().path();
    let method = request.method().to_string();
    let auth = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    state
        .requests
        .lock()
        .unwrap()
        .push((method.clone(), path.into(), auth));
    if *state.denied.lock().unwrap() {
        return (StatusCode::FORBIDDEN, "private-body-do-not-log-token").into_response();
    }
    let files = state.files.lock().unwrap();
    if path.starts_with("/api/models/test/model/revision/") {
        let siblings: Vec<_> = files.keys().map(|name| json!({"rfilename":name})).collect();
        return axum::Json(json!({"id":"test/model", "sha":SHA, "siblings":siblings}))
            .into_response();
    }
    if path.starts_with(&format!("/api/models/test/model/tree/{SHA}")) {
        let entries: Vec<_> = files
            .iter()
            .map(|(name, data)| json!({"type":"file", "oid":SHA, "size":data.len(), "path":name}))
            .collect();
        return axum::Json(json!(entries)).into_response();
    }
    if let Some(name) = path.strip_prefix(&format!("/test/model/resolve/{SHA}/")) {
        if state.forbidden.lock().unwrap().contains(name) {
            return StatusCode::FORBIDDEN.into_response();
        }
        if state.missing.lock().unwrap().contains(name) {
            return StatusCode::NOT_FOUND.into_response();
        }
        if let Some(data) = files.get(name) {
            let etag = format!("\"{:x}\"", Sha256::digest(data));
            return Response::builder()
                .header("etag", etag)
                .header("x-repo-commit", SHA)
                .header("content-length", data.len())
                .body(Body::from(if method == "HEAD" {
                    Vec::new()
                } else {
                    data.clone()
                }))
                .unwrap();
        }
    }
    (StatusCode::NOT_FOUND, format!("unexpected path {path}")).into_response()
}

#[test]
fn downloads_pinned_selected_assets_reuses_cache_and_works_offline() {
    let tmp = tempfile::tempdir().unwrap();
    let server = HubServer::new(fixture_files());
    let s = settings(
        tmp.path(),
        &[
            ("HF_ENDPOINT", &server.endpoint),
            ("HF_TOKEN", "test-token"),
        ],
    );
    let options = HubOptions {
        revision: Some("release/v1".into()),
        offline: false,
    };
    let dir = resolve_remote("test/model", &options, ModelRole::Target, &s).unwrap();
    assert_eq!(dir.file_name().unwrap(), SHA);
    assert!(dir.join("head.safetensors").exists());
    assert!(dir.join("preprocessor_config.json").exists());
    assert!(!dir.join("model.bin").exists());
    assert!(!dir.join("alternative.safetensors").exists());
    assert!(!dir.join("modeling_custom.py").exists());
    let requests = server.state.requests.lock().unwrap().clone();
    assert!(
        requests
            .iter()
            .all(|(_, _, token)| token.as_deref() == Some("Bearer test-token"))
    );
    assert!(
        requests
            .iter()
            .filter(|(_, path, _)| path.contains("/resolve/"))
            .all(|(_, path, _)| path.contains(SHA))
    );
    let count = server.gets("model.safetensors");
    assert_eq!(
        resolve_remote("test/model", &options, ModelRole::Target, &s).unwrap(),
        dir
    );
    assert_eq!(server.gets("model.safetensors"), count);
    drop(server);
    assert_eq!(
        resolve_remote(
            "test/model",
            &HubOptions {
                offline: true,
                ..options
            },
            ModelRole::Target,
            &s
        )
        .unwrap(),
        dir
    );
}

#[test]
fn simultaneous_downloads_share_a_valid_standard_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let server = HubServer::new(fixture_files());
    let s = settings(tmp.path(), &[("HF_ENDPOINT", &server.endpoint)]);
    let paths = thread::scope(|scope| {
        let first = scope
            .spawn(|| resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s));
        let second = scope
            .spawn(|| resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s));
        (
            first.join().unwrap().unwrap(),
            second.join().unwrap().unwrap(),
        )
    });
    assert_eq!(paths.0, paths.1);
    assert_eq!(
        resolve_offline(
            &s.cache.join("models--test--model"),
            "main",
            ModelRole::Target
        )
        .unwrap(),
        paths.0
    );
}

#[test]
fn draft_download_does_not_require_its_own_tokenizer() {
    let tmp = tempfile::tempdir().unwrap();
    let mut assets = fixture_files();
    assets.remove("tokenizer.json");
    let mut body: serde_json::Value = serde_json::from_slice(&config()).unwrap();
    body["model_type"] = json!("qwen3");
    body["num_hidden_layers"] = json!(5);
    body["intermediate_size"] = json!(1024);
    body["layer_types"] = json!(vec!["full_attention"; 5]);
    let root = json!({"architectures":["DFlash2DraftModel"], "transformer_layer_config":body,
        "block_size":16, "target_layer_ids":[3,7,11,15,19], "conv_group_size":1,
        "conv_kernel_size":2, "selector_rank":32, "selector_top_k":4, "mask_token_id":100});
    assets.insert("config.json".into(), serde_json::to_vec(&root).unwrap());
    let server = HubServer::new(assets);
    let s = settings(tmp.path(), &[("HF_ENDPOINT", &server.endpoint)]);
    let dir = resolve_remote("test/model", &HubOptions::default(), ModelRole::Draft, &s).unwrap();
    assert!(!dir.join("tokenizer.json").exists());
    assert!(dir.join("model.safetensors").exists());
}

#[test]
fn missing_weight_response_is_not_success_and_retry_repairs_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let server = HubServer::new(fixture_files());
    server
        .state
        .missing
        .lock()
        .unwrap()
        .insert("model.safetensors".into());
    let s = settings(tmp.path(), &[("HF_ENDPOINT", &server.endpoint)]);
    let error =
        resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s).unwrap_err();
    assert!(
        error.to_string().contains("incomplete model snapshot"),
        "{error}"
    );
    assert!(!s.cache.join("models--test--model/refs/main").exists());
    server.state.missing.lock().unwrap().clear();
    let dir = resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s).unwrap();
    assert!(dir.join("model.safetensors").exists());
    assert!(s.cache.join("models--test--model/refs/main").exists());
}

#[test]
fn rejects_unsupported_architecture_before_weight_download() {
    let tmp = tempfile::tempdir().unwrap();
    let mut files = fixture_files();
    files.insert("config.json".into(), br#"{"model_type":"llama"}"#.to_vec());
    let server = HubServer::new(files);
    let s = settings(tmp.path(), &[("HF_ENDPOINT", &server.endpoint)]);
    let error =
        resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s).unwrap_err();
    assert!(error.to_string().contains("unsupported target"));
    assert_eq!(server.gets("model.safetensors"), 0);
}

#[test]
fn index_selects_only_named_shards_and_rejects_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let mut files = fixture_files();
    files.remove("model.safetensors");
    files.insert(
        "model.safetensors.index.json".into(),
        br#"{"weight_map":{"weight":"shards/part.safetensors"}}"#.to_vec(),
    );
    files.insert("shards/part.safetensors".into(), weights());
    let server = HubServer::new(files);
    let s = settings(tmp.path(), &[("HF_ENDPOINT", &server.endpoint)]);
    let dir = resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s).unwrap();
    assert!(dir.join("shards/part.safetensors").exists());
    fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"weight":"../escape.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        resolve_remote(
            "test/model",
            &HubOptions {
                offline: true,
                ..HubOptions::default()
            },
            ModelRole::Target,
            &s
        )
        .is_err()
    );
}

#[test]
fn gated_weights_explain_license_and_auth_even_when_metadata_is_public() {
    let tmp = tempfile::tempdir().unwrap();
    let server = HubServer::new(fixture_files());
    server
        .state
        .forbidden
        .lock()
        .unwrap()
        .insert("model.safetensors".into());
    let s = settings(tmp.path(), &[("HF_ENDPOINT", &server.endpoint)]);
    let message = resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("HF_TOKEN") && message.contains("gated-model"),
        "{message}"
    );
}

#[test]
fn gated_access_errors_are_actionable_without_echoing_server_body() {
    let tmp = tempfile::tempdir().unwrap();
    let server = HubServer::new(fixture_files());
    *server.state.denied.lock().unwrap() = true;
    let s = settings(
        tmp.path(),
        &[
            ("HF_ENDPOINT", &server.endpoint),
            ("HF_TOKEN", "test-secret"),
        ],
    );
    let message = resolve_remote("test/model", &HubOptions::default(), ModelRole::Target, &s)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("HF_TOKEN") && message.contains("gated-model"),
        "{message}"
    );
    assert!(!message.contains("test-secret") && !message.contains("private-body"));
}

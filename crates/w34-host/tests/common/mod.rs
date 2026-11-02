#![allow(dead_code)]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::Arc,
    time::Duration,
};
#[cfg(feature = "cuda")]
use w34_host::server::{OpenAiServerState, router};

/// Returns `true` when `CI` or `W34_STRICT_TESTS` demands zero skipped fixtures/GPUs.
pub fn strict_test_env() -> bool {
    for var in ["W34_STRICT_TESTS", "CI"] {
        if let Ok(val) = std::env::var(var) {
            let trimmed = val.trim();
            if trimmed == "1" || trimmed.eq_ignore_ascii_case("true") {
                return true;
            }
        }
    }
    false
}

/// Resolves an optional cached HF checkpoint; logs `[SKIP ...]` locally or panics in strict/CI runs.
pub fn require_or_skip_cached_model(repo_id: &str, test_name: &str) -> Option<PathBuf> {
    match w34_host::loader::resolve_hf_model_dir(repo_id) {
        Ok(dir) => Some(dir),
        Err(err) => {
            assert!(
                !strict_test_env(),
                "[{test_name}] required cached HF checkpoint '{repo_id}' missing in strict/CI mode: {err}"
            );
            eprintln!("[SKIP {test_name}] cached HF checkpoint '{repo_id}' unavailable: {err}");
            None
        }
    }
}

/// Checks for `min_gpus` visible CUDA devices; logs `[SKIP ...]` locally or panics in strict/CI runs.
#[cfg(feature = "cuda")]
pub fn require_or_skip_gpus(min_gpus: usize, test_name: &str) -> bool {
    let count = w34_host::cuda::CudaDevice::count().unwrap_or(0).max(0) as usize;
    if count >= min_gpus {
        return true;
    }
    assert!(
        !strict_test_env(),
        "[{test_name}] requires >= {min_gpus} visible CUDA devices (found {count}) in strict/CI mode"
    );
    eprintln!("[SKIP {test_name}] requires >= {min_gpus} visible CUDA devices (found {count})");
    false
}

#[cfg(feature = "cuda")]
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Exercise the real HTTP transport, decoding chunked bodies before legacy SSE assertions.
#[cfg(feature = "cuda")]
pub fn raw_request(
    state: &Arc<OpenAiServerState>,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let state = state.clone();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = std::thread::spawn(move || {
        runtime().block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, router(state))
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        })
    });
    let mut socket = TcpStream::connect(address).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(180)))
        .unwrap();
    let body = body.unwrap_or("");
    write!(socket, "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut response = Vec::new();
    let result = socket.read_to_end(&mut response);
    let _ = stop.send(());
    server.join().unwrap();
    result.unwrap();
    let split = response.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = std::str::from_utf8(&response[..split]).unwrap();
    let mut body = &response[split + 4..];
    let mut decoded = Vec::new();
    if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        loop {
            let end = body.windows(2).position(|w| w == b"\r\n").unwrap();
            let size = usize::from_str_radix(
                std::str::from_utf8(&body[..end])
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap(),
                16,
            )
            .unwrap();
            body = &body[end + 2..];
            if size == 0 {
                break;
            }
            decoded.extend_from_slice(&body[..size]);
            assert_eq!(&body[size..size + 2], b"\r\n");
            body = &body[size + 2..];
        }
    } else {
        decoded.extend_from_slice(body);
    }
    format!("{head}\r\n\r\n{}", String::from_utf8(decoded).unwrap())
}

#[cfg(feature = "cuda")]
pub fn request(
    state: &Arc<OpenAiServerState>,
    method: &str,
    path: &str,
    body: serde_json::Value,
) -> (String, serde_json::Value) {
    let response = raw_request(state, method, path, Some(&body.to_string()));
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    (head.to_owned(), serde_json::from_str(body).unwrap())
}

#[cfg(feature = "cuda")]
pub fn shutdown(state: &Arc<OpenAiServerState>) {
    runtime().block_on(state.worker.shutdown()).unwrap();
}

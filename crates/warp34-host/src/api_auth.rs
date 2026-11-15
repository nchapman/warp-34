//! Optional API authentication. Layer only onto protected routes; no path is exempt here.

use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// A validated API credential stored only as a fixed-length digest.
/// Deliberately does not implement `Debug`.
#[derive(Clone)]
pub struct ApiKey {
    digest: [u8; 32],
}

impl ApiKey {
    /// Reject empty keys and anything other than visible ASCII, without echoing input.
    pub fn new(value: String) -> Result<Self, String> {
        if !valid_token(&value) {
            return Err("API key must be nonempty visible ASCII without whitespace".into());
        }
        Ok(Self {
            digest: Sha256::digest(value.as_bytes()).into(),
        })
    }
}

fn valid_token(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_graphic())
}

fn credential(headers: &HeaderMap) -> Option<&str> {
    if headers.get_all(header::AUTHORIZATION).iter().count() > 1
        || headers.get_all("x-api-key").iter().count() > 1
    {
        return None;
    }
    let token = if let Some(value) = headers.get(header::AUTHORIZATION) {
        let (scheme, token) = value.to_str().ok()?.split_once(' ')?;
        if !scheme.eq_ignore_ascii_case("Bearer") {
            return None;
        }
        // RFC 6750 allows one or more SP between scheme and credential.
        token.trim_start_matches(' ')
    } else {
        headers.get("x-api-key")?.to_str().ok()?
    };
    valid_token(token).then_some(token)
}

/// Authenticate without modifying credentials (Responses state partitioning needs them).
pub async fn authenticate(
    State(key): State<Option<ApiKey>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(key) = key else {
        return next.run(request).await;
    };
    if let Some(token) = credential(request.headers()) {
        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        if bool::from(key.digest.ct_eq(&digest)) {
            return next.run(request).await;
        }
    }
    let path = request.uri().path();
    let body = if path == "/v1/messages" || path.starts_with("/v1/messages/") {
        serde_json::json!({"type": "error", "error": {
            "type": "authentication_error", "message": "Invalid API key"
        }})
    } else {
        serde_json::json!({"error": {
            "message": "Invalid API key", "type": "authentication_error",
            "param": null, "code": "invalid_api_key"
        }})
    };
    (
        StatusCode::UNAUTHORIZED,
        [
            (header::WWW_AUTHENTICATE, "Bearer"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        Json(body),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, middleware, routing::any};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    const SECRET: &str = "test-secret-DoNotEcho";

    struct Server {
        url: String,
        hits: Arc<AtomicUsize>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn server(enabled: bool) -> Server {
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let app = Router::new().fallback(any(move |request: Request| {
            counter.fetch_add(1, Ordering::SeqCst);
            async move {
                // Echo credentials only on success to verify they are not rewritten.
                Json(serde_json::json!({
                    "authorization": request.headers().get("authorization").and_then(|v| v.to_str().ok()),
                    "x-api-key": request.headers().get("x-api-key").and_then(|v| v.to_str().ok()),
                    "uri": request.uri().to_string(),
                }))
            }
        })).layer(middleware::from_fn_with_state(
            enabled.then(|| ApiKey::new(SECRET.into()).unwrap()), authenticate,
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Server { url, hits, task }
    }

    #[test]
    fn validates_without_echoing_secrets() {
        for input in [
            "",
            " ",
            "secret value",
            "secret\t",
            "secret\n",
            "secret\x7f",
            "sécret",
            "secret\0",
        ] {
            let error = ApiKey::new(input.into()).err().expect("must reject");
            assert_eq!(
                error,
                "API key must be nonempty visible ASCII without whitespace"
            );
            assert!(!error.contains("secret"));
        }
        assert!(ApiKey::new(SECRET.into()).is_ok());
    }

    #[tokio::test]
    async fn enabled_credentials_and_precedence_over_http() {
        let server = server(true).await;
        let client = reqwest::Client::new();
        let bearer = format!("Bearer {SECRET}");
        let mixed = format!("bEaReR {SECRET}");
        let spaced = format!("Bearer   {SECRET}");
        for headers in [
            vec![("Authorization", bearer.as_str())],
            vec![("X-API-Key", SECRET)],
            vec![("aUtHoRiZaTiOn", mixed.as_str())],
            vec![("Authorization", spaced.as_str())],
            vec![("Authorization", bearer.as_str()), ("x-api-key", SECRET)],
            vec![("Authorization", bearer.as_str()), ("x-api-key", "wrong")],
        ] {
            let mut request = client.post(format!("{}/v1/responses?keep=yes", server.url));
            for (name, value) in &headers {
                request = request.header(*name, *value);
            }
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body: serde_json::Value = response.json().await.unwrap();
            for (name, value) in headers {
                assert_eq!(body[name.to_ascii_lowercase()], value);
            }
            assert_eq!(body["uri"], "/v1/responses?keep=yes");
        }
        assert_eq!(server.hits.load(Ordering::SeqCst), 6);
    }

    #[tokio::test]
    async fn rejects_before_handler_over_http() {
        let server = server(true).await;
        let client = reqwest::Client::new();
        let bearer = format!("Bearer {SECRET}");
        let joined = format!("Bearer {SECRET}, Bearer {SECRET}");
        let cases = vec![
            vec![],
            vec![("x-api-key", "wrong")],
            vec![("Authorization", "Bearer wrong"), ("x-api-key", SECRET)],
            vec![("Authorization", "Basic wrong"), ("x-api-key", SECRET)],
            vec![("Authorization", ""), ("x-api-key", SECRET)],
            vec![("Authorization", "Bearer")],
            vec![("Authorization", "Bearer ")],
            vec![("Authorization", "Bearer\twrong")],
            vec![("Authorization", "Bearer wrong extra")],
            vec![("Authorization", joined.as_str())],
            vec![
                ("Authorization", bearer.as_str()),
                ("Authorization", bearer.as_str()),
            ],
            vec![("x-api-key", SECRET), ("x-api-key", SECRET)],
            vec![
                ("Authorization", bearer.as_str()),
                ("x-api-key", SECRET),
                ("x-api-key", "wrong"),
            ],
        ];
        for path in [
            "/v1/responses",
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/health",
            "/dashboard",
            "/metrics",
        ] {
            for headers in &cases {
                let mut request = client.post(format!(
                    "{}{path}?api_key={SECRET}&x-api-key={SECRET}&access_token={SECRET}",
                    server.url
                ));
                for (name, value) in headers {
                    request = request.header(*name, *value);
                }
                let response = request.send().await.unwrap();
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{headers:?}");
                assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
                assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
                let text = response.text().await.unwrap();
                assert!(!text.contains(SECRET));
                assert!(!text.contains("wrong"));
                let body: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(body["error"]["type"], "authentication_error");
                if path.starts_with("/v1/messages") {
                    assert_eq!(body["type"], "error");
                } else {
                    assert_eq!(body["error"]["code"], "invalid_api_key");
                    assert!(body["error"]["param"].is_null());
                }
                assert_eq!(server.hits.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[tokio::test]
    async fn disabled_passes_missing_invalid_and_duplicate_credentials() {
        let server = server(false).await;
        let client = reqwest::Client::new();
        for headers in [
            vec![],
            vec![("authorization", "Basic wrong")],
            vec![("x-api-key", "one"), ("x-api-key", "two")],
        ] {
            let mut request = client.get(format!("{}/v1/messages", server.url));
            for (name, value) in headers {
                request = request.header(name, value);
            }
            assert_eq!(request.send().await.unwrap().status(), StatusCode::OK);
        }
        assert_eq!(server.hits.load(Ordering::SeqCst), 3);
    }
}

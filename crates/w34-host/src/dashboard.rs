//! Read-only React dashboard, embedded at compile time. Build assets with
//! `mise run dashboard-build` before invoking Cargo directly.
use axum::{
    Router,
    extract::Path,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use include_dir::{Dir, include_dir};

static ASSETS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../../dashboard/dist");
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; style-src-attr 'unsafe-inline'; img-src 'self'; connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

pub fn router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(index))
        .route("/favicon.svg", get(favicon))
        .route("/assets/{*path}", get(asset))
}

async fn index() -> Response {
    embedded_file("index.html", false)
}

async fn favicon() -> Response {
    embedded_file("favicon.svg", false)
}

async fn asset(Path(path): Path<String>) -> Response {
    // Lookup is exclusively inside the embedded bundle, never the filesystem.
    if path.split('/').any(|part| part == ".." || part == ".") {
        return StatusCode::NOT_FOUND.into_response();
    }
    embedded_file(&format!("assets/{path}"), true)
}

fn embedded_file(path: &str, immutable: bool) -> Response {
    let Some(file) = ASSETS.get_file(path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let content_type = match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    (
        [
            (header::CONTENT_TYPE, content_type),
            (
                header::CACHE_CONTROL,
                if immutable {
                    "public, max-age=31536000, immutable"
                } else {
                    "no-cache"
                },
            ),
            (header::CONTENT_SECURITY_POLICY, CSP),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::REFERRER_POLICY, "no-referrer"),
        ],
        file.contents(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embedded_dashboard_and_assets_serve_without_filesystem() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async { axum::serve(listener, router()).await.unwrap() });
        let client = reqwest::Client::new();
        let response = client.get(&base).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");
        assert_eq!(
            response.headers()[header::X_CONTENT_TYPE_OPTIONS],
            "nosniff"
        );
        assert!(
            response.headers()[header::CONTENT_SECURITY_POLICY]
                .to_str()
                .unwrap()
                .contains("frame-ancestors 'none'")
        );
        let html = response.text().await.unwrap();
        assert!(html.contains("warp-34"));
        // Vite's real output is the contract: follow every generated asset reference.
        let mut assets = 0;
        for url in html.split('"').filter(|part| part.starts_with("/assets/")) {
            let response = client.get(format!("{base}{url}")).send().await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{url}");
            assert!(
                response.headers()[header::CACHE_CONTROL]
                    .to_str()
                    .unwrap()
                    .contains("immutable")
            );
            assert!(!response.bytes().await.unwrap().is_empty());
            assets += 1;
        }
        assert!(assets >= 2, "must embed the production JS and CSS");
        for path in [
            "/assets/missing.js",
            "/assets/package.json",
            "/not-a-dashboard-route",
        ] {
            assert_eq!(
                client
                    .get(format!("{base}{path}"))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                StatusCode::NOT_FOUND
            );
        }
        assert_eq!(
            client
                .get(format!("{base}/favicon.svg"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        server.abort();
    }
}

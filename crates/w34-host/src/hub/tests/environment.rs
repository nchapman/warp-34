//! Exercise the production entrypoint with real SDK environment handling.
//! Child processes isolate credentials/configuration from parallel tests.
use super::*;
use std::process::Command;

fn child(root: &Path, endpoint: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        "hub::tests::environment::sdk_configuration_child",
        "--nocapture",
    ]);
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("HF_")
            || name.starts_with("HUGGINGFACE_")
            || name.starts_with("HUGGING_FACE_")
            || matches!(name.as_ref(), "HOME" | "XDG_CACHE_HOME")
        {
            command.env_remove(key);
        }
    }
    command
        .env("W34_HUB_TEST_CHILD", "1")
        .env("HOME", root)
        .env("HF_HOME", root)
        .env("HF_ENDPOINT", endpoint)
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .env("NO_PROXY", "127.0.0.1,localhost,::1");
    command
}

fn run(command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn sdk_configuration_child() {
    if std::env::var_os("W34_HUB_TEST_CHILD").is_none() {
        return;
    }
    let dir = resolve_model("test/model", &HubOptions::default(), ModelRole::Target).unwrap();
    assert!(dir.join("model.safetensors").is_file());
    assert_eq!(
        dir,
        cache_dir().join("models--test--model/snapshots").join(SHA)
    );
}

#[test]
fn sdk_owns_saved_login_token_precedence_and_opt_out() {
    let tmp = tempfile::tempdir().unwrap();
    let server = HubServer::new(fixture_files());
    fs::write(tmp.path().join("token"), "saved-token\n").unwrap();
    let custom_token = tmp.path().join("custom-token");
    fs::write(&custom_token, "custom-token\n").unwrap();
    for (vars, expected) in [
        (vec![], Some("Bearer saved-token")),
        (
            vec![("HF_TOKEN_PATH", custom_token.to_str().unwrap())],
            Some("Bearer custom-token"),
        ),
        (
            vec![
                ("HF_TOKEN", "env-token"),
                ("HF_TOKEN_PATH", custom_token.to_str().unwrap()),
            ],
            Some("Bearer env-token"),
        ),
        (
            vec![
                ("HF_TOKEN", "env-token"),
                ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"),
            ],
            None,
        ),
    ] {
        server.state.requests.lock().unwrap().clear();
        run(child(tmp.path(), &server.endpoint).envs(vars));
        let requests = server.state.requests.lock().unwrap();
        assert!(!requests.is_empty());
        assert!(
            requests
                .iter()
                .all(|(_, _, auth)| auth.as_deref() == expected)
        );
    }
}

#[test]
fn sdk_cache_environment_and_offline_lookup_work_without_a_home() {
    let tmp = tempfile::tempdir().unwrap();
    let server = HubServer::new(fixture_files());
    let cache = tmp.path().join("explicit-cache");
    run(child(tmp.path(), &server.endpoint)
        .env_remove("HF_HOME")
        .env_remove("HOME")
        .env("HF_HUB_CACHE", &cache)
        .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"));
    assert!(cache.join("models--test--model/refs/main").exists());

    // Prove both that a reachable Hub receives no requests and that an
    // unreachable Hub/unreadable token path cannot prevent offline loading.
    server.state.requests.lock().unwrap().clear();
    for endpoint in [server.endpoint.as_str(), "http://127.0.0.1:1"] {
        run(child(tmp.path(), endpoint)
            .env_remove("HF_HOME")
            .env_remove("HOME")
            .env("HF_HUB_CACHE", &cache)
            .env("HF_HUB_OFFLINE", "1")
            .env("HF_TOKEN_PATH", tmp.path()));
    }
    assert!(server.state.requests.lock().unwrap().is_empty());

    let xdg = tmp.path().join("xdg");
    run(child(tmp.path(), &server.endpoint)
        .env_remove("HF_HOME")
        .env("XDG_CACHE_HOME", &xdg)
        .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"));
    assert!(
        xdg.join("huggingface/hub/models--test--model/refs/main")
            .exists()
    );
}

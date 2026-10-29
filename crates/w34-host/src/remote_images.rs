//! Opt-in remote image transport. No credentials, proxies, redirects, or
//! non-public destinations; DNS answers are checked at the actual connect seam.

use std::{net::IpAddr, time::Duration};
use ureq::{
    Agent,
    config::Config,
    http::Uri,
    unversioned::{
        resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver},
        transport::{DefaultConnector, NextTimeout},
    },
};

pub const MAX_REMOTE_IMAGE_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const REMOTE_IMAGE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug)]
struct PublicResolver;

impl Resolver for PublicResolver {
    fn resolve(
        &self,
        uri: &Uri,
        config: &Config,
        timeout: NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        let addresses = DefaultResolver::default().resolve(uri, config, timeout)?;
        // Fail the entire answer, not just the first address, to avoid failover
        // into a private network. The connector consumes these exact addresses.
        if addresses.iter().any(|address| !is_public(address.ip())) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "remote images require public Internet addresses",
            )
            .into());
        }
        Ok(addresses)
    }
}

fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_broadcast()
                || ip.is_documentation()
                || a == 0
                || a >= 224
                || (a == 100 && (64..=127).contains(&b))
                || (a == 198 && (18..=19).contains(&b))
                || (a == 192 && b == 0 && c == 0))
        }
        IpAddr::V6(ip) => {
            let s = ip.segments();
            // Only global unicast; exclude special-protocol, documentation and
            // 6to4 ranges (the latter can embed a non-public IPv4 destination).
            s[0] & 0xe000 == 0x2000
                && !(s[0] == 0x2001 && (s[1] <= 0x01ff || s[1] == 0x0db8))
                && s[0] != 0x2002
                && !(s[0] == 0x3fff && s[1] & 0xf000 == 0)
        }
    }
}

fn config(timeout: Duration) -> Config {
    Agent::config_builder()
        .proxy(None) // Override environment proxies so address filtering cannot be bypassed.
        .max_redirects(0)
        .timeout_global(Some(timeout))
        .timeout_resolve(Some(Duration::from_secs(3)))
        .timeout_connect(Some(Duration::from_secs(3)))
        .max_response_header_size(16 * 1024)
        .build()
}

fn validate_url(url: &str) -> Result<(), String> {
    if url.len() > 8192 {
        return Err("remote image URL exceeds 8192 bytes".into());
    }
    let uri: Uri = url.parse().map_err(|_| "invalid remote image URL")?;
    if !matches!(uri.scheme_str(), Some("http" | "https")) || uri.host().is_none() {
        return Err("remote image URL must use HTTP or HTTPS".into());
    }
    if uri.authority().is_some_and(|a| a.as_str().contains('@')) {
        return Err("credentials are not permitted in remote image URLs".into());
    }
    Ok(())
}

fn fetch_with_agent(agent: &Agent, url: &str, byte_limit: u64) -> Result<Vec<u8>, String> {
    validate_url(url)?;
    let mut response = agent
        .get(url)
        .header("Accept", "image/png, image/jpeg, image/x-portable-pixmap")
        .header("Accept-Encoding", "identity")
        .call()
        .map_err(|e| format!("remote image request failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "remote image HTTP status {} (redirects are not followed)",
            response.status()
        ));
    }
    if response
        .headers()
        .get("content-encoding")
        .is_some_and(|v| !v.as_bytes().eq_ignore_ascii_case(b"identity"))
    {
        return Err("compressed HTTP image responses are not supported".into());
    }
    if response
        .body()
        .content_length()
        .is_some_and(|len| len > byte_limit)
    {
        return Err(format!("remote image exceeds {byte_limit} byte limit"));
    }
    // ureq's limit is exclusive: reserve one sentinel byte so an exactly-at-cap
    // payload succeeds while chunked/unknown-length bodies cannot exceed the cap.
    let bytes = response
        .body_mut()
        .with_config()
        .limit(byte_limit + 1)
        .read_to_vec()
        .map_err(|e| format!("remote image body read failed: {e}"))?;
    if bytes.len() as u64 > byte_limit {
        return Err(format!("remote image exceeds {byte_limit} byte limit"));
    }
    Ok(bytes)
}

pub fn fetch_image_bytes(url: &str, timeout: Duration) -> Result<Vec<u8>, String> {
    if timeout.is_zero() {
        return Err("remote image request deadline exceeded".into());
    }
    let agent = Agent::with_parts(
        config(timeout.min(REMOTE_IMAGE_TIMEOUT)),
        DefaultConnector::default(),
        PublicResolver,
    );
    fetch_with_agent(&agent, url, MAX_REMOTE_IMAGE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::TcpListener,
    };

    fn exchange(response: Vec<u8>, limit: u64) -> Result<Vec<u8>, String> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/image", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = [0; 4096];
            let _ = socket.read(&mut request);
            let _ = socket.write_all(&response);
        });
        // Only tests bypass public-address policy; all transport controls match production.
        let agent = Agent::new_with_config(config(Duration::from_secs(2)));
        let result = fetch_with_agent(&agent, &url, limit);
        server.join().unwrap();
        result
    }

    #[test]
    fn public_address_policy_rejects_local_and_special_ranges() {
        for address in [
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "172.16.0.1",
            "192.168.1.1",
            "192.0.0.1",
            "192.0.2.1",
            "198.18.0.1",
            "224.0.0.1",
            "255.255.255.255",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "fe80::1",
            "ff02::1",
            "64:ff9b::a00:1",
            "2002:7f00:1::",
            "2001:db8::1",
            "3fff::1",
        ] {
            assert!(!is_public(address.parse().unwrap()), "{address}");
        }
        for address in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            assert!(is_public(address.parse().unwrap()), "{address}");
        }
        assert!(
            fetch_image_bytes("http://127.0.0.1:1/image", REMOTE_IMAGE_TIMEOUT)
                .unwrap_err()
                .contains("public Internet")
        );
        assert!(
            fetch_image_bytes("http://localhost:1/image", REMOTE_IMAGE_TIMEOUT)
                .unwrap_err()
                .contains("public Internet")
        );
    }

    #[test]
    fn byte_caps_status_redirects_encoding_and_truncation() {
        assert_eq!(
            exchange(
                b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc".to_vec(),
                3
            )
            .unwrap(),
            b"abc"
        );
        assert_eq!(
            exchange(
                b"HTTP/1.1 200 OK\r\nContent-Encoding: Identity\r\nContent-Length: 3\r\n\r\nabc"
                    .to_vec(),
                3
            )
            .unwrap(),
            b"abc"
        );
        for response in [
            b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nabcd".as_slice(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nabcd",
            b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/private\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 3\r\n\r\nabc",
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\na",
        ] {
            assert!(exchange(response.to_vec(), 3).is_err(), "{response:?}");
        }
        for url in [
            "file:///etc/passwd",
            "http://user:password@example.com/x",
            "ftp://example.com/image",
        ] {
            assert!(validate_url(url).is_err());
        }
    }

    #[test]
    fn expired_request_budget_prevents_another_fetch() {
        let error = crate::vision::parse_image_url_with_deadline(
            "http://127.0.0.1:1/image",
            true,
            std::time::Instant::now(),
        )
        .unwrap_err();
        assert!(error.contains("deadline exceeded"), "{error}");
    }

    #[test]
    fn body_stall_obeys_total_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/image", listener.local_addr().unwrap());
        let (release, hold) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = [0; 4096];
            let _ = socket.read(&mut request);
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\na")
                .unwrap();
            let _ = hold.recv_timeout(Duration::from_secs(3));
        });
        let agent = Agent::new_with_config(config(Duration::from_millis(200)));
        let start = std::time::Instant::now();
        let error = fetch_with_agent(&agent, &url, 3).unwrap_err();
        assert!(error.contains("timeout"), "{error}");
        assert!(start.elapsed() < Duration::from_secs(2));
        let _ = release.send(());
        server.join().unwrap();
    }

    #[test]
    fn real_png_and_jpeg_pixels_survive_http_transport() {
        let image = image::RgbImage::from_pixel(32, 32, image::Rgb([200, 80, 30]));
        for format in [image::ImageFormat::Png, image::ImageFormat::Jpeg] {
            let mut bytes = std::io::Cursor::new(Vec::new());
            image.write_to(&mut bytes, format).unwrap();
            let bytes = bytes.into_inner();
            let mut response =
                format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", bytes.len()).into_bytes();
            response.extend_from_slice(&bytes);
            let fetched = exchange(response, MAX_REMOTE_IMAGE_BYTES).unwrap();
            assert_eq!(fetched, bytes);
            let actual = crate::vision::decode_image_bytes_to_rgb(&fetched, None, true).unwrap();
            let expected = crate::vision::decode_image_bytes_to_rgb(&bytes, None, true).unwrap();
            assert_eq!(actual, expected);
            assert_eq!((actual.1, actual.2), (32, 32));
        }
    }
}

use w34_host::vision::{decode_image_bytes_to_rgb, parse_image_url_to_rgb};

#[test]
fn shared_decoder_preserves_the_larger_local_file_budget() {
    // A valid local PPM between the HTTP (16 MiB) and CLI (64 MiB) caps.
    // HTTP ingress enforces its own smaller budget before calling this decoder.
    let (w, h) = (4096, 1536);
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.resize(ppm.len() + w * h * 3, 42);
    let (rgb, actual_h, actual_w) = decode_image_bytes_to_rgb(&ppm, None, true).unwrap();
    assert_eq!((actual_h, actual_w, rgb.len()), (h, w, w * h * 3));
    assert!(rgb.iter().all(|&v| v == 42));
}

#[test]
fn rejects_oversized_http_payload_bad_dimensions_and_corrupt_bytes() {
    // 1. Empty, unknown-format (require_known_format=true), and truncated/oversized P6 payloads fail cleanly.
    assert!(decode_image_bytes_to_rgb(&[], None, true).is_err());
    assert!(decode_image_bytes_to_rgb(b"not an image", None, true).is_err());
    assert!(decode_image_bytes_to_rgb(b"P6\n4096 4096\n255\n\x00", None, true).is_err());
    assert!(decode_image_bytes_to_rgb(b"P6\n8192 8192\n255\n", None, true).is_err());

    // 2. HTTP data URI exceeding the 16 MiB HTTP ingress cap is rejected before decoding.
    let oversized_uri = format!(
        "data:image/png;base64,{}",
        "A".repeat(22 * 1024 * 1024 + 64)
    );
    assert!(parse_image_url_to_rgb(&oversized_uri).is_err());
}

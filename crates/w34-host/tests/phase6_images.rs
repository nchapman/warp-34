use w34_host::vision::{load_cli_image_to_rgb, parse_image_url_to_rgb};

#[test]
fn remote_images_are_not_silently_replaced_by_synthetic_pixels() {
    for url in [
        "http://example.invalid/image.png",
        "https://example.invalid/image.jpg",
    ] {
        assert!(
            parse_image_url_to_rgb(url).is_err(),
            "HTTP API accepted {url} without opt-in"
        );
        assert!(
            load_cli_image_to_rgb(url).is_err(),
            "CLI accepted {url} without opt-in"
        );
    }
}

#[test]
fn multimodal_requests_use_operator_policy_not_client_opt_in() {
    let messages = serde_json::json!([{"role": "user", "allow_remote_images": true,
        "content": [{"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/image.png"}}]}]);
    let disabled = w34_host::server::parse_openai_messages_multimodal(&messages).unwrap_err();
    assert!(disabled.contains("disabled"), "{disabled}");
    let forbidden =
        w34_host::server::parse_openai_messages_with_image_policy(&messages, true).unwrap_err();
    assert!(forbidden.contains("public Internet"), "{forbidden}");
    assert!(
        w34_host::vision::load_cli_image_with_policy("http://127.0.0.1:1/image", true)
            .unwrap_err()
            .contains("public Internet")
    );
}

#[test]
fn image_count_is_checked_before_any_network_or_decoding() {
    let parts = vec![
        serde_json::json!({"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/image"}});
        5
    ];
    let messages = serde_json::json!([{"role":"user", "content":parts}]);
    let error =
        w34_host::server::parse_openai_messages_with_image_policy(&messages, true).unwrap_err();
    assert!(error.contains("at most 4 images"), "{error}");
}

#[test]
fn oversized_compressed_images_are_rejected_before_rgb_allocation() {
    let image = image::GrayImage::new(8193, 2048);
    let mut png = std::io::Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png).unwrap();
    let result = w34_host::vision::decode_image_bytes_to_rgb(png.get_ref(), None, true);
    assert!(result.is_err(), "oversized source image was decoded");
}

#[test]
fn migrated_oracle_fixture_preserves_original_pixels() {
    let (rgb, h, w) = parse_image_url_to_rgb("rgb://https://w34.test/sample.ppm").unwrap();
    assert_eq!((h, w), (64, 64));
    // Original HTTP-fixture bytes, before HTTP URLs became real network inputs.
    assert_eq!(
        &rgb[..16],
        &[
            28, 156, 27, 154, 26, 153, 24, 151, 23, 150, 21, 149, 20, 147, 18, 146
        ]
    );
}

#[test]
fn explicit_rgb_fixture_scheme_is_offline_and_deterministic() {
    let first = parse_image_url_to_rgb("rgb://fixture/a").unwrap();
    assert_eq!(first, parse_image_url_to_rgb("rgb://fixture/a").unwrap());
    assert_ne!(
        first.0,
        parse_image_url_to_rgb("rgb://distinct-image").unwrap().0
    );
}

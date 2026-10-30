use w34_host::vision::decode_image_bytes_to_rgb;

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

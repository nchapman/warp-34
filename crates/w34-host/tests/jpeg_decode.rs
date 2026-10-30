use sha2::{Digest, Sha256};
use w34_host::vision::decode_image_bytes_to_rgb;

// Pillow 12.3.0 / libjpeg-turbo 3.1.4.1:
// Image.open(path).convert("RGB").tobytes(), hashed with hashlib.sha256.
#[test]
fn pillow_rgb_oracle() {
    for (bytes, expected) in [
        (
            include_bytes!("../../../tools/oracle/fixtures/coco_sample.png").as_slice(),
            "8050ada414034a490fc2bd6ed13a8d78b7d2ce7d896ba4fcd4651317b9471efe",
        ),
        (
            include_bytes!("../../../tools/oracle/fixtures/coco_sample.jpg").as_slice(),
            "4fa798e60eb66d35db3c855706bfbab379625b13801a26cb9737a7eda2d5adaf",
        ),
    ] {
        let (rgb, h, w) = decode_image_bytes_to_rgb(bytes, None, true).unwrap();
        assert_eq!((h, w, rgb.len()), (480, 640, 480 * 640 * 3));
        assert_eq!(format!("{:x}", Sha256::digest(&rgb)), expected);
    }
}

#[test]
fn jpeg_bounds_and_corruption() {
    let jpeg = include_bytes!("../../../tools/oracle/fixtures/coco_sample.jpg");
    for bytes in [&b"\xff\xd8\xff"[..], &jpeg[..jpeg.len() / 2]] {
        assert!(decode_image_bytes_to_rgb(bytes, Some("image/jpeg"), true).is_err());
    }
    let oversized = vec![0; 64 * 1024 * 1024 + 1];
    assert!(
        decode_image_bytes_to_rgb(&oversized, Some("image/jpeg"), true)
            .unwrap_err()
            .contains("64 MiB")
    );

    // Locate the baseline SOF in this fixture and mutate only dimensions.
    let sof = jpeg.windows(2).position(|b| b == [0xff, 0xc0]).unwrap();
    for (h, w) in [(0u16, 640u16), (480, 16385), (4097, 4096)] {
        let mut bytes = jpeg.to_vec();
        bytes[sof + 5..sof + 7].copy_from_slice(&h.to_be_bytes());
        bytes[sof + 7..sof + 9].copy_from_slice(&w.to_be_bytes());
        assert!(
            decode_image_bytes_to_rgb(&bytes, None, true).is_err(),
            "{w}x{h}"
        );
    }
    // Errors must release the serialization lock and native handle.
    assert!(decode_image_bytes_to_rgb(jpeg, None, true).is_ok());
}

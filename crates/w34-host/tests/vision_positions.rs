// Keep the exact decimal values emitted by the FP16 oracle, widened to FP32.
#![allow(clippy::excessive_precision)]
// Oracle: installed vLLM qwen3_vl.triton_pos_embed_interpolate, CUDA_VISIBLE_DEVICES=1.
// x = (((torch.arange(27, device="cuda") * 137) % 509 - 254) / 256).half().reshape(9,3)
// y = triton_pos_embed_interpolate(x, 1, 4, 6, 3, 2, torch.float16)
// Exact FP16 outputs, widened to FP32; no model load and no tolerance.
use w34_host::vision::interpolate_pos_embed_and_rot_ids;
#[test]
fn native_fp16_positions() {
    let weights: Vec<f32> = (0..27)
        .map(|i| ((i * 137 % 509) as f32 - 254.0) / 256.0)
        .collect();
    let expected: &[f32] = &[
        -0.9921875,
        -0.45703125,
        0.078125,
        -0.349853515625,
        -0.60986328125,
        -0.074951171875,
        -0.43212890625,
        0.102783203125,
        0.6376953125,
        -0.320068359375,
        -0.05029296875,
        0.48486328125,
        0.292236328125,
        -0.76318359375,
        -0.22802734375,
        0.53662109375,
        -0.5185546875,
        -0.381103515625,
        -0.208251953125,
        -0.203369140625,
        0.331787109375,
        -0.2288818359375,
        -0.223876953125,
        0.1785888671875,
        0.383544921875,
        0.12353515625,
        -0.5341796875,
        0.23046875,
        0.765625,
        -0.6875,
        -0.382080078125,
        -0.1119384765625,
        0.0255279541015625,
        -0.53515625,
        0.0,
        -0.1275634765625,
        0.1275634765625,
        0.0,
        0.53515625,
        -0.025543212890625,
        0.1119384765625,
        0.3818359375,
        0.6875,
        -0.765625,
        -0.23046875,
        0.53466796875,
        -0.12359619140625,
        -0.383544921875,
        -0.1785888671875,
        0.223876953125,
        0.2288818359375,
        -0.33154296875,
        0.203369140625,
        0.2083740234375,
        0.381103515625,
        0.5185546875,
        -0.53662109375,
        0.22802734375,
        0.76318359375,
        -0.2919921875,
        -0.48486328125,
        0.05023193359375,
        0.3203125,
        -0.6376953125,
        -0.102783203125,
        0.43212890625,
        0.074951171875,
        0.6103515625,
        0.35009765625,
        -0.078125,
        0.45703125,
        0.9921875,
    ];
    let (actual, ids, cu) = interpolate_pos_embed_and_rot_ids(&weights, 3, 3, 2, &[[1, 4, 6]]);
    assert_eq!(cu, [0, 24]);
    let mut expected_ids = Vec::new();
    for bh in 0..2 {
        for bw in 0..3 {
            for mh in 0..2 {
                for mw in 0..2 {
                    expected_ids.extend([bh * 2 + mh, bw * 2 + mw]);
                }
            }
        }
    }
    assert_eq!(ids, expected_ids);
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "position element {i}: {a} != {e}");
    }
}

#[test]
fn native_positions_round_half_fma_once() {
    // Actual 27B PNG: row 235 (h=5,w=37), channel 25. The last native
    // fma.rn.f16 rounds down; FP32 FMA followed by FP16 rounds up instead.
    let hidden = 1152;
    let mut weights = vec![0.0; 48 * 48 * hidden];
    for (h, w, value) in [
        (8, 44, 0.04150390625),
        (8, 45, 0.041259765625),
        (9, 44, 0.029541015625),
        (9, 45, 0.01275634765625),
    ] {
        weights[(h * 48 + w) * hidden + 25] = value;
        weights[(h * 48 + w) * hidden + 26] = -value;
    }
    let (actual, ids, _) =
        interpolate_pos_embed_and_rot_ids(&weights, 48, hidden, 2, &[[1, 30, 40]]);
    assert_eq!(&ids[235 * 2..235 * 2 + 2], &[5, 37]);
    assert_eq!(actual[235 * hidden + 25], 0.039093017578125);
    assert_eq!(actual[235 * hidden + 26], -0.039093017578125);
}

// FNV-1a over all widened FP32 output bits (little endian), same GPU1 oracle.
#[test]
fn native_positions_full_grid_fingerprints() {
    let cases = [
        (3, 1, 4, 6, 2, 3, 17218227075470719985u64),
        // Installed native Triton 3.7.1, sm86: model-width compilation uses
        // three FP16 FMAs, unlike the one-element-per-thread small-width case.
        // Independent synthetic weights, not captured model activations.
        (48, 1, 30, 40, 2, 128, 4295628623726909022u64),
        (48, 1, 30, 40, 2, 129, 4868932220638141922u64),
        (48, 1, 30, 40, 2, 256, 2425186116355096283u64),
        (48, 1, 30, 40, 2, 768, 14882857233327210807u64),
        (48, 1, 30, 40, 2, 1024, 10239377902525264241u64),
        (48, 1, 30, 40, 2, 1152, 17443693302721934485u64),
        (48, 1, 10, 14, 2, 5, 1642739686885855319u64),
        (48, 2, 18, 22, 2, 8, 11403488438285464701u64),
        (48, 1, 1, 1, 1, 5, 4284868331168657079u64),
        (48, 1, 48, 48, 2, 8, 4782556115949589068u64),
    ];
    for (n, t, h, w, m, d, expected) in cases {
        let weights: Vec<f32> = (0..n * n * d)
            .map(|i| ((i * 137 % 509) as f32 - 254.0) / 256.0)
            .collect();
        let (actual, ids, cu) = interpolate_pos_embed_and_rot_ids(&weights, n, d, m, &[[t, h, w]]);
        assert_eq!(actual.len(), t * h * w * d);
        assert_eq!(ids.len(), t * h * w * 2);
        assert_eq!(cu, [0, (t * h * w) as i32]);
        let mut hash = 14695981039346656037u64;
        for x in actual {
            for b in x.to_le_bytes() {
                hash = (hash ^ u64::from(b)).wrapping_mul(1099511628211);
            }
        }
        assert_eq!(hash, expected, "grid {n}: {t}x{h}x{w}, hidden {d}");
    }
}

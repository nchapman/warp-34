//! Real safetensors boundary tests for loading and TP slicing, without a GPU.
use super::*;
use crate::cuda::f32_to_f16;
use serde_json::json;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)>) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "w34-quant-{}-{}-{}",
            std::process::id(),
            nanos,
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new().create(&dir).unwrap();
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend(bytes);
            header.insert(
                name.into(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        let file_path = dir.join("model.safetensors");
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_path)
            .unwrap();
        out.write_all(&file).unwrap();
        Self(dir)
    }
    fn archive(&self) -> SafetensorsArchive {
        SafetensorsArchive::open(&self.0).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn halves(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn ct(bits: usize, group_size: usize, symmetric: bool) -> QuantScheme {
    QuantScheme::CompressedTensors {
        num_bits: bits,
        group_size,
        symmetric,
        ignore: vec![],
    }
}

#[test]
fn compressed_tensors_loading_and_tp_slicing_preserve_weights() {
    let (n, k) = (16usize, 256usize);
    for bits in [4usize, 8] {
        for group in [32usize, 64, 128, 256] {
            for asym in [false, true] {
                if (bits == 8 && asym) || (bits == 4 && group == 256) {
                    continue;
                }
                let ng = k / group;
                let pack = 32 / bits;
                let mask = (1usize << bits) - 1;
                let mut packed = vec![0u32; n * k / pack];
                let mut zeros = vec![0u32; n / 8 * ng];
                let mut scales = vec![0u16; n * ng];
                let mut expected = vec![0u16; n * k];
                for row in 0..n {
                    for g in 0..ng {
                        let z = if asym {
                            (row + g * 3) % 16
                        } else {
                            1 << (bits - 1)
                        };
                        let scale = (row + g + 1) as f32 / 256.0;
                        scales[row * ng + g] = f32_to_f16(scale);
                        if asym {
                            zeros[(row / 8) * ng + g] |= (z as u32) << ((row % 8) * 4);
                        }
                        for col in g * group..(g + 1) * group {
                            let q = (row * 7 + col * 13) & mask;
                            packed[row * (k / pack) + col / pack] |=
                                (q as u32) << ((col % pack) * bits);
                            expected[row * k + col] =
                                f32_to_f16((q as i32 - z as i32) as f32 * scale);
                        }
                    }
                }
                let mut tensors = vec![
                    (
                        "proj.weight_packed",
                        "I32",
                        vec![n, k / pack],
                        words(&packed),
                    ),
                    ("proj.weight_scale", "F16", vec![n, ng], halves(&scales)),
                ];
                if asym {
                    tensors.push((
                        "proj.weight_zero_point",
                        "I32",
                        vec![n / 8, ng],
                        words(&zeros),
                    ));
                }
                let fixture = Fixture::new(tensors);
                let loaded =
                    RawLinearHost::load(&fixture.archive(), "proj", &ct(bits, group, !asym))
                        .unwrap();
                assert_eq!(loaded.dims(), (n, k));
                assert_eq!(loaded.to_f16_vec(), expected);
                for rank in 0..2 {
                    assert_eq!(
                        loaded
                            .slice_rows(rank * n / 2, (rank + 1) * n / 2)
                            .to_f16_vec(),
                        expected[rank * n / 2 * k..(rank + 1) * n / 2 * k]
                    );
                    let column_reference: Vec<_> = expected
                        .chunks_exact(k)
                        .flat_map(|row| row[rank * k / 2..(rank + 1) * k / 2].iter().copied())
                        .collect();
                    assert_eq!(
                        loaded
                            .slice_cols(rank * k / 2, (rank + 1) * k / 2)
                            .to_f16_vec(),
                        column_reference
                    );
                }
            }
        }
    }
}

#[test]
fn raw_loading_reports_missing_scales_zero_points_and_bad_g_idx() {
    let packed = (
        "proj.weight_packed",
        "I32",
        vec![16, 32],
        words(&vec![0x99999999; 16 * 32]),
    );
    let fixture = Fixture::new(vec![packed.clone()]);
    let error = RawLinearHost::load(&fixture.archive(), "proj", &ct(4, 32, false))
        .err()
        .unwrap();
    assert!(error.to_string().contains("proj.weight_scale"), "{error}");
    let scales = (
        "proj.weight_scale",
        "F16",
        vec![16, 8],
        halves(&vec![f32_to_f16(0.125); 16 * 8]),
    );
    let fixture = Fixture::new(vec![packed.clone(), scales.clone()]);
    let error = RawLinearHost::load(&fixture.archive(), "proj", &ct(4, 32, false))
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("proj.weight_zero_point"),
        "{error}"
    );
    let mut g_idx: Vec<u32> = (0..256).map(|i| i / 32).collect();
    g_idx.swap(0, 32);
    let fixture = Fixture::new(vec![
        packed,
        scales,
        ("proj.weight_g_idx", "I32", vec![256], words(&g_idx)),
    ]);
    let error = RawLinearHost::load(&fixture.archive(), "proj", &ct(4, 32, true))
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("non-sequential g_idx"),
        "{error}"
    );
}

#[test]
fn unquantized_f16_bf16_f32_load_to_the_same_fp16_spine() {
    let values = [-2.0f32, -0.5, 0.0, 0.25, 1.0, 4.0];
    let expected: Vec<_> = values.iter().copied().map(f32_to_f16).collect();
    for (dtype, bytes) in [
        ("F16", halves(&expected)),
        (
            "BF16",
            halves(
                &values
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
            ),
        ),
        ("F32", values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    ] {
        let fixture = Fixture::new(vec![("proj.weight", dtype, vec![2, 3], bytes)]);
        let loaded =
            RawLinearHost::load(&fixture.archive(), "proj", &QuantScheme::Unquantized).unwrap();
        assert_eq!(loaded.dims(), (2, 3));
        assert_eq!(loaded.to_f16_vec(), expected, "{dtype}");
    }
}

#[test]
fn compressed_tensors_weight_shape_validates_and_resolves_embed_tokens_bits() {
    let (n, k) = (8usize, 128usize);
    let packed_u8 = vec![0x82828282u32; n * (k / 4)];
    let scales = vec![f32_to_f16(0.25); n];
    let shape_i64: Vec<u8> = [n as i64, k as i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // 8-bit embed_tokens in a model whose linear scheme is 4-bit asymmetric
    let fixture = Fixture::new(vec![
        (
            "model.embed_tokens.weight_packed",
            "I32",
            vec![n, k / 4],
            words(&packed_u8),
        ),
        (
            "model.embed_tokens.weight_scale",
            "F16",
            vec![n, 1],
            halves(&scales),
        ),
        (
            "model.embed_tokens.weight_shape",
            "I64",
            vec![2],
            shape_i64.clone(),
        ),
    ]);
    let loaded =
        RawLinearHost::load(&fixture.archive(), "model.embed_tokens", &ct(4, 128, false)).unwrap();
    assert_eq!(loaded.dims(), (n, k));
    assert_eq!(loaded.to_f16_vec(), vec![f32_to_f16(0.5); n * k]);

    // Non-embedding layer with conflicting weight_shape (8-bit tensor vs 4-bit config) fails fast
    let bad_fixture = Fixture::new(vec![
        (
            "proj.weight_packed",
            "I32",
            vec![n, k / 4],
            words(&packed_u8),
        ),
        ("proj.weight_scale", "F16", vec![n, 1], halves(&scales)),
        ("proj.weight_shape", "I64", vec![2], shape_i64),
    ]);
    let err = RawLinearHost::load(&bad_fixture.archive(), "proj", &ct(4, 128, true))
        .err()
        .unwrap();
    assert!(err.to_string().contains("weight_shape"), "{err}");

    // Truncated weight_shape (e.g. 8 bytes declared for I64 [2]) returns an error instead of panicking
    let short_fixture = Fixture::new(vec![
        (
            "proj.weight_packed",
            "I32",
            vec![n, k / 4],
            words(&packed_u8),
        ),
        ("proj.weight_scale", "F16", vec![n, 1], halves(&scales)),
        ("proj.weight_shape", "I64", vec![2], vec![0u8; 8]),
    ]);
    let err = RawLinearHost::load(&short_fixture.archive(), "proj", &ct(8, 128, true))
        .err()
        .unwrap();
    assert!(err.to_string().contains("expected 16 bytes"), "{err}");

    // Ignored / unquantized 2D BF16 .weight takes precedence over an experimental 1D U8 .weight_packed sidecar
    let bf16_val = (1.5f32.to_bits() >> 16) as u16;
    let sidecar_fixture = Fixture::new(vec![
        (
            "mtp.layers.0.mlp.gate_proj.weight",
            "BF16",
            vec![n, k],
            halves(&vec![bf16_val; n * k]),
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight_packed",
            "U8",
            vec![n * k / 2],
            vec![0u8; n * k / 2],
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight_scale",
            "U8",
            vec![n, 2],
            vec![0u8; n * 2],
        ),
    ]);
    let loaded = RawLinearHost::load(
        &sidecar_fixture.archive(),
        "mtp.layers.0.mlp.gate_proj",
        &ct(4, 128, true),
    )
    .unwrap();
    assert_eq!(loaded.dims(), (n, k));
    assert_eq!(loaded.to_f16_vec(), vec![f32_to_f16(1.5); n * k]);
}

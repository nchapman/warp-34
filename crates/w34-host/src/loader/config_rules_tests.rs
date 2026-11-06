use super::*;

// Quantization metadata copied verbatim from revision
// 0fbb9a38f631ea63678f4b51a491e7ecb26c681c (Qwen3.5-2B-W4A16-HGPTQ-g64).
fn fixture() -> Json {
    serde_json::from_str(include_str!("config_rules_2b.json")).unwrap()
}
fn parse(q: &Json) -> io::Result<QuantScheme> {
    ModelConfig::parse_quant_compressed_tensors(q)
}

#[test]
fn pinned_2b_actual_names_and_exclusions() {
    let scheme = parse(&fixture()).unwrap();
    for name in [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.3.self_attn.q_proj",
        "mtp.layers.0.self_attn.q_proj",
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.mlp.linear_fc2",
        "model.visual.merger.linear_fc1",
    ] {
        assert_eq!(
            scheme.for_linear(name).unwrap().raw_group_size(),
            64,
            "{name}"
        );
        assert_eq!(scheme.input_hadamard(name).unwrap(), Some(64), "{name}");
    }
    for name in [
        "lm_head",
        "model.lm_head",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.mlp.gate",
        "mtp.layers.0.linear_attn.in_proj_qkv",
        "model.visual.patch_embed.proj",
    ] {
        assert_eq!(
            scheme.for_linear(name).unwrap(),
            QuantScheme::Unquantized,
            "{name}"
        );
        assert_eq!(scheme.input_hadamard(name).unwrap(), None, "{name}");
    }
}

#[test]
fn synthetic_9b_channel_and_h16() {
    let mut q = fixture();
    let target = q["config_groups"]["group_0"]["targets"]
        .as_array_mut()
        .unwrap()
        .pop()
        .unwrap();
    let mut channel = q["config_groups"]["group_0"].clone();
    channel["targets"] = serde_json::json!([target]);
    channel["weights"]["strategy"] = "channel".into();
    channel["weights"]["group_size"] = Json::Null;
    q["config_groups"]["vision_fc2"] = channel;
    let mut h16 = q["transform_config"]["config_groups"]["v"].clone();
    h16["head_dim"] = 16.into();
    for i in 0..2 {
        q["transform_config"]["config_groups"]["v"]["apply"][i]["targets"]
            .as_array_mut()
            .unwrap()
            .pop();
        h16["apply"][i]["targets"] = serde_json::json!([target]);
    }
    q["transform_config"]["config_groups"]["fc2"] = h16;
    let scheme = parse(&q).unwrap();
    let name = "model.visual.blocks.12.mlp.linear_fc2";
    assert_eq!(scheme.for_linear(name).unwrap().raw_group_size(), 0);
    assert_eq!(scheme.for_linear(name).unwrap().bits(), 4);
    assert_eq!(scheme.input_hadamard(name).unwrap(), Some(16));
    assert_eq!(
        scheme
            .input_hadamard("model.language_model.layers.0.mlp.down_proj")
            .unwrap(),
        Some(64)
    );
}

#[test]
fn malformed_and_unsupported_transforms_rejected() {
    for (key, value) in [
        ("type", serde_json::json!("random")),
        ("head_dim", serde_json::json!(32)),
        ("randomize", serde_json::json!(true)),
        ("requires_grad", serde_json::json!(true)),
    ] {
        let mut q = fixture();
        q["transform_config"]["config_groups"]["v"][key] = value;
        assert!(parse(&q).is_err(), "{key}");
    }
    for (key, value) in [
        ("location", serde_json::json!("output")),
        ("inverse", serde_json::json!(true)),
        ("targets", serde_json::json!(["re:("])),
        ("ignore", serde_json::json!([42])),
    ] {
        let mut q = fixture();
        q["transform_config"]["config_groups"]["v"]["apply"][1][key] = value;
        assert!(parse(&q).is_err(), "{key}");
    }
}

#[test]
fn matching_conflicts_fail_closed_and_ignore_wins() {
    let name = "model.language_model.layers.0.mlp.down_proj";
    let mut q = fixture();
    q["transform_config"]["config_groups"]["other"] =
        q["transform_config"]["config_groups"]["v"].clone();
    let scheme = parse(&q).unwrap();
    assert!(scheme.for_linear(name).is_err());
    assert!(scheme.input_hadamard(name).is_err());
    q["ignore"] = serde_json::json!([name]);
    assert_eq!(parse(&q).unwrap().input_hadamard(name).unwrap(), None);
    let mut q = fixture();
    let mut group = q["config_groups"]["group_0"].clone();
    group["weights"]["strategy"] = "channel".into();
    group["weights"]["group_size"] = Json::Null;
    q["config_groups"]["conflict"] = group;
    assert!(parse(&q).unwrap().for_linear(name).is_err());
    let mut q = fixture();
    q["transform_config"]["config_groups"]["v"]["apply"][0]["targets"] =
        serde_json::json!(["different"]);
    assert!(parse(&q).unwrap().input_hadamard(name).is_err());
    let mut q = fixture();
    q["config_groups"]["group_0"]["targets"] = serde_json::json!(["different"]);
    assert!(parse(&q).unwrap().input_hadamard(name).is_err());
}

#[test]
fn legacy_unchanged_and_regex_is_bounded() {
    let mut q = fixture();
    q.as_object_mut().unwrap().remove("transform_config");
    let scheme = parse(&q).unwrap();
    assert!(matches!(
        scheme,
        QuantScheme::CompressedTensors { rules: None, .. }
    ));
    assert_eq!(scheme.for_linear("anything").unwrap(), scheme);
    assert_eq!(scheme.input_hadamard("anything").unwrap(), None);
    assert!(
        !matches(
            &[compile_pattern("re:layers\\.\\d+").unwrap()],
            "model.layers.1"
        )
        .unwrap()
    );
    assert!(
        matches(
            &[compile_pattern("re:^(a+)+$(?!x)").unwrap()],
            &format!("{}!", "a".repeat(1000))
        )
        .is_err()
    );
}

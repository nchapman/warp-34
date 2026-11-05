use serde_json::json;
use w34_host::server::completion_sampling_params;

#[test]
fn legacy_integer_logprobs_regression() {
    let params = completion_sampling_params(&json!({"logprobs": 0}), 8).unwrap();
    assert!(params.logprobs);
    assert_eq!(params.top_logprobs, 0);
}

#[test]
fn legacy_bounds_and_absence() {
    for n in [0, 1, 5, 20] {
        let p = completion_sampling_params(&json!({"logprobs":n}), 8).unwrap();
        assert!(p.logprobs);
        assert_eq!(p.top_logprobs, n as usize);
    }
    for value in [
        json!(-1),
        json!(21),
        json!(1.5),
        json!(true),
        json!(false),
        json!("2"),
    ] {
        assert!(completion_sampling_params(&json!({"logprobs":value}), 8).is_err());
    }
    for request in [json!({}), json!({"logprobs":null})] {
        assert!(!completion_sampling_params(&request, 8).unwrap().logprobs);
    }
    assert!(completion_sampling_params(&json!({"logprobs":1,"top_logprobs":2}), 8).is_err());
    assert!(completion_sampling_params(&json!({"logprobs":1,"min_tokens":9}), 8).is_err());
}

#[test]
fn chat_contract_unchanged() {
    use w34_host::server::native_sampling_params;
    let p = native_sampling_params(&json!({"logprobs":true,"top_logprobs":20}), 8).unwrap();
    assert!(p.logprobs);
    assert_eq!(p.top_logprobs, 20);
    assert!(native_sampling_params(&json!({"logprobs":1}), 8).is_err());
    assert!(native_sampling_params(&json!({"logprobs":false,"top_logprobs":1}), 8).is_err());
}

#[test]
fn legacy_full_and_stream_shapes_and_unicode_offsets() {
    use w34_host::server::completion_logprobs;
    let logs = vec![
        json!({"token":"é","logprob":-0.2,"top_logprobs":[{"token":"é","logprob":-0.2},{"token":"a","logprob":-2.0}]}),
        json!({"token":" hello","logprob":-0.3,"top_logprobs":[]}),
    ];
    let expected = json!({"tokens":["é"," hello"],"token_logprobs":[-0.2,-0.3],"top_logprobs":[{"é":-0.2,"a":-2.0},{}],"text_offset":[0,1]});
    assert_eq!(completion_logprobs(&logs, &mut 0), expected);
    let mut offset = 0;
    for (i, log) in logs.iter().enumerate() {
        let chunk = completion_logprobs(std::slice::from_ref(log), &mut offset);
        for key in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
            assert_eq!(chunk[key], json!([expected[key][i]]));
        }
        assert!(chunk.get("content").is_none());
    }
    assert_eq!(offset, 7);
    assert_eq!(
        completion_logprobs(&[], &mut 0),
        json!({"tokens":[],"token_logprobs":[],"top_logprobs":[],"text_offset":[]})
    );
}

#[test]
fn legacy_request_object_boundary_and_nullable_chat_option() {
    for request in [
        json!(null),
        json!(false),
        json!(7),
        json!("request"),
        json!([]),
    ] {
        assert!(completion_sampling_params(&request, 8).is_err());
    }
    let p = completion_sampling_params(&json!({"logprobs":2,"top_logprobs":null}), 8).unwrap();
    assert!(p.logprobs);
    assert_eq!(p.top_logprobs, 2);
}

#[test]
fn byte_level_split_utf8_offsets_follow_stream_decoder() {
    use tokenizers::{Tokenizer, decoders::byte_level::ByteLevel, models::wordlevel::WordLevel};
    use w34_host::server::completion_logprobs;
    // ByteLevel maps these vocabulary characters to C3, A9, 78 respectively.
    let model = WordLevel::builder()
        .vocab(
            [("Ã".into(), 0), ("©".into(), 1), ("x".into(), 2)]
                .into_iter()
                .collect(),
        )
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    assert_eq!(tokenizer.decode(&[0], false).unwrap(), "�");
    assert_eq!(tokenizer.decode(&[1], false).unwrap(), "�");
    let mut decoder = tokenizer.decode_stream(false);
    let mut text = String::new();
    let mut logs = Vec::new();
    for (id, expected) in [(0, None), (1, Some("é")), (2, Some("x"))] {
        let decoded = decoder.step(id).unwrap();
        assert_eq!(decoded.as_deref(), expected);
        logs.push(json!({"token":tokenizer.decode(&[id], false).unwrap(),
            "logprob":-0.5,"top_logprobs":[],"_text_offset":text.chars().count()}));
        text.push_str(decoded.as_deref().unwrap_or_default());
    }
    assert_eq!(text, "éx");
    let full = completion_logprobs(&logs, &mut 0);
    assert_eq!(full["text_offset"], json!([0, 0, 1]));
    assert_eq!(full["tokens"].as_array().unwrap().len(), 3);
    let mut offset = 0;
    for (i, log) in logs.iter().enumerate() {
        let chunk = completion_logprobs(std::slice::from_ref(log), &mut offset);
        for key in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
            assert_eq!(chunk[key], json!([full[key][i]]));
        }
        assert!(chunk.get("_text_offset").is_none());
    }
}

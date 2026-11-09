//! Exact next-token masks from the official Python XGrammar used by vLLM.
//! Regenerate with tools/oracle/xgrammar_masks.py, never with the Rust backend.
use serde_json::Value;
use w34_host::constraints::ConstraintFactory;

#[test]
fn no_tools_keeps_call_markup_as_content() {
    let text = "<tool_call>\n<function=ping>\n\n</function>\n</tool_call>";
    let mut parser = w34_host::parser::Qwen3Parser::with_tools(false, &[]).unwrap();
    for ch in text.chars() {
        parser.push_str(&ch.to_string());
    }
    let turn = parser.finish().1;
    assert!(turn.tool_calls.is_empty());
    assert_eq!(turn.content.as_deref(), Some(text));
}

#[test]
fn parallel_tool_calls_false_stops_after_one_call() {
    use w34_host::tool_constraints::ToolChoice;
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/xgrammar_masks.json")).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&fixture["tokenizer"]).unwrap())
            .unwrap();
    let factory = ConstraintFactory::new(tokenizer.clone(), 260, &[256]).unwrap();
    let tools = serde_json::json!([{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{},"additionalProperties":false}}}]);
    for choice in [ToolChoice::Auto, ToolChoice::Required] {
        let mut c = factory
            .tools_with_parallel(&tools, &choice, false, false)
            .unwrap();
        let mut whitespace_loop = c.clone();
        let mut blocked = false;
        for id in tokenizer
            .encode("<tool_call>\n<function=ping>\n        ", false)
            .unwrap()
            .get_ids()
        {
            if whitespace_loop.advance(*id as i32).is_err() {
                blocked = true;
                break;
            }
        }
        assert!(
            blocked,
            "empty XML calls must not admit unbounded whitespace"
        );
        let text = "<tool_call>\n<function=ping>\n\n</function>\n</tool_call>";
        for id in tokenizer.encode(text, false).unwrap().get_ids() {
            c.advance(*id as i32).unwrap();
        }
        assert!(c.is_token_allowed(256));
        let first = tokenizer.encode("<", false).unwrap().get_ids()[0];
        assert!(!c.is_token_allowed(first as i32));
    }
}

#[test]
fn masks_and_rollback_match_official_xgrammar() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/xgrammar_masks.json")).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&fixture["tokenizer"]).unwrap())
            .unwrap();
    let vocab = fixture["vocab_size"].as_u64().unwrap() as usize;
    let factory =
        ConstraintFactory::new(tokenizer, vocab, &[fixture["eos"].as_u64().unwrap() as u32])
            .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut constraint = match case["kind"].as_str().unwrap() {
            "regex" => factory.regex(case["source"].as_str().unwrap(), false),
            "schema" => factory.json_schema(case["source"].clone(), false),
            "grammar" => factory.lark(case["source"].as_str().unwrap(), false),
            "qwen_tools" => factory.tools(
                &case["source"],
                &w34_host::tool_constraints::ToolChoice::Required,
                false,
            ),
            kind => panic!("unknown fixture kind: {kind}"),
        }
        .unwrap();
        let tokens = case["tokens"].as_array().unwrap();
        let masks = case["masks"].as_array().unwrap();
        for (step, (token, expected)) in tokens.iter().zip(masks).enumerate() {
            let expected: Vec<u32> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|word| word.as_u64().unwrap() as u32)
                .collect();
            // Unused bits beyond model vocabulary are not token decisions.
            for id in 0..vocab {
                assert_eq!(
                    constraint.is_token_allowed(id as i32),
                    expected[id / 32] & (1 << (id % 32)) != 0,
                    "{} step={step} token={id}",
                    case["kind"]
                );
            }
            let before = constraint.current_bitmask();
            let token = token.as_i64().unwrap() as i32;
            constraint.advance(token).unwrap();
            constraint.rollback(1).unwrap();
            assert_eq!(constraint.current_bitmask(), before);
            constraint.advance(token).unwrap();
        }
        assert!(constraint.is_terminal_eos());
        constraint.rollback(tokens.len()).unwrap();
        assert_eq!(constraint.committed_len(), 0);
        for token in tokens {
            constraint.advance(token.as_i64().unwrap() as i32).unwrap();
        }
        assert!(constraint.is_terminal_eos());
    }
}

use serde_json::json;
use tokenizers::Tokenizer;
use w34_host::{
    constraints::{ConstraintFactory, StructuralTag, TokenBitmaskConstraint},
    parser::{SamplerV1, SamplingParams},
};

const EOS: i32 = 256;

fn tokenizer() -> Tokenizer {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort();
    let mut vocab = serde_json::Map::new();
    for (id, ch) in alphabet.into_iter().enumerate() {
        vocab.insert(ch.to_string(), json!(id));
    }
    vocab.insert("<|im_end|>".into(), json!(EOS));
    // A token crossing the reasoning/JSON boundary exercises byte-level state.
    vocab.insert("</think>{\"ok\":".into(), json!(257));
    vocab.insert("<think>".into(), json!(258));
    vocab.insert("</think>".into(), json!(259));
    vocab.insert("<tool_call>".into(), json!(260));
    vocab.insert("</tool_call>".into(), json!(261));
    vocab.insert("<|endoftext|>".into(), json!(262));
    Tokenizer::from_bytes(serde_json::to_vec(&json!({
        "version":"1.0", "truncation":null,"padding":null,
        "added_tokens":[
            {"id":256,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":257,"content":"</think>{\"ok\":","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
            {"id":258,"content":"<think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
            {"id":259,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
            {"id":260,"content":"<tool_call>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
            {"id":261,"content":"</tool_call>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
            {"id":262,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ],
        "normalizer":null,
        "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "post_processor":null,
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"BPE","vocab":vocab,"merges":[]}
    })).unwrap()).unwrap()
}

fn setup() -> (Tokenizer, ConstraintFactory) {
    let tok = tokenizer();
    let factory = ConstraintFactory::new(tok.clone(), 266, &[EOS as u32]).unwrap();
    (tok, factory)
}

fn feed(c: &mut TokenBitmaskConstraint, tok: &Tokenizer, text: &str) -> Result<(), String> {
    for id in tok.encode(text, false).unwrap().get_ids() {
        c.advance(*id as i32)?;
    }
    Ok(())
}

#[test]
fn grammar_completion_eos_commits_and_rolls_back_losslessly() {
    let (tok, factory) = setup();
    for mut constraint in [
        factory.regex("ok", false).unwrap(),
        factory.json_object(false).unwrap(),
    ] {
        let text =
            if constraint.is_token_allowed(tok.encode("o", false).unwrap().get_ids()[0] as i32) {
                "ok"
            } else {
                "{}"
            };
        feed(&mut constraint, &tok, text).unwrap();
        let before = constraint.current_bitmask();
        let count = constraint.committed_len();
        constraint.advance(EOS).unwrap();
        assert_eq!(constraint.committed_len(), count + 1);
        assert!(constraint.advance(EOS).is_err());
        constraint.rollback(1).unwrap();
        assert_eq!(constraint.current_bitmask(), before);
        constraint.advance(EOS).unwrap();
        constraint.rollback(count + 1).unwrap();
        feed(&mut constraint, &tok, text).unwrap();
        assert_eq!(constraint.current_bitmask(), before);
    }
}

#[test]
fn json_object_enforces_every_byte_not_just_first_brace() {
    let (tok, factory) = setup();
    for text in [
        "{}",
        r#"{"a":[true,false,null,1.25,-3e2,{"escaped":"quote\"slash\\"}]}"#,
        "{\"unicode\":\"é中😀\"}",
    ] {
        let mut c = factory.json_object(false).unwrap();
        assert!(!c.is_token_allowed(EOS));
        feed(&mut c, &tok, text).unwrap();
        assert!(c.is_token_allowed(EOS), "{text}");
        assert!(!c.is_token_allowed(262)); // non-EOS special token
        assert!(!c.is_token_allowed(265)); // padded model-vocab entry
    }
    for text in ["[]", "{x", "{\"x\":truX", "{\"x\":1,}", "{}x"] {
        let mut c = factory.json_object(false).unwrap();
        assert!(feed(&mut c, &tok, text).is_err(), "accepted {text}");
    }
}

#[test]
fn strict_schema_required_enum_nested_array_and_extra_keys() {
    let (tok, factory) = setup();
    let schema = json!({"type":"object","properties":{
        "city":{"enum":["Paris","Rome"]},
        "days":{"type":"array","items":{"type":"integer","minimum":1,"maximum":3},"minItems":1,"maxItems":2}
    },"required":["city","days"],"additionalProperties":false});
    for (text, accepted) in [
        (r#"{"city":"Paris","days":[1,3]}"#, true),
        (r#"{"city":"Oslo","days":[1]}"#, false),
        (r#"{"city":"Rome"}"#, false),
        (r#"{"city":"Rome","days":[]}"#, false),
        (r#"{"city":"Rome","days":[4]}"#, false),
        (r#"{"city":"Rome","days":[1],"extra":0}"#, false),
    ] {
        let mut c = factory.json_schema(schema.clone(), false).unwrap();
        let result = feed(&mut c, &tok, text);
        assert_eq!(
            result.is_ok() && c.is_token_allowed(EOS),
            accepted,
            "{text}: {result:?}"
        );
    }
}

#[test]
fn thinking_transition_clone_and_rollback_are_lossless() {
    let (tok, factory) = setup();
    let mut c = factory.json_schema(json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}), true).unwrap();
    feed(&mut c, &tok, "Reason freely: not JSON!\n").unwrap();
    let before = c.current_bitmask();
    let count = c.committed_len();
    let mut independent = c.clone();
    assert!(c.is_token_allowed(257));
    assert!(c.is_token_allowed(259)); // standalone non-special </think> added token
    assert!(!c.is_token_allowed(260)); // <tool_call> banned inside reasoning
    assert!(!c.is_token_allowed(262)); // true special token remains disallowed
    c.advance(257).unwrap();
    assert!(!c.is_token_allowed(EOS));
    assert!(feed(&mut c, &tok, "invalid").is_err());
    feed(&mut c, &tok, "true}").unwrap();
    assert!(c.is_token_allowed(EOS));
    c.rollback(c.committed_len() - count).unwrap();
    assert_eq!(c.current_bitmask(), before);
    feed(&mut c, &tok, "</think>{\"ok\":false}").unwrap();
    feed(&mut independent, &tok, "</think>{\"ok\":false}").unwrap();
    assert_eq!(c.current_bitmask(), independent.current_bitmask());
    assert_eq!(c.committed_len(), independent.committed_len());
}

#[test]
fn regex_and_recursive_cfg_have_transactional_masks() {
    let (tok, factory) = setup();
    let mut regex = factory.regex("[A-Z]{2}[0-9]{2}", false).unwrap();
    feed(&mut regex, &tok, "AB12").unwrap();
    assert!(regex.is_token_allowed(EOS));
    regex.rollback(2).unwrap();
    assert!(feed(&mut regex, &tok, "X").is_err());
    feed(&mut regex, &tok, "34").unwrap();
    let mut cfg = factory
        .lark("start: \"(\" start \")\" | \"x\"", false)
        .unwrap();
    feed(&mut cfg, &tok, "((x))").unwrap();
    assert!(cfg.is_token_allowed(EOS));
}

#[test]
fn invalid_commit_and_rollback_do_not_mutate_sampler() {
    let (tok, factory) = setup();
    let c = factory.regex("[0-9]+", false).unwrap();
    let mut sampler = SamplerV1::new(SamplingParams::default(), &[], Some(c));
    let before = format!("{sampler:?}");
    let invalid = tok.encode("x", false).unwrap().get_ids()[0];
    assert!(sampler.try_commit_token(invalid as i32).is_err());
    assert!(sampler.try_rollback_tokens(1).is_err());
    assert_eq!(format!("{sampler:?}"), before);
    let digit = tok.encode("1", false).unwrap().get_ids()[0];
    sampler.try_commit_token(digit as i32).unwrap();
    sampler.try_rollback_tokens(1).unwrap();
    assert_eq!(format!("{sampler:?}"), before);
}

#[test]
fn structural_tags_enforce_triggered_payloads_and_rollback() {
    let (tok, factory) = setup();
    let tags = [StructuralTag {
        begin: "<answer>".into(),
        end: "</answer>".into(),
        schema: json!({"type":"integer","minimum":1,"maximum":9}),
    }];
    let initial = factory
        .structural_tags(&tags, &["<answer".into()], false)
        .unwrap();
    for text in [
        "free text",
        "before <answer>3</answer> after",
        "<answer>1</answer><answer>9</answer>",
    ] {
        let mut c = initial.clone();
        feed(&mut c, &tok, text).unwrap();
        assert!(c.is_token_allowed(EOS), "{text}");
    }
    for text in [
        "before <answer>bad",
        "<answer>0",
        "<answer>2</wrong>",
        "<answer_bad>",
    ] {
        let mut c = initial.clone();
        assert!(
            feed(&mut c, &tok, text).is_err(),
            "unconstrained bypass: {text}"
        );
    }
    let mut c = initial.clone();
    feed(&mut c, &tok, "<answer>5</answer>").unwrap();
    c.rollback(c.committed_len()).unwrap();
    assert_eq!(c.current_bitmask(), initial.current_bitmask());
}

#[test]
fn tool_choice_xml_and_hermes_schema_adherence() {
    use w34_host::{
        parser::Qwen3Parser,
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();
    let tools = json!([
        {"type":"function","function":{"name":"weather","parameters":{
            "type":"object","properties":{"city":{"enum":["Paris","Rome"]},"label":{"type":"string"}},
            "required":["city","label"],"additionalProperties":false
        }}},
        {"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{},"additionalProperties":false}}}
    ]);
    for format in [ToolFormat::QwenXml, ToolFormat::HermesJson] {
        let valid = match format {
            ToolFormat::QwenXml => {
                "<tool_call>\n<function=weather>\n<parameter=city>\n\"Paris\"\n</parameter>\n<parameter=label>\n\"true\"\n</parameter>\n</function>\n</tool_call>"
            }
            ToolFormat::HermesJson => {
                r#"<tool_call>{"name":"weather","arguments":{"city":"Paris","label":"true"}}</tool_call>"#
            }
        };
        for choice in [
            ToolChoice::Auto,
            ToolChoice::Required,
            ToolChoice::Named("weather".into()),
        ] {
            let mut c = factory.tools(&tools, &choice, format, true).unwrap();
            assert!(
                feed(&mut c.clone(), &tok, "thinking <tool_call>").is_err(),
                "tool call bypassed reasoning constraint"
            );
            feed(&mut c, &tok, "reasoning\n</think>").unwrap();
            let before = c.current_bitmask();
            let count = c.committed_len();
            feed(&mut c, &tok, valid).unwrap();
            assert!(c.is_token_allowed(EOS));
            let parsed = Qwen3Parser::parse_complete(valid, false);
            assert_eq!(parsed.tool_calls.len(), 1);
            assert_eq!(parsed.tool_calls[0].function.name, "weather");
            let args: serde_json::Value =
                serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
            assert_eq!(args, json!({"city":"Paris","label":"true"}));
            c.rollback(c.committed_len() - count).unwrap();
            assert_eq!(c.current_bitmask(), before);
            for invalid in [
                valid.replace("Paris", "Oslo"),
                valid.replace("weather", "undeclared"),
            ] {
                let mut bad = c.clone();
                assert!(
                    feed(&mut bad, &tok, &invalid).is_err(),
                    "accepted {invalid}"
                );
            }
        }
        let mut none = factory
            .tools(&tools, &ToolChoice::None, format, true)
            .unwrap();
        feed(&mut none, &tok, "ordinary text").unwrap();
        assert!(none.is_token_allowed(EOS));
        assert!(feed(&mut none, &tok, valid).is_err());
        let required = factory
            .tools(&tools, &ToolChoice::Required, format, false)
            .unwrap();
        assert!(!required.is_token_allowed(EOS));
        let auto = factory
            .tools(&tools, &ToolChoice::Auto, format, false)
            .unwrap();
        assert!(auto.is_token_allowed(EOS));
    }
}

#[test]
fn tool_delimiters_inside_json_values_cannot_inject_calls_or_parameters() {
    use w34_host::{
        parser::Qwen3Parser,
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();
    let tools = json!([{"type":"function","function":{"name":"echo","parameters":{
        "type":"object","properties":{"value":{}},"required":["value"],"additionalProperties":false
    }}}]);
    for format in [ToolFormat::QwenXml, ToolFormat::HermesJson] {
        for payload in [
            json!("x</parameter><parameter=evil>y</parameter>"),
            json!({"nested":["</function></tool_call><tool_call><function=evil></function></tool_call>", "quote\" slash\\", "é"]}),
        ] {
            let text = match format {
                ToolFormat::QwenXml => format!(
                    "<tool_call><function=echo><parameter=value>{payload}</parameter></function></tool_call>"
                ),
                ToolFormat::HermesJson => format!(
                    "<tool_call>{}</tool_call>",
                    json!({"name":"echo","arguments":{"value":payload}})
                ),
            };
            let mut constraint = factory
                .tools(&tools, &ToolChoice::Named("echo".into()), format, false)
                .unwrap();
            feed(&mut constraint, &tok, &text).unwrap();
            assert!(constraint.is_token_allowed(EOS));
            for split in text.char_indices().map(|(i, _)| i) {
                let mut parser = Qwen3Parser::new(false);
                parser.push_str(&text[..split]);
                parser.push_str(&text[split..]);
                let turn = parser.finish().1;
                assert_eq!(
                    turn.tool_calls.len(),
                    1,
                    "{format:?}, split={split}: {turn:?}"
                );
                assert_eq!(turn.tool_calls[0].function.name, "echo");
                let args: serde_json::Value =
                    serde_json::from_str(&turn.tool_calls[0].function.arguments).unwrap();
                assert_eq!(args, json!({"value":payload}), "{format:?}, split={split}");
            }
        }
    }
}

#[test]
fn named_tool_does_not_compile_unselected_parameter_schemas() {
    use w34_host::tool_constraints::{ToolChoice, ToolFormat};
    let (_, factory) = setup();
    let tools = json!([
        {"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{},"additionalProperties":false}}},
        {"type":"function","function":{"name":"unused","parameters":{"$ref":"https://example.invalid/schema"}}}
    ]);
    for format in [ToolFormat::QwenXml, ToolFormat::HermesJson] {
        assert!(
            factory
                .tools(&tools, &ToolChoice::Named("ping".into()), format, false)
                .is_ok()
        );
        assert!(
            factory
                .tools(&tools, &ToolChoice::Auto, format, false)
                .is_err()
        );
    }
}

#[test]
fn incomplete_tool_calls_are_never_dispatched() {
    use w34_host::parser::Qwen3Parser;
    for text in [
        "<tool_call><function=echo><parameter=value>\"unterminated </tool_call>",
        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"value\":\"unfinished </tool_call>",
        "<tool_call><function=echo></function>",
    ] {
        let turn = Qwen3Parser::parse_complete(text, false);
        assert!(turn.tool_calls.is_empty(), "{turn:?}");
    }
}

#[test]
fn unsupported_grammars_fail_closed() {
    let (_, factory) = setup();
    assert!(
        factory
            .json_schema(json!({"$ref":"https://example.invalid/schema"}), false)
            .is_err()
    );
    assert!(factory.regex("(?=a)a", false).is_err());
    assert!(
        factory
            .json_schema(
                json!({"type":"object","x-guidance":{"lenient":true}}),
                false
            )
            .is_err()
    );
    assert!(factory.lark("start[stop   = \"x\"]: /a+/", false).is_err());
    assert!(factory.lark("start: \"max_tokens\"", false).is_ok());
    assert!(factory.lark("start[max_tokens=1]: /a+/", false).is_err());
    assert!(ConstraintFactory::new(tokenizer(), 260, &[]).is_err());
    assert!(ConstraintFactory::new(tokenizer(), 260, &[999]).is_err());
}

#[test]
fn deferred_advance_and_rollback_match_eager_bitmasks() {
    let (tok, factory) = setup();
    let schema = json!({
        "type": "object",
        "properties": { "ok": { "type": "boolean" } },
        "required": ["ok"],
        "additionalProperties": false
    });
    let mut eager = factory.json_schema(schema.clone(), false).unwrap();
    let mut deferred = factory.json_schema(schema, false).unwrap();
    let ids: Vec<i32> = tok
        .encode(r#"{"ok":true}"#, false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();

    for &id in &ids {
        let before = deferred.current_bitmask();
        deferred.advance_deferred(id).unwrap();
        assert!(deferred.has_pending());
        // Unmaterialized rollback restores exact pre-step state without touching matcher.
        deferred.rollback(1).unwrap();
        assert!(!deferred.has_pending());
        assert_eq!(deferred.current_bitmask(), before);

        eager.advance(id).unwrap();
        deferred.advance_deferred(id).unwrap();
        deferred.materialize_pending().unwrap();
        assert!(!deferred.has_pending());
        assert_eq!(deferred.current_bitmask(), eager.current_bitmask());
    }
    assert!(deferred.is_token_allowed(EOS));
}

#[test]
fn xml_unquoted_and_quoted_strings_and_code_snippets_roundtrip() {
    use serde_json::Value;
    use w34_host::{
        parser::Qwen3Parser,
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "exec_snippet",
            "parameters": {
                "type": "object",
                "properties": {
                    "lang": { "enum": ["rust", "python"] },
                    "code": { "type": "string" }
                },
                "required": ["lang", "code"],
                "additionalProperties": false
            }
        }
    }]);

    // 1. Native unquoted enum + unquoted multi-line code containing `<` and generics
    let mut c = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    let unquoted_call = "<tool_call>\n<function=exec_snippet>\n<parameter=lang>\nrust\n</parameter>\n<parameter=code>\nif x < 10 {\n    let v: Vec<i32> = Vec::new();\n}\n</parameter>\n</function>\n</tool_call>";
    feed(&mut c, &tok, unquoted_call).unwrap();
    assert!(c.is_token_allowed(EOS));
    let parsed_unquoted =
        Qwen3Parser::parse_complete_with_tools(unquoted_call, false, Some(&tools));
    assert_eq!(parsed_unquoted.tool_calls.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(&parsed_unquoted.tool_calls[0].function.arguments).unwrap(),
        json!({
            "lang": "rust",
            "code": "if x < 10 {\n    let v: Vec<i32> = Vec::new();\n}"
        })
    );

    // 2. JSON-quoted enum + JSON-quoted string produces the exact same parsed result
    let mut c_quoted = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    let quoted_call = "<tool_call>\n<function=exec_snippet>\n<parameter=lang>\n\"rust\"\n</parameter>\n<parameter=code>\n\"if x < 10 {\\n    let v: Vec<i32> = Vec::new();\\n}\"\n</parameter>\n</function>\n</tool_call>";
    feed(&mut c_quoted, &tok, quoted_call).unwrap();
    assert!(c_quoted.is_token_allowed(EOS));
    let parsed_quoted = Qwen3Parser::parse_complete_with_tools(quoted_call, false, Some(&tools));
    assert_eq!(parsed_quoted.tool_calls, parsed_unquoted.tool_calls);

    // 3. Invalid unquoted enum value is rejected
    let mut c_bad = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    assert!(
        feed(
            &mut c_bad,
            &tok,
            "<tool_call>\n<function=exec_snippet>\n<parameter=lang>\ngo\n</parameter>"
        )
        .is_err()
    );
}

#[test]
fn xml_unordered_parameters_and_no_duplicates() {
    use serde_json::Value;
    use w34_host::{
        parser::Qwen3Parser,
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": { "enum": ["Paris", "Rome"] },
                    "units": { "enum": ["c", "f"] },
                    "days": { "type": "integer", "minimum": 1, "maximum": 7 }
                },
                "required": ["city", "days"],
                "additionalProperties": false
            }
        }
    }]);

    // Reverse order (`days`, then `units`, then `city`) is accepted
    let mut c1 = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    let reverse_call = "<tool_call>\n<function=forecast>\n<parameter=days>\n3\n</parameter>\n<parameter=units>\nc\n</parameter>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>";
    feed(&mut c1, &tok, reverse_call).unwrap();
    assert!(c1.is_token_allowed(EOS));
    let parsed = Qwen3Parser::parse_complete_with_tools(reverse_call, false, Some(&tools));
    assert_eq!(
        serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).unwrap(),
        json!({"city": "Rome", "units": "c", "days": 3})
    );

    // Omitting optional `units` with `(days, city)` order is accepted
    let mut c2 = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    feed(
        &mut c2,
        &tok,
        "<tool_call>\n<function=forecast>\n<parameter=days>\n5\n</parameter>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    )
    .unwrap();
    assert!(c2.is_token_allowed(EOS));

    // Omitting required `days` is rejected at `</function>`
    let mut c_missing = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    assert!(
        feed(
            &mut c_missing,
            &tok,
            "<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>"
        )
        .is_err()
    );

    // Duplicate parameter `<parameter=city>` is rejected
    let mut c_dup = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    assert!(
        feed(
            &mut c_dup,
            &tok,
            "<tool_call>\n<function=forecast>\n<parameter=city>\nParis\n</parameter>\n<parameter=city>\nRome\n</parameter>"
        )
        .is_err()
    );
}

#[test]
fn xml_tools_propagate_defs_to_parameter_schemas() {
    use w34_host::tool_constraints::{ToolChoice, ToolFormat};
    let (tok, factory) = setup();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "navigate",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "object",
                        "properties": {
                            "point": { "$ref": "#/$defs/Point" },
                            "label": { "$ref": "#/$defs/Label" }
                        },
                        "required": ["point"],
                        "additionalProperties": false,
                        "$defs": {
                            "Label": { "type": "string" }
                        }
                    }
                },
                "required": ["destination"],
                "additionalProperties": false,
                "$defs": {
                    "Point": {
                        "type": "object",
                        "properties": {
                            "x": { "type": "integer" },
                            "y": { "type": "integer" }
                        },
                        "required": ["x", "y"],
                        "additionalProperties": false
                    }
                }
            }
        }
    }]);

    let mut c = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    feed(
        &mut c,
        &tok,
        "<tool_call>\n<function=navigate>\n<parameter=destination>\n{\"point\":{\"x\":10,\"y\":20},\"label\":\"home\"}\n</parameter>\n</function>\n</tool_call>",
    )
    .unwrap();
    assert!(c.is_token_allowed(EOS));

    let mut c_bad = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    assert!(
        feed(
            &mut c_bad,
            &tok,
            "<tool_call>\n<function=navigate>\n<parameter=destination>\n{\"point\":{\"x\":10,\"z\":20}}\n</parameter>"
        )
        .is_err()
    );
}

#[test]
fn parallel_tool_calls_and_auto_suffix_prohibition() {
    use w34_host::tool_constraints::{ToolChoice, ToolFormat};
    let (tok, factory) = setup();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "ping",
            "parameters": { "type": "object", "properties": {}, "additionalProperties": false }
        }
    }]);

    let two_calls = "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n<tool_call>\n<function=ping>\n</function>\n</tool_call>";

    // parallel_tool_calls = true allows two calls
    let mut multi = factory
        .tools_with_options(
            &tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            true,
            false,
        )
        .unwrap();
    feed(&mut multi, &tok, two_calls).unwrap();
    assert!(multi.is_token_allowed(EOS));

    // parallel_tool_calls = false allows one call and rejects a second call
    let mut single = factory
        .tools_with_options(
            &tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            false,
            false,
        )
        .unwrap();
    feed(
        &mut single,
        &tok,
        "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
    )
    .unwrap();
    assert!(single.is_token_allowed(EOS));
    assert!(feed(&mut single, &tok, "\n<tool_call>").is_err());

    // ToolChoice::Auto with parallel_tool_calls = false rejects second call and non-whitespace suffix
    let mut auto_single = factory
        .tools_with_options(&tools, &ToolChoice::Auto, ToolFormat::QwenXml, false, false)
        .unwrap();
    feed(
        &mut auto_single,
        &tok,
        "Let me check.\n<tool_call>\n<function=ping>\n</function>\n</tool_call>",
    )
    .unwrap();
    assert!(auto_single.is_token_allowed(EOS));
    assert!(feed(&mut auto_single, &tok, "\nTrailing prose is forbidden").is_err());
}

#[test]
fn choice_constraint_enforces_exact_literals_and_rollback() {
    let (tok, factory) = setup();
    let mut c = factory
        .choice(
            &["approve".into(), "reject".into(), "needs_review".into()],
            false,
        )
        .unwrap();
    let approve_ids = tok.encode("approve", false).unwrap();
    for &id in approve_ids.get_ids() {
        c.advance(id as i32).unwrap();
    }
    assert!(c.is_token_allowed(EOS));

    // Speculative rollback of all 7 tokens and switching to another valid choice succeeds
    c.rollback(approve_ids.len()).unwrap();
    feed(&mut c, &tok, "reject").unwrap();
    assert!(c.is_token_allowed(EOS));
    assert!(feed(&mut c, &tok, "_extra").is_err());
}

#[test]
fn xml_pydantic_anyof_nullable_strings_and_string_int_unions() {
    use serde_json::Value;
    use w34_host::{
        parser::Qwen3Parser,
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "configure",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "note": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "null" }
                        ]
                    },
                    "budget": {
                        "anyOf": [
                            { "type": "string", "enum": ["auto", "unlimited"] },
                            { "type": "integer", "minimum": 1, "maximum": 100 }
                        ]
                    }
                },
                "required": ["title", "note", "budget"],
                "additionalProperties": false
            }
        }
    }]);

    // 1. Unquoted string for `title` + unquoted string for `note` + unquoted enum `"auto"` for `budget`
    let mut c1 = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    let call_str = "<tool_call>\n<function=configure>\n<parameter=title>\nprod\n</parameter>\n<parameter=note>\nhello world\n</parameter>\n<parameter=budget>\nauto\n</parameter>\n</function>\n</tool_call>";
    feed(&mut c1, &tok, call_str).unwrap();
    assert!(c1.is_token_allowed(EOS));
    let p1 = Qwen3Parser::parse_complete_with_tools(call_str, false, Some(&tools));
    assert_eq!(
        serde_json::from_str::<Value>(&p1.tool_calls[0].function.arguments).unwrap(),
        json!({"title": "prod", "note": "hello world", "budget": "auto"})
    );

    // 2. Unquoted string for `title` + JSON `null` for `note` + integer `50` for `budget`
    let mut c2 = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    let call_null_int = "<tool_call>\n<function=configure>\n<parameter=title>\nstage\n</parameter>\n<parameter=note>\nnull\n</parameter>\n<parameter=budget>\n50\n</parameter>\n</function>\n</tool_call>";
    feed(&mut c2, &tok, call_null_int).unwrap();
    assert!(c2.is_token_allowed(EOS));
    let p2 = Qwen3Parser::parse_complete_with_tools(call_null_int, false, Some(&tools));
    assert_eq!(
        serde_json::from_str::<Value>(&p2.tool_calls[0].function.arguments).unwrap(),
        json!({"title": "stage", "note": null, "budget": 50})
    );

    // 3. Bare `null` for non-nullable `title` is rejected
    let mut c3 = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    let res3 = feed(
        &mut c3,
        &tok,
        "<tool_call>\n<function=configure>\n<parameter=title>\nnull\n</parameter>",
    );
    assert!(
        res3.is_err(),
        "grammar:\n{}",
        w34_host::tool_constraints::tool_grammar(
            &tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml
        )
        .unwrap()
    );

    // 4. Spaced XML delimiter variants (`< / parameter >`, `<parameter =note>`) are rejected in unquoted values
    let mut c_spaced_delim = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    assert!(
        feed(
            &mut c_spaced_delim,
            &tok,
            "<tool_call>\n<function=configure>\n<parameter=title>\nhello < / parameter >\n</parameter>"
        )
        .is_err()
    );

    // 5. Quoted values with leading whitespace (`<parameter=title> "prod" </parameter>`) decode to `"prod"`,
    // and unquoted enum values with layout whitespace canonicalize cleanly.
    let call_ws_quoted = "<tool_call>\n<function=configure>\n<parameter=title> \"prod\" </parameter>\n<parameter=note> \" hello \" </parameter>\n<parameter=budget>\n  auto\n</parameter>\n</function>\n</tool_call>";
    let p_ws = Qwen3Parser::parse_complete_with_tools(call_ws_quoted, false, Some(&tools));
    assert_eq!(
        serde_json::from_str::<Value>(&p_ws.tool_calls[0].function.arguments).unwrap(),
        json!({"title": "prod", "note": " hello ", "budget": "auto"})
    );

    // 6. `minLength: 1` rejects empty quoted string `""` while accepting non-empty unquoted/quoted strings
    let min_len_tools = json!([{
        "type": "function",
        "function": {
            "name": "non_empty",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "minLength": 1 }
                },
                "required": ["query"],
                "additionalProperties": false
            }
        }
    }]);
    let mut c_min_empty = factory
        .tools(
            &min_len_tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            false,
        )
        .unwrap();
    assert!(
        feed(
            &mut c_min_empty,
            &tok,
            "<tool_call>\n<function=non_empty>\n<parameter=query>\n\"\"\n</parameter>"
        )
        .is_err()
    );
    let mut c_min_ok = factory
        .tools(
            &min_len_tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            false,
        )
        .unwrap();
    feed(
        &mut c_min_ok,
        &tok,
        "<tool_call>\n<function=non_empty>\n<parameter=query>\nrust\n</parameter>\n</function>\n</tool_call>",
    )
    .unwrap();
    assert!(c_min_ok.is_token_allowed(EOS));

    // 7. `anyOf: [{"type": "string"}, {"type": "string", "minLength": 1}]` permits quoted empty string `""`,
    // whereas `oneOf` is never flattened into an unquoted union.
    let union_tools = json!([{
        "type": "function",
        "function": {
            "name": "check_unions",
            "parameters": {
                "type": "object",
                "properties": {
                    "any_str": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "string", "minLength": 1 }
                        ]
                    }
                },
                "required": ["any_str"],
                "additionalProperties": false
            }
        }
    }]);
    let mut c_union_ok = factory
        .tools(
            &union_tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            false,
        )
        .unwrap();
    feed(
        &mut c_union_ok,
        &tok,
        "<tool_call>\n<function=check_unions>\n<parameter=any_str>\n\"\"\n</parameter>\n</function>\n</tool_call>",
    )
    .unwrap();
    assert!(c_union_ok.is_token_allowed(EOS));

    let oneof_tools = json!([{
        "type": "function",
        "function": {
            "name": "check_oneof",
            "parameters": {
                "type": "object",
                "properties": {
                    "one_str": {
                        "oneOf": [
                            { "type": "string" },
                            { "type": "string", "minLength": 1 }
                        ]
                    }
                },
                "required": ["one_str"],
                "additionalProperties": false
            }
        }
    }]);
    let oneof_grammar = w34_host::tool_constraints::tool_grammar(
        &oneof_tools,
        &ToolChoice::Required,
        ToolFormat::QwenXml,
    )
    .unwrap();
    assert!(
        !oneof_grammar.contains("unquoted_xml_param_end"),
        "oneOf must not be flattened into unquoted_xml_param_end: {oneof_grammar}"
    );
    assert!(
        factory
            .tools(
                &oneof_tools,
                &ToolChoice::Required,
                ToolFormat::QwenXml,
                false
            )
            .is_err()
    );

    // 8. Unquoted enum branch rejects horizontal padding (`\nauto \n`) while accepting `\nauto\n` and ` "auto" `
    let mut c_enum_pad = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    assert!(
        feed(
            &mut c_enum_pad,
            &tok,
            "<tool_call>\n<function=configure>\n<parameter=title>\nprod\n</parameter>\n<parameter=note>\nok\n</parameter>\n<parameter=budget>\nauto \n</parameter>"
        )
        .is_err()
    );
}

#[test]
fn donor_qwen_xml_and_hermes_tool_oracle_parity() {
    use serde_json::Value;
    use w34_host::{
        parser::Qwen3Parser,
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tools/oracle/goldens/qwen_xml_tool_calls.json"
    ))
    .unwrap();
    let tools = &fixture["tools"];

    for case in fixture["valid_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let format = match case["format"].as_str().unwrap() {
            "xml" => ToolFormat::QwenXml,
            "json" => ToolFormat::HermesJson,
            other => panic!("unknown format {other}"),
        };
        let choice = ToolChoice::parse(&case["tool_choice"]).unwrap();
        let parallel = case["parallel_tool_calls"].as_bool().unwrap();
        let thinking = case["enable_thinking"].as_bool().unwrap();
        let raw = case["raw"].as_str().unwrap();

        let mut c = factory
            .tools_with_options(tools, &choice, format, parallel, thinking)
            .unwrap();
        let ids: Vec<i32> = tok
            .encode(raw, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();
        let mid = ids.len() / 2;
        for &id in &ids[..mid] {
            c.advance(id).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        let mid_mask = c.current_bitmask();
        for &id in &ids[mid..] {
            c.advance(id).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        assert!(c.is_token_allowed(EOS), "{name}: EOS not allowed at end");
        // Verify speculative rollback from completion back to midpoint
        c.rollback(ids.len() - mid).unwrap();
        assert_eq!(c.current_bitmask(), mid_mask, "{name}: rollback mismatch");
        for &id in &ids[mid..] {
            c.advance(id).unwrap();
        }
        assert!(c.is_token_allowed(EOS), "{name}: EOS after replay");

        let parsed = Qwen3Parser::parse_complete_with_tools(raw, thinking, Some(tools));
        assert_eq!(
            parsed.reasoning_content.as_deref(),
            case["expected_reasoning"].as_str(),
            "{name}: reasoning mismatch"
        );
        assert_eq!(
            parsed.content.as_deref(),
            case["expected_content"].as_str(),
            "{name}: content mismatch"
        );
        let expected_calls = case["expected_calls"].as_array().unwrap();
        assert_eq!(
            parsed.tool_calls.len(),
            expected_calls.len(),
            "{name}: tool call count"
        );
        for (actual, expected) in parsed.tool_calls.iter().zip(expected_calls.iter()) {
            assert_eq!(actual.function.name, expected["name"].as_str().unwrap());
            let actual_args: Value = serde_json::from_str(&actual.function.arguments).unwrap();
            assert_eq!(&actual_args, &expected["arguments"], "{name}: args");
        }
    }

    for case in fixture["invalid_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let format = match case["format"].as_str().unwrap() {
            "xml" => ToolFormat::QwenXml,
            "json" => ToolFormat::HermesJson,
            other => panic!("unknown format {other}"),
        };
        let choice = ToolChoice::parse(&case["tool_choice"]).unwrap();
        let parallel = case["parallel_tool_calls"].as_bool().unwrap();
        let thinking = case["enable_thinking"].as_bool().unwrap();
        let raw = case["raw"].as_str().unwrap();

        let mut c = factory
            .tools_with_options(tools, &choice, format, parallel, thinking)
            .unwrap();
        let accepted = feed(&mut c, &tok, raw).is_ok() && c.is_token_allowed(EOS);
        assert!(
            !accepted,
            "{name}: invalid tool call was accepted through EOS: {raw}"
        );
    }
}

#[test]
fn jump_forward_ff_tokens_and_speculative_draft_repair() {
    use serde_json::json;
    use w34_host::{
        parser::{SamplerV1, SamplingParams},
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();

    // 1. JSON Schema property key & const value jump-forward (plus compact whitespace_flexible=false)
    let const_schema = json!({
        "type": "object",
        "properties": {
            "customer_status": { "const": "verified_active" }
        },
        "required": ["customer_status"],
        "additionalProperties": false
    });
    let c_const = factory.json_schema(const_schema, false).unwrap();
    let mut sampler_const = SamplerV1::new(SamplingParams::default(), &[], Some(c_const));
    // Commit `{"` so the grammar enters the required property key `"customer_status"`
    for &id in tok.encode("{\"", false).unwrap().get_ids() {
        sampler_const.try_commit_token(id as i32).unwrap();
    }
    let (forced_key, stops) = sampler_const.collect_forced_drafts(64, 2, &[EOS]).unwrap();
    assert!(!stops);
    let key_str = tok
        .decode(
            &forced_key.iter().map(|&id| id as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap();
    assert_eq!(key_str, "customer_status\"");
    for &id in &forced_key {
        sampler_const.try_commit_token(id).unwrap();
    }
    // Commit `:"` to enter the const value `"verified_active"`
    for &id in tok.encode(":\"", false).unwrap().get_ids() {
        sampler_const.try_commit_token(id as i32).unwrap();
    }
    let (forced_val, _) = sampler_const.collect_forced_drafts(64, 20, &[EOS]).unwrap();
    let val_str = tok
        .decode(
            &forced_val.iter().map(|&id| id as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap();
    assert_eq!(val_str, "verified_active\"");

    // Also verify `compute_ff_tokens()` on a single-choice / deterministic literal constraint
    let mut c_choice = factory
        .choice(&["deterministic-output-token-span".into()], false)
        .unwrap();
    let ff = c_choice.compute_ff_tokens().unwrap();
    assert_eq!(
        tok.decode(&ff.iter().map(|&id| id as u32).collect::<Vec<_>>(), false)
            .unwrap(),
        "deterministic-output-token-span"
    );

    // 2. QwenXml tool call: forced prefix -> choice ("Paris" | "Rome") -> forced middle -> repair_and_force_drafts
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": { "type": "string", "enum": ["Paris", "Rome"] },
                    "days": { "type": "integer", "minimum": 1, "maximum": 7 }
                },
                "required": ["city", "days"],
                "additionalProperties": false
            }
        }
    }]);
    let c_xml = factory
        .tools_with_options(
            &tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            false,
            false,
        )
        .unwrap();
    let mut sampler = SamplerV1::new(SamplingParams::default(), &[], Some(c_xml));

    // Commit `<tool_call>` (token 259) and `\n` so the grammar enters `<function=forecast>`
    let open_ids = tok
        .encode("<tool_call>\n<", false)
        .unwrap()
        .get_ids()
        .to_vec();
    for &id in &open_ids {
        sampler.try_commit_token(id as i32).unwrap();
    }
    // Now `function=forecast>` is 100% forced!
    let (forced_fn, stops) = sampler
        .collect_forced_drafts(32, open_ids.len(), &[EOS])
        .unwrap();
    assert!(!stops);
    let forced_fn_str = tok
        .decode(
            &forced_fn.iter().map(|&id| id as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap();
    assert_eq!(forced_fn_str, "function=forecast>");
    for &id in &forced_fn {
        sampler.try_commit_token(id).unwrap();
    }

    // Feed `\n<parameter=city>\n` so we are right at the `city` value (`Paris` | `Rome`)
    for &id in tok.encode("\n<parameter=city>\n", false).unwrap().get_ids() {
        sampler.try_commit_token(id as i32).unwrap();
    }
    // At the start of `city`, multiple options exist (`"Paris"`, `"Rome"`, `Paris`, `Rome`), so not forced
    assert!(sampler.next_forced_token(&[EOS], 30).unwrap().is_none());

    // Suppose the drafter proposes `Paris\n<` followed by garbage tokens `[0, 0, 0, 0]`:
    // `repair_and_force_drafts` must keep `Paris\n<` and overwrite the garbage suffix
    // with the forced closing tag `/parameter>`!
    let mut drafts: Vec<i32> = tok
        .encode("Paris\n<", false)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    let valid_prefix_len = drafts.len();
    drafts.extend([0; 11]);
    let k_budget = drafts.len();
    sampler
        .repair_and_force_drafts(&mut drafts, k_budget, 30, &[EOS])
        .unwrap();
    assert_eq!(
        drafts.len(),
        k_budget,
        "repair_and_force_drafts must preserve full k_budget length for CUDA Graph verify"
    );
    let repaired_suffix = tok
        .decode(
            &drafts[valid_prefix_len..]
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>(),
            false,
        )
        .unwrap();
    assert_eq!(repaired_suffix, "/parameter>");
}

#[test]
fn json_schema_normalization_for_llguidance_handles_real_world_openapi_and_pydantic() {
    let (tok, factory) = setup();
    let schema = json!({
        "type": "dict",
        "properties": {
            "file_path": { "type": "str", "format": "path" },
            "secret": { "type": "string", "format": "password", "writeOnly": true },
            "count": { "type": "int", "format": "int64" },
            "ratio": { "type": "float" },
            "active": { "type": "bool" },
            "tags": {
                "type": "list",
                "items": { "type": "string" },
                "uniqueItems": true,
                "unevaluatedItems": false
            },
            "note": { "type": "string", "nullable": true },
            "labels": {
                "type": "object",
                "propertyNames": { "enum": ["env", "region"] },
                "additionalProperties": { "type": "string" }
            }
        },
        "required": ["file_path", "secret", "count", "ratio", "active", "tags", "note", "labels"],
        "unevaluatedProperties": false
    });

    let mut c1 = factory.json_schema(schema.clone(), false).unwrap();
    feed(
        &mut c1,
        &tok,
        r#"{"file_path":"/tmp/a","secret":"pw","count":42,"ratio":1.5,"active":true,"tags":["x"],"note":null,"labels":{"env":"prod"}}"#,
    )
    .unwrap();
    assert!(c1.is_token_allowed(EOS));

    // `unevaluatedProperties: false` lowered to `additionalProperties: false` rejects extra keys
    let mut c_extra = factory.json_schema(schema.clone(), false).unwrap();
    assert!(
        feed(
            &mut c_extra,
            &tok,
            r#"{"file_path":"/tmp/a","secret":"pw","count":42,"ratio":1.5,"active":true,"tags":["x"],"note":"hi","labels":{"env":"prod"},"extra":1}"#,
        )
        .is_err()
    );

    // `propertyNames.enum` lowered to explicit `properties` + `additionalProperties: false` rejects undeclared label keys
    let mut c_bad_label = factory.json_schema(schema, false).unwrap();
    assert!(
        feed(
            &mut c_bad_label,
            &tok,
            r#"{"file_path":"/tmp/a","secret":"pw","count":42,"ratio":1.5,"active":true,"tags":["x"],"note":"hi","labels":{"invalid_key":"prod"}}"#,
        )
        .is_err()
    );
}

#[test]
fn xml_unquoted_numeric_boolean_bracket_empty_and_length_pattern_format_strings() {
    use w34_host::{
        parser::Qwen3Parser,
        tool_constraints::{ToolChoice, ToolFormat},
    };
    let (tok, factory) = setup();
    let tools = json!([{
        "name": "inspect/item",
        "parameters": {
            "properties": {
                "zip": { "$ref": "#/$defs/ZipCode" },
                "opt_id": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                },
                "slug": { "type": "string", "pattern": "^[a-z0-9_-]+$" },
                "code": { "type": "string", "minLength": 3, "maxLength": 5 },
                "date": { "type": "string", "format": "date" }
            },
            "required": ["zip", "opt_id"],
            "$defs": {
                "ZipCode": { "type": "string" }
            }
        }
    }]);

    // 1. Unquoted numeric string ("02138") on `$ref` string and ("12345") on `Optional[str]`,
    // plus unquoted pattern, minLength/maxLength, and date format
    let call_1 = "<tool_call>\n<function=inspect/item>\n<parameter=zip>\n02138\n</parameter>\n<parameter=opt_id>\n12345\n</parameter>\n<parameter=slug>\nmy_slug-1\n</parameter>\n<parameter=code>\nabcd\n</parameter>\n<parameter=date>\n2026-04-19\n</parameter>\n</function>\n</tool_call>";
    let mut c1 = factory
        .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
        .unwrap();
    feed(&mut c1, &tok, call_1).unwrap();
    assert!(c1.is_token_allowed(EOS));
    let parsed_1 = Qwen3Parser::parse_complete_with_tools(call_1, false, Some(&tools));
    let args_1: serde_json::Value =
        serde_json::from_str(&parsed_1.tool_calls[0].function.arguments).unwrap();
    assert_eq!(
        args_1,
        json!({
            "zip": "02138",
            "opt_id": "12345",
            "slug": "my_slug-1",
            "code": "abcd",
            "date": "2026-04-19"
        }),
        "numeric-looking unquoted strings on string / Optional[str] / $ref string must stay JSON strings"
    );

    // 2. Boolean string `true`, JSON-quoted `"[INFO] ok"`, empty string `<parameter=zip></parameter>`, and bare `null` on `opt_id`
    for (zip_val, opt_val, expected_zip, expected_opt) in [
        ("true", "null", json!("true"), json!(null)),
        (
            "\"[INFO] ok\"",
            "\"null\"",
            json!("[INFO] ok"),
            json!("null"),
        ),
        ("\"{k: v}\"", "false", json!("{k: v}"), json!("false")),
        ("", "null", json!(""), json!(null)),
    ] {
        let xml = format!(
            "<tool_call>\n<function=inspect/item>\n<parameter=zip>{zip_val}</parameter>\n<parameter=opt_id>\n{opt_val}\n</parameter>\n</function>\n</tool_call>"
        );
        let mut c = factory
            .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
            .unwrap();
        feed(&mut c, &tok, &xml).unwrap_or_else(|e| panic!("rejected valid {xml}: {e}"));
        assert!(c.is_token_allowed(EOS));
        let parsed = Qwen3Parser::parse_complete_with_tools(&xml, false, Some(&tools));
        let args: serde_json::Value =
            serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args, json!({"zip": expected_zip, "opt_id": expected_opt}));
    }

    // 3. Rejections: unquoted `null` on non-nullable `zip`, too-short `code`, invalid `slug`, invalid `date`
    for bad_xml in [
        "<tool_call>\n<function=inspect/item>\n<parameter=zip>\nnull\n</parameter>\n<parameter=opt_id>\nok\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=inspect/item>\n<parameter=zip>\nok\n</parameter>\n<parameter=opt_id>\nok\n</parameter>\n<parameter=code>\nab\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=inspect/item>\n<parameter=zip>\nok\n</parameter>\n<parameter=opt_id>\nok\n</parameter>\n<parameter=slug>\nBAD SLUG\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=inspect/item>\n<parameter=zip>\nok\n</parameter>\n<parameter=opt_id>\nok\n</parameter>\n<parameter=date>\n2026/04/19\n</parameter>\n</function>\n</tool_call>",
    ] {
        let mut c_bad = factory
            .tools(&tools, &ToolChoice::Required, ToolFormat::QwenXml, false)
            .unwrap();
        assert!(
            feed(&mut c_bad, &tok, bad_xml).is_err(),
            "unexpectedly accepted {bad_xml}"
        );
    }

    // 4. Open-dictionary XML tool (`additionalProperties: {"type": "string", "minLength": 2}` without `properties`)
    // and `"parameters": null` / `{}`
    let open_tools = json!([
        {
            "type": "function",
            "function": {
                "name": "set_tags",
                "parameters": {
                    "type": "object",
                    "additionalProperties": { "type": "string", "minLength": 2 }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "noop",
                "parameters": null
            }
        }
    ]);
    let open_call = "<tool_call>\n<function=set_tags>\n<parameter=env>\nprod\n</parameter>\n<parameter=build>\n123\n</parameter>\n</function>\n</tool_call>";
    let mut c_open = factory
        .tools(
            &open_tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            false,
        )
        .unwrap();
    feed(&mut c_open, &tok, open_call).unwrap();
    assert!(c_open.is_token_allowed(EOS));
    let parsed_open = Qwen3Parser::parse_complete_with_tools(open_call, false, Some(&open_tools));
    let args_open: serde_json::Value =
        serde_json::from_str(&parsed_open.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args_open, json!({"env": "prod", "build": "123"}));

    // Too-short open-dictionary value (`minLength: 2` violated by `"x"`) is rejected
    let mut c_open_short = factory
        .tools(
            &open_tools,
            &ToolChoice::Required,
            ToolFormat::QwenXml,
            false,
        )
        .unwrap();
    assert!(
        feed(
            &mut c_open_short,
            &tok,
            "<tool_call>\n<function=set_tags>\n<parameter=env>\nx\n</parameter>\n</function>\n</tool_call>"
        )
        .is_err()
    );
}

#[test]
fn request_constraint_vllm_structured_outputs_and_non_destructive_runtime_probes() {
    use w34_host::{
        parser::{Qwen3Parser, SamplerV1, SamplingParams},
        request_constraints::{MAX_TOOLS_SPEC_BYTES, RequestConstraint, parse_request_tool_format},
        tool_constraints::ToolFormat,
    };
    let (tok, factory) = setup();

    // 1. `choice` allows leading whitespace (`\n\n`) after `</think>`
    let mut c_choice = factory
        .choice(&["APPROVED".into(), "REJECTED".into()], true)
        .unwrap();
    feed(&mut c_choice, &tok, "thinking...\n</think>\n\nAPPROVED").unwrap();
    assert!(c_choice.is_token_allowed(EOS));

    // 2. vLLM `StructuredOutputsParams` with `null` siblings, stringified `json`, `json_object: true`,
    // `extra_body`, `strict: false`, flat `response_format`, and `qwen3_coder` alias
    assert!(
        RequestConstraint::parse(&json!({
            "structured_outputs": {
                "json": "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}}}",
                "regex": null,
                "choice": null,
                "grammar": null,
                "structural_tag": null,
                "json_object": false,
                "disable_fallback": false
            }
        }))
        .unwrap()
        .is_some()
    );
    assert!(matches!(
        RequestConstraint::parse(&json!({
            "structured_outputs": {
                "json": null,
                "json_object": true
            }
        }))
        .unwrap(),
        Some(RequestConstraint::JsonObject)
    ));
    assert!(
        RequestConstraint::parse(&json!({
            "response_format": {
                "type": "json_schema",
                "schema": {"type": "object"}
            }
        }))
        .unwrap()
        .is_some()
    );
    assert!(
        RequestConstraint::parse(&json!({
            "extra_body": {
                "guided_choice": ["yes", "no"],
                "tool_call_format": "qwen3_coder"
            }
        }))
        .unwrap()
        .is_some()
    );
    assert_eq!(
        parse_request_tool_format(&json!({"extra_body": {"tool_call_format": "qwen3_coder"}}))
            .unwrap(),
        ToolFormat::QwenXml
    );

    // 3. Oversized `tools` degrades cleanly (`Ok(None)`) on `tool_choice: "auto"` / `"none"`
    // while failing closed (`Err`) on `tool_choice: "required"`
    let huge_desc = "x".repeat(MAX_TOOLS_SPEC_BYTES + 1024);
    let huge_tools = json!([{
        "type": "function",
        "function": { "name": "big", "description": huge_desc, "parameters": {"type": "object"} }
    }]);
    assert!(
        RequestConstraint::parse(&json!({
            "tools": huge_tools,
            "tool_choice": "auto"
        }))
        .unwrap()
        .is_none()
    );
    assert!(
        RequestConstraint::parse(&json!({
            "tools": huge_tools,
            "tool_choice": "required"
        }))
        .is_err()
    );

    // 4. `Qwen3Parser` parses multiple `<function=...>` blocks in one `<tool_call>` and
    // Hermes JSON with `"arguments": null`, `"parameters"`, and markdown fences
    let multi_fn = "<tool_call>\n<function=f1>\n<parameter=a>\n1\n</parameter>\n</function>\n<function=f2>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>";
    let parsed_multi = Qwen3Parser::parse_complete(multi_fn, false);
    assert_eq!(parsed_multi.tool_calls.len(), 2);
    assert_eq!(parsed_multi.tool_calls[0].function.name, "f1");
    assert_eq!(parsed_multi.tool_calls[1].function.name, "f2");

    let fenced_hermes =
        "<tool_call>\n```json\n{\"name\": \"ping\", \"arguments\": null}\n```\n</tool_call>";
    let parsed_hermes = Qwen3Parser::parse_complete(fenced_hermes, false);
    assert_eq!(parsed_hermes.tool_calls.len(), 1);
    assert_eq!(parsed_hermes.tool_calls[0].function.name, "ping");
    assert_eq!(parsed_hermes.tool_calls[0].function.arguments, "{}");

    // 5. `SamplerV1::append_gpu_row` does not ban EOS when the grammar forces EOS (`is_terminal_eos`)
    // even when `min_tokens` or `ignore_eos` is active
    let mut c_term = factory.choice(&["OK".into()], false).unwrap();
    feed(&mut c_term, &tok, "OK").unwrap();
    assert_eq!(c_term.single_allowed_token(), Some(EOS));
    let params = SamplingParams {
        min_tokens: 100,
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let sampler = SamplerV1::new(params, &[], Some(c_term));
    let mut bitmask_words = Vec::new();
    let mut sparse = Vec::new();
    let cfg = sampler
        .append_gpu_row(
            tok.get_vocab_size(true),
            &[EOS],
            0.5,
            0.5,
            -1,
            false,
            &mut bitmask_words,
            &mut sparse,
        )
        .unwrap();
    assert!(
        sparse.is_empty() && cfg.sparse_count == 0,
        "terminal grammar EOS must not be soft-banned by min_tokens or ignore_eos"
    );
}

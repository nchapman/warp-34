use serde_json::json;
use w34_host::server::{
    format_chatml_prompt, format_chatml_prompt_with_options, parse_openai_messages_multimodal,
};

#[test]
fn native_jinja_oracle_byte_parity() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tools/oracle/goldens/qwen_native_chatml.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let (messages, _) = parse_openai_messages_multimodal(&case["messages"]).unwrap();
        let actual = format_chatml_prompt_with_options(
            &messages,
            Some(&case["tools"]),
            case["enable_thinking"].as_bool().unwrap(),
            case["preserve_thinking"].as_bool().unwrap(),
        );
        assert_eq!(
            actual,
            case["expected"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn malformed_tool_history_and_template_roles_are_rejected() {
    for bad in [
        json!({"role":"assistant","tool_calls":{}}),
        json!({"role":"assistant","tool_calls":[{"function":{"name":"x","arguments":"not json"}}]}),
        json!({"role":"assistant","tool_calls":[{"function":{"name":"x","arguments":[]}}]}),
        json!({"role":"assistant","tool_calls":[{"function":{"name":"x>bad","arguments":{}}}]}),
        json!({"role":"assistant","tool_calls":[{"function":{"name":"x","arguments":{"bad>key":0}}}]}),
        json!({"role":"tool","tool_calls":[{"function":{"name":"x","arguments":{}}}]}),
        json!({"role":"system","content":"late system"}),
        json!({"role":"unknown","content":"bad role"}),
    ] {
        assert!(
            parse_openai_messages_multimodal(&json!([
                {"role":"user","content":"hello"}, bad
            ]))
            .is_err()
        );
    }
    assert!(
        parse_openai_messages_multimodal(&json!([
            {"role":"system","content":"no user"}
        ]))
        .is_err()
    );
}

#[test]
fn system_and_tools_include_native_qwen_instructions() {
    let (messages, _) = parse_openai_messages_multimodal(&json!([
        {"role":"system", "content":"  Be precise.  "},
        {"role":"user", "content":"Weather?"}
    ]))
    .unwrap();
    let tools =
        json!([{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]);
    let prompt = format_chatml_prompt(&messages, Some(&tools), false);
    assert!(prompt.starts_with("<|im_start|>system\n# Tools\n\n"));
    assert!(prompt.contains("<function=example_function_name>"));
    assert!(prompt.contains("Required parameters MUST be specified"));
    assert!(prompt.contains("</IMPORTANT>\n\nBe precise.<|im_end|>\n"));
    assert_eq!(prompt.matches("<|im_start|>system").count(), 1);
}

#[test]
fn assistant_tool_history_and_consecutive_responses_are_not_dropped() {
    let (messages, _) = parse_openai_messages_multimodal(&json!([
        {"role":"user","content":"Compare weather."},
        {"role":"assistant","content":null,"reasoning_content":"Need two cities.","tool_calls":[
            {"id":"a","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\",\"days\":2}"}},
            {"id":"b","type":"function","function":{"name":"weather","arguments":{"city":"Rome"}}}
        ]},
        {"role":"tool","tool_call_id":"a","content":"  sunny  "},
        {"role":"tool","tool_call_id":"b","content":"rain"}
    ])).unwrap();
    let prompt = format_chatml_prompt(&messages, None, true);
    assert!(prompt.contains("<|im_start|>assistant\n<think>\nNeed two cities.\n</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>\n<tool_call>"));
    assert!(prompt.contains("<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n<tool_response>\nrain\n</tool_response><|im_end|>\n"));
    assert!(!prompt.contains("<|im_start|>tool"));
}

#[test]
fn old_reasoning_is_removed_but_current_turn_reasoning_is_preserved() {
    let (messages, _) = parse_openai_messages_multimodal(&json!([
        {"role":"user","content":"first"},
        {"role":"assistant","content":"<think>\nold thought\n</think>\n\nold answer"},
        {"role":"user","content":"next"},
        {"role":"assistant","content":"<think>\nnew thought\n</think>\n\nnew answer"}
    ]))
    .unwrap();
    let prompt = format_chatml_prompt(&messages, None, false);
    assert!(prompt.contains("<|im_start|>assistant\nold answer<|im_end|>"));
    assert!(!prompt.contains("old thought"));
    assert!(
        prompt.contains(
            "<|im_start|>assistant\n<think>\nnew thought\n</think>\n\nnew answer<|im_end|>"
        )
    );
}

#[test]
fn hermes_json_tool_format_prompt_and_history_serialization() {
    use w34_host::{chatml::format_chatml_prompt_with_tool_format, tool_constraints::ToolFormat};
    let (messages, _) = parse_openai_messages_multimodal(&json!([
        {"role":"user","content":"Weather in Paris?"},
        {"role":"assistant","content":null,"tool_calls":[
            {"id":"c1","type":"function","function":{"name":"weather","arguments":{"city":"Paris","days":2}}}
        ]},
        {"role":"tool","tool_call_id":"c1","content":"sunny"}
    ]))
    .unwrap();
    let tools = json!([{
        "type":"function",
        "function":{"name":"weather","parameters":{"type":"object"}}
    }]);
    let prompt = format_chatml_prompt_with_tool_format(
        &messages,
        Some(&tools),
        false,
        false,
        ToolFormat::HermesJson,
    );
    assert!(prompt.contains("return a json object with function name and arguments within <tool_call></tool_call> XML tags"));
    assert!(prompt.contains("<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n</tool_call><|im_end|>\n"));
}

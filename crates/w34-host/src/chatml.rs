//! Qwen 3.5/3.6 chat-template serialization, including multi-step tool history.
//!
//! Mirrors the checkpoints' `chat_template.jinja` text branch. Images are expanded
//! into model-specific patch tokens by the request parser before reaching here.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatToolCall {
    pub function: ChatFunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatFunctionCall {
    pub name: String,
    #[serde(default, deserialize_with = "deserialize_arguments")]
    pub arguments: Map<String, Value>,
}

fn deserialize_arguments<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Map<String, Value>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    let value = if let Value::String(text) = value {
        serde_json::from_str(&text).map_err(serde::de::Error::custom)?
    } else {
        value
    };
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(serde::de::Error::custom(
            "tool arguments must be a JSON object",
        )),
    }
}

pub fn parse_history_tools(value: Option<&Value>) -> Result<Vec<ChatToolCall>, String> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(Vec::new());
    };
    let calls: Vec<ChatToolCall> = serde_json::from_value(value.clone())
        .map_err(|e| format!("invalid assistant tool_calls: {e}"))?;
    for call in &calls {
        if !valid_xml_name(&call.function.name)
            || call
                .function
                .arguments
                .keys()
                .any(|key| !valid_xml_name(key))
        {
            return Err("tool and parameter names must be nonempty and contain no XML delimiters or whitespace".into());
        }
    }
    Ok(calls)
}

fn valid_xml_name(name: &str) -> bool {
    !name.is_empty()
        && !name
            .chars()
            .any(|c| c.is_whitespace() || "<>=/".contains(c))
}

// Transformers overrides Jinja's HTML-escaping tojson with json.dumps, using
// UTF-8, insertion-order keys, and a space after commas and colons.
fn template_json(value: &Value) -> String {
    match value {
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(template_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(key).unwrap(),
                        template_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => value.to_string(),
    }
}

const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Formats a list of chat messages into a ChatML prompt string.
///
/// # Examples
///
/// ```
/// use w34_host::chatml::{ChatMessage, format_chatml_prompt};
///
/// let messages = vec![ChatMessage {
///     role: "user".into(),
///     content: Some("Hello".into()),
///     reasoning_content: None,
///     tool_calls: Vec::new(),
/// }];
/// let prompt = format_chatml_prompt(&messages, None, false);
/// assert!(prompt.contains("Hello"));
/// ```
pub fn format_chatml_prompt(
    messages: &[ChatMessage],
    tools: Option<&Value>,
    enable_thinking: bool,
) -> String {
    format_chatml_prompt_with_options(messages, tools, enable_thinking, false)
}

pub fn format_chatml_prompt_with_options(
    messages: &[ChatMessage],
    tools: Option<&Value>,
    enable_thinking: bool,
    preserve_thinking: bool,
) -> String {
    let mut out = String::new();
    let system = messages.first().filter(|m| m.role == "system");
    let tools = tools.and_then(Value::as_array).filter(|t| !t.is_empty());
    if let Some(tools) = tools {
        out.push_str(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>",
        );
        for tool in tools {
            out.push('\n');
            out.push_str(&template_json(tool));
        }
        out.push_str("\n</tools>");
        out.push_str(TOOL_INSTRUCTIONS);
        if let Some(content) = system
            .and_then(|m| m.content.as_deref())
            .map(str::trim)
            .filter(|c| !c.is_empty())
        {
            out.push_str("\n\n");
            out.push_str(content);
        }
        out.push_str("<|im_end|>\n");
    } else if let Some(system) = system {
        out.push_str("<|im_start|>system\n");
        out.push_str(system.content.as_deref().unwrap_or("").trim());
        out.push_str("<|im_end|>\n");
    }
    let last_query = messages.iter().rposition(|m| {
        let content = m.content.as_deref().unwrap_or("").trim();
        m.role == "user"
            && !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>"))
    });
    for (index, message) in messages.iter().enumerate() {
        let mut content = message.content.as_deref().unwrap_or("").trim();
        match message.role.as_str() {
            "system" => continue,
            "tool" => {
                if index == 0 || messages[index - 1].role != "tool" {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if messages.get(index + 1).is_none_or(|m| m.role != "tool") {
                    out.push_str("<|im_end|>\n");
                }
            }
            "assistant" => {
                let reasoning = if let Some(reasoning) = message.reasoning_content.as_deref() {
                    reasoning
                } else if let Some((before, _)) = content.split_once("</think>") {
                    let reasoning = before.rsplit("<think>").next().unwrap_or(before);
                    content = content
                        .rsplit("</think>")
                        .next()
                        .unwrap_or("")
                        .trim_start_matches('\n');
                    reasoning
                } else {
                    ""
                };
                out.push_str("<|im_start|>assistant\n");
                if preserve_thinking || last_query.is_some_and(|last| index > last) {
                    out.push_str("<think>\n");
                    out.push_str(reasoning.trim());
                    out.push_str("\n</think>\n\n");
                }
                out.push_str(content);
                for (call_index, call) in message.tool_calls.iter().enumerate() {
                    if call_index > 0 {
                        out.push('\n');
                    } else if !content.trim().is_empty() {
                        out.push_str("\n\n");
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&call.function.name);
                    out.push_str(">\n");
                    for (name, value) in &call.function.arguments {
                        out.push_str("<parameter=");
                        out.push_str(name);
                        out.push_str(">\n");
                        match value {
                            Value::String(text) => out.push_str(text),
                            _ => out.push_str(&template_json(value)),
                        }
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            _ => {
                out.push_str("<|im_start|>");
                out.push_str(&message.role);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }
    out.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        out.push_str("<think>\n");
    } else {
        out.push_str("<think>\n\n</think>\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_user_and_assistant_formatting() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("What is the capital of France?".into()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some("The capital of France is Paris.".into()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
        ];

        let prompt = format_chatml_prompt(&messages, None, false);
        assert!(prompt.contains("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nThe capital of France is Paris.<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_chatml_thinking_enabled_vs_disabled() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some("Hello".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
        }];

        // Thinking enabled: leaves open `<think>\n` tag for assistant to fill
        let p_enabled = format_chatml_prompt(&messages, None, true);
        assert!(p_enabled.ends_with("<|im_start|>assistant\n<think>\n"));

        // Thinking disabled: emits closed `<think>\n\n</think>\n\n`
        let p_disabled = format_chatml_prompt(&messages, None, false);
        assert!(p_disabled.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_chatml_reasoning_stripping_and_preservation() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("Question 1".into()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some("Answer 1".into()),
                reasoning_content: Some("Old private reasoning that should be stripped".into()),
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Question 2".into()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some("Answer 2".into()),
                reasoning_content: Some("Current turn reasoning".into()),
                tool_calls: Vec::new(),
            },
        ];

        // Default: strips older reasoning
        let p_default = format_chatml_prompt_with_options(&messages, None, true, false);
        assert!(!p_default.contains("Old private reasoning that should be stripped"));
        assert!(p_default.contains("Current turn reasoning"));

        // Preserve thinking: retains all reasoning
        let p_preserved = format_chatml_prompt_with_options(&messages, None, true, true);
        assert!(p_preserved.contains("Old private reasoning that should be stripped"));
        assert!(p_preserved.contains("Current turn reasoning"));
    }

    #[test]
    fn test_chatml_tool_calls_and_instructions() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "lookup_stock",
                "description": "Look up stock ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"}
                    },
                    "required": ["ticker"]
                }
            }
        }]);

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some("What is AAPL stock?".into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
        }];

        let prompt = format_chatml_prompt(&messages, Some(&tools), false);
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("lookup_stock"));
        assert!(prompt.contains(TOOL_INSTRUCTIONS));
    }

    #[test]
    fn test_parse_history_tools_validation() {
        // Valid tool call
        let valid_json = serde_json::json!([{
            "function": {
                "name": "search",
                "arguments": {"query": "rust"}
            }
        }]);
        let parsed = parse_history_tools(Some(&valid_json)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].function.name, "search");

        // Invalid tool call with XML delimiter in function name
        let invalid_json = serde_json::json!([{
            "function": {
                "name": "bad<tag>",
                "arguments": {}
            }
        }]);
        assert!(parse_history_tools(Some(&invalid_json)).is_err());
    }
}

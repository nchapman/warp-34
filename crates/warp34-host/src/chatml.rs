//! Qwen 3.5/3.6 chat-template serialization, including multi-step tool history.
//!
//! Mirrors the checkpoints' `chat_template.jinja` text branch. Images are expanded
//! into model-specific patch tokens by the request parser before reaching here.

use std::sync::{Arc, OnceLock};

use minijinja::value::Kwargs;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

/// Embedded fallback Qwen ChatML Jinja2 template used when a checkpoint omits
/// both `chat_template.jinja` and `tokenizer_config.json["chat_template"]`.
pub const DEFAULT_QWEN_CHAT_TEMPLATE: &str = include_str!("default_qwen_chat_template.jinja");

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

fn tojson_filter(value: minijinja::Value, kwargs: Kwargs) -> Result<String, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent").ok().flatten();
    let _ensure_ascii: Option<bool> = kwargs.get("ensure_ascii").ok().flatten();
    let _sort_keys: Option<bool> = kwargs.get("sort_keys").ok().flatten();
    let json_val = serde_json::to_value(&value).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson serialization failed: {e}"),
        )
    })?;
    if indent.is_some() {
        serde_json::to_string_pretty(&json_val).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("tojson pretty serialization failed: {e}"),
            )
        })
    } else {
        Ok(template_json(&json_val))
    }
}

/// Compiled Jinja2 chat template loaded from a model checkpoint (`chat_template.jinja`
/// or `tokenizer_config.json`), with an embedded Qwen ChatML template as fallback.
#[derive(Clone)]
pub struct ChatTemplate {
    env: Arc<minijinja::Environment<'static>>,
    source_name: String,
    bos_token: String,
    eos_token: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("source_name", &self.source_name)
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .finish()
    }
}

impl ChatTemplate {
    /// Returns the shared embedded default Qwen ChatML Jinja2 template.
    pub fn default_qwen() -> Self {
        static DEFAULT: OnceLock<ChatTemplate> = OnceLock::new();
        DEFAULT
            .get_or_init(|| {
                Self::from_source_with_tokens(
                    DEFAULT_QWEN_CHAT_TEMPLATE,
                    "default-qwen",
                    String::new(),
                    "<|im_end|>".to_string(),
                )
                .expect("embedded default Qwen chat template must compile")
            })
            .clone()
    }

    /// Compiles an arbitrary Jinja2 chat template string.
    pub fn from_source(source: &str, name: &str) -> Result<Self, String> {
        Self::from_source_with_tokens(source, name, String::new(), "<|im_end|>".to_string())
    }

    fn from_source_with_tokens(
        source: &str,
        name: &str,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self, String> {
        let mut env = minijinja::Environment::new();
        env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Bound template recursion and instruction count against pathological chat
        // templates (matches vLLM #50300 / GHSA-4hhp-h66f-j5j7 resource-exhaustion fix).
        env.set_recursion_limit(100);
        env.set_fuel(Some(10_000_000));
        env.add_filter("tojson", tojson_filter);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_template_owned("chat".to_string(), source.to_string())
            .map_err(|e| format!("failed to compile chat template '{name}': {e}"))?;
        Ok(Self {
            env: Arc::new(env),
            source_name: name.to_string(),
            bos_token,
            eos_token,
        })
    }

    /// Loads the chat template from a HuggingFace model directory following `vLLM` precedence:
    /// 1. `chat_template.jinja`
    /// 2. `tokenizer_config.json` -> `chat_template`
    /// 3. Embedded default Qwen ChatML template
    pub fn from_model_dir(model_dir: &std::path::Path) -> Self {
        let tc_path = model_dir.join("tokenizer_config.json");
        let tc_text = match std::fs::read_to_string(&tc_path) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                eprintln!("warning: failed to read {}: {e}", tc_path.display());
                None
            }
        };
        if let Some(ref text) = tc_text
            && let Err(e) = serde_json::from_str::<Value>(text)
        {
            eprintln!("warning: malformed {}: {e}", tc_path.display());
        }
        let (bos_token, eos_token) = tc_text.as_deref().map_or_else(
            || (String::new(), "<|im_end|>".to_string()),
            extract_special_tokens_from_tokenizer_config,
        );

        let jinja_path = model_dir.join("chat_template.jinja");
        match std::fs::read_to_string(&jinja_path) {
            Ok(source) => {
                match Self::from_source_with_tokens(
                    &source,
                    "chat_template.jinja",
                    bos_token.clone(),
                    eos_token.clone(),
                ) {
                    Ok(tmpl) => return tmpl,
                    Err(err) => {
                        eprintln!(
                            "warning: {err}; falling back to embedded default Qwen chat template"
                        );
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                eprintln!(
                    "warning: failed to read {}: {e}; falling back to tokenizer_config/default",
                    jinja_path.display()
                );
            }
        }
        if let Some(source) = tc_text
            .as_deref()
            .and_then(extract_chat_template_from_tokenizer_config)
        {
            match Self::from_source_with_tokens(
                &source,
                "tokenizer_config.json",
                bos_token,
                eos_token,
            ) {
                Ok(tmpl) => return tmpl,
                Err(err) => {
                    eprintln!(
                        "warning: {err}; falling back to embedded default Qwen chat template"
                    );
                }
            }
        }
        Self::default_qwen()
    }

    /// Name of the loaded template source (`chat_template.jinja`, `tokenizer_config.json`, or `default-qwen`).
    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    /// Renders `messages` and optional `tools` with arbitrary `chat_template_kwargs`.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: Option<&Value>,
        add_generation_prompt: bool,
        kwargs: &Map<String, Value>,
    ) -> Result<String, String> {
        let mut ctx = kwargs.clone();
        ctx.insert(
            "messages".into(),
            serde_json::to_value(messages).map_err(|e| e.to_string())?,
        );
        ctx.insert(
            "add_generation_prompt".into(),
            Value::Bool(add_generation_prompt),
        );
        ctx.entry("bos_token".to_string())
            .or_insert_with(|| Value::String(self.bos_token.clone()));
        ctx.entry("eos_token".to_string())
            .or_insert_with(|| Value::String(self.eos_token.clone()));
        if let Some(t) = tools.filter(|v| v.as_array().is_some_and(|a| !a.is_empty())) {
            ctx.insert("tools".into(), t.clone());
        } else {
            ctx.remove("tools");
        }
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| format!("missing compiled chat template: {e}"))?;
        tmpl.render(minijinja::value::Serde(&ctx))
            .map_err(|e| format!("chat template render failed ({}): {e}", self.source_name))
    }
}

fn extract_special_tokens_from_tokenizer_config(json_text: &str) -> (String, String) {
    let Ok(root) = serde_json::from_str::<Value>(json_text) else {
        return (String::new(), "<|im_end|>".to_string());
    };
    let token_str = |key: &str| -> Option<String> {
        let v = root.get(key)?;
        if let Some(s) = v.as_str() {
            return Some(s.to_owned());
        }
        v.get("content").and_then(Value::as_str).map(str::to_owned)
    };
    let bos = token_str("bos_token").unwrap_or_default();
    let eos = token_str("eos_token").unwrap_or_else(|| "<|im_end|>".to_string());
    (bos, eos)
}

/// Formats a list of chat messages into a ChatML prompt string using the embedded default template.
///
/// # Examples
///
/// ```
/// use warp34_host::chatml::{ChatMessage, format_chatml_prompt};
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
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), Value::Bool(enable_thinking));
    kwargs.insert("preserve_thinking".into(), Value::Bool(preserve_thinking));
    match ChatTemplate::default_qwen().render(messages, tools, true, &kwargs) {
        Ok(rendered) => rendered,
        Err(err) => {
            eprintln!("warning: default chat template render failed: {err}");
            let mut fallback = String::new();
            for m in messages {
                fallback.push_str("<|im_start|>");
                fallback.push_str(&m.role);
                fallback.push('\n');
                if let Some(c) = m.content.as_deref() {
                    fallback.push_str(c.trim());
                }
                fallback.push_str("<|im_end|>\n");
            }
            fallback.push_str("<|im_start|>assistant\n");
            if enable_thinking {
                fallback.push_str("<think>\n");
            } else {
                fallback.push_str("<think>\n\n</think>\n\n");
            }
            fallback
        }
    }
}

/// Inspects a checkpoint directory's `chat_template.jinja` (falling back to
/// `tokenizer_config.json` `chat_template`) to determine whether the model's
/// Jinja chat template defaults `preserve_thinking` / `_preserve_thinking` to `true`
/// (e.g. `Qwen3.8` and `qwen3.8-froggeric-v22.5` templates) or `false` (official `Qwen3.5` / `Qwen3.6`).
pub fn detect_default_preserve_thinking(model_dir: &std::path::Path) -> bool {
    let tmpl = std::fs::read_to_string(model_dir.join("chat_template.jinja"))
        .ok()
        .or_else(|| {
            let tc = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()?;
            extract_chat_template_from_tokenizer_config(&tc)
        });
    detect_preserve_thinking_from_template(tmpl.as_deref().unwrap_or(""))
}

fn extract_chat_template_from_tokenizer_config(json_text: &str) -> Option<String> {
    let root = serde_json::from_str::<serde_json::Value>(json_text).ok()?;
    let ct = root.get("chat_template")?;
    if let Some(s) = ct.as_str() {
        return Some(s.to_owned());
    }
    let arr = ct.as_array()?;
    arr.iter()
        .find(|item| item.get("name").and_then(|n| n.as_str()) == Some("default"))
        .or_else(|| arr.first())
        .and_then(|item| item.get("template").and_then(|t| t.as_str()))
        .map(str::to_owned)
}

pub fn detect_preserve_thinking_from_template(template: &str) -> bool {
    let norm = template
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let compact: String = norm.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    compact.contains("_preserve_thinking=true")
        || compact.contains("preserve_thinking|default(true)")
        || norm.contains("preserve_thinking is undefined or ")
        || (norm.contains("preserve_thinking is undefined")
            && compact.contains("setpreserve_thinking=true"))
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
        assert!(prompt.contains(
            "If you choose to call a function ONLY reply in the following format with NO suffix:"
        ));
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

    #[test]
    fn test_detect_preserve_thinking_from_template_variants() {
        // Official Qwen3.5 (no preserve_thinking flag) -> false
        assert!(!detect_preserve_thinking_from_template(
            "{%- if loop.index0 > ns.last_query_index %}"
        ));
        // Official Qwen3.6 (explicitly requires preserve_thinking is defined and true) -> false
        assert!(!detect_preserve_thinking_from_template(
            "{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}"
        ));
        // Qwen3.8 standard template -> true
        assert!(detect_preserve_thinking_from_template(
            "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}"
        ));
        // froggeric v22.5 template -> true
        assert!(detect_preserve_thinking_from_template(
            "{%- if preserve_reasoning is defined %}{%- set _preserve_thinking = preserve_reasoning %}{%- else %}{%- set _preserve_thinking = true %}{%- endif %}"
        ));
        // default(true) filter template (including uppercase True / tight spacing) -> true
        assert!(detect_preserve_thinking_from_template(
            "{%- set preserve_thinking = preserve_thinking | default(true) %}"
        ));
        assert!(detect_preserve_thinking_from_template(
            "{%- set preserve_thinking=preserve_thinking|default( True ) %}"
        ));
        // Explicit undefined -> false guard must NOT false-positive -> false
        assert!(!detect_preserve_thinking_from_template(
            "{%- if preserve_thinking is undefined %}{%- set preserve_thinking = false %}{%- endif %}"
        ));
        // Modern list-form `chat_template` in `tokenizer_config.json`
        let list_cfg = r#"{"chat_template":[{"name":"tool_use","template":"x"},{"name":"default","template":"{%- set _preserve_thinking = true %}"}]}"#;
        let extracted = extract_chat_template_from_tokenizer_config(list_cfg).unwrap();
        assert!(detect_preserve_thinking_from_template(&extracted));
    }

    #[test]
    fn test_multi_turn_tool_trajectory_roundtrip_prefix_parity() {
        use crate::parser::Qwen3Parser;
        use serde_json::json;

        let tools_vec = vec![json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"}
                    },
                    "required": ["command"]
                }
            }
        })];
        let tools_val = json!(tools_vec);

        let mut history = vec![
            ChatMessage {
                role: "system".into(),
                content: Some("You are a coding agent.".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Check the Node version and list files.".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
        ];

        let prompt_t1 = format_chatml_prompt_with_options(&history, Some(&tools_val), true, true);
        let raw_assistant_t1 = "Let me check the environment first.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nnode --version && ls -la\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>";

        // Parse via streaming chunks with native tool schema enabled.
        let mut parser = Qwen3Parser::with_tools(true, &tools_vec).unwrap();
        let mut streamed_reasoning = String::new();
        let mut streamed_content = String::new();
        let mut streamed_calls = Vec::new();
        for chunk in [
            "Let me check ",
            "the environment first.\n</think>",
            "\n\n",
            "<tool_call>\n<function=bash>\n<parameter=command>\nnode --version && ls -la\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>",
        ] {
            let d = parser.push_str(chunk);
            if let Some(r) = d.reasoning_delta {
                streamed_reasoning.push_str(&r);
            }
            if let Some(c) = d.content_delta {
                streamed_content.push_str(&c);
            }
            streamed_calls.extend(d.completed_tool_calls);
        }
        let (df, turn) = parser.finish();
        if let Some(r) = df.reasoning_delta {
            streamed_reasoning.push_str(&r);
        }
        if let Some(c) = df.content_delta {
            streamed_content.push_str(&c);
        }
        streamed_calls.extend(df.completed_tool_calls);

        assert!(
            streamed_content.is_empty(),
            "no spurious content delta allowed: {streamed_content:?}"
        );
        assert_eq!(turn.content, None);
        assert_eq!(streamed_calls.len(), 1);

        let tool_calls_json = serde_json::to_value(&streamed_calls).unwrap();
        let history_calls = parse_history_tools(Some(&tool_calls_json)).unwrap();
        history.push(ChatMessage {
            role: "assistant".into(),
            content: (!streamed_content.is_empty()).then_some(streamed_content),
            reasoning_content: Some(streamed_reasoning),
            tool_calls: history_calls,
        });
        history.push(ChatMessage {
            role: "tool".into(),
            content: Some("v24.9.0\ntotal 0".into()),
            reasoning_content: None,
            tool_calls: vec![],
        });

        let prompt_t2 = format_chatml_prompt_with_options(&history, Some(&tools_val), true, true);
        let expected_prefix = format!("{prompt_t1}{raw_assistant_t1}<|im_end|>\n");
        assert!(
            prompt_t2.starts_with(&expected_prefix),
            "Turn 2 prompt must start with exact Turn 1 prompt + output prefix!\nprompt_t2:\n{prompt_t2}\nexpected_prefix:\n{expected_prefix}"
        );
    }

    #[test]
    fn test_chat_template_reasoning_effort_and_model_dir_precedence() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: Some("Be helpful.".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Hi".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
        ];
        let tmpl = ChatTemplate::default_qwen();

        let mut kw_low = Map::new();
        kw_low.insert("reasoning_effort".into(), Value::String("low".into()));
        let p_low = tmpl.render(&msgs, None, true, &kw_low).unwrap();
        assert!(p_low.contains("Reasoning effort is set to low."));
        assert!(p_low.ends_with("<|im_start|>assistant\n<think>\n"));

        let mut kw_high = Map::new();
        kw_high.insert("reasoning_effort".into(), Value::String("high".into()));
        let p_high = tmpl.render(&msgs, None, true, &kw_high).unwrap();
        assert!(p_high.contains("Reasoning effort is set to xhigh."));
        assert!(p_high.ends_with("<|im_start|>assistant\n<think>\n"));

        let mut kw_med = Map::new();
        kw_med.insert("reasoning_effort".into(), Value::String("medium".into()));
        let p_med = tmpl.render(&msgs, None, true, &kw_med).unwrap();
        assert!(!p_med.contains("Reasoning effort is set to"));
        assert!(p_med.ends_with("<|im_start|>assistant\n<think>\n"));

        let mut kw_none = Map::new();
        kw_none.insert("reasoning_effort".into(), Value::String("none".into()));
        let p_none = tmpl.render(&msgs, None, true, &kw_none).unwrap();
        assert!(p_none.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));

        // Precedence: chat_template.jinja > tokenizer_config.json > default-qwen
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            ChatTemplate::from_model_dir(dir.path()).source_name(),
            "default-qwen"
        );

        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "TC:{{ messages[0].content }}:{{ reasoning_effort | default('unset') }}"}"#,
        )
        .unwrap();
        let tc_tmpl = ChatTemplate::from_model_dir(dir.path());
        assert_eq!(tc_tmpl.source_name(), "tokenizer_config.json");
        assert_eq!(
            tc_tmpl.render(&msgs, None, true, &kw_low).unwrap(),
            "TC:Be helpful.:low"
        );

        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "JINJA:{{ messages[1].content }}:{{ reasoning_effort | default('unset') }}",
        )
        .unwrap();
        let j_tmpl = ChatTemplate::from_model_dir(dir.path());
        assert_eq!(j_tmpl.source_name(), "chat_template.jinja");
        assert_eq!(
            j_tmpl.render(&msgs, None, true, &kw_high).unwrap(),
            "JINJA:Hi:high"
        );

        // Negative step slicing (used by Qwen3.5/3.6/3.8 chat_template.jinja: `_msgs[::-1]`)
        let slice_tmpl = ChatTemplate::from_source(
            "{%- for m in messages[::-1] %}{{ m.role }},{%- endfor %}",
            "slice-test",
        )
        .unwrap();
        assert_eq!(
            slice_tmpl.render(&msgs, None, false, &Map::new()).unwrap(),
            "user,system,"
        );
    }

    #[test]
    fn test_real_cached_qwen38_chat_template_if_present() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let snap = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--nchapman--Qwen3.8-27B-W4A16-HGPTQ/snapshots/cc31dcc0856d9b8b53ef4f22253f6c753f0f692e",
        );
        if !snap.join("chat_template.jinja").is_file() {
            return;
        }
        let tmpl = ChatTemplate::from_model_dir(&snap);
        assert_eq!(tmpl.source_name(), "chat_template.jinja");
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: Some("You are helpful.".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Hello".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some("Hi there!".into()),
                reasoning_content: Some("Thinking...".into()),
                tool_calls: vec![],
            },
            ChatMessage {
                role: "user".into(),
                content: Some("Run a tool".into()),
                reasoning_content: None,
                tool_calls: vec![],
            },
        ];
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run shell",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}
            }
        }]);
        let mut kw = Map::new();
        kw.insert("enable_thinking".into(), Value::Bool(true));
        kw.insert("reasoning_effort".into(), Value::String("low".into()));
        let rendered = tmpl.render(&msgs, Some(&tools), true, &kw).unwrap();
        assert!(rendered.contains("Reasoning effort is set to low."));
        assert!(rendered.contains("<think>\nThinking...\n</think>"));
        assert!(rendered.contains("# Tools"));
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn chat_template_fuel_bounds_runaway_loops() {
        let tmpl = ChatTemplate::from_source_with_tokens(
            "{% for i in range(100000) %}{% for j in range(100000) %}{{ i }}{% endfor %}{% endfor %}",
            "runaway",
            String::new(),
            String::new(),
        )
        .unwrap();
        let err = tmpl
            .render(&[], None, false, &Map::new())
            .expect_err("runaway template must fail on fuel exhaustion");
        assert!(
            err.to_lowercase().contains("fuel"),
            "expected fuel exhaustion error, got: {err}"
        );
    }
}

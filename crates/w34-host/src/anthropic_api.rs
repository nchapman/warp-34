//! Anthropic Messages API (`/v1/messages`, `/v1/messages/count_tokens`) request
//! translation, Claude Code `cch=` billing-header normalization, sync message
//! builder, typed SSE state machine, and Anthropic error envelopes.

use serde_json::{Map, Value, json};

use crate::parser::{ParsedAssistantTurn, ParsedToolCall, StreamParseDelta};

const THINKING_SIGNATURE: &str = "w34-thinking-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnthropicErrorType {
    InvalidRequest,
    Authentication,
    NotFound,
    Overloaded,
    Api,
}

impl AnthropicErrorType {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request_error",
            Self::Authentication => "authentication_error",
            Self::NotFound => "not_found_error",
            Self::Overloaded => "overloaded_error",
            Self::Api => "api_error",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnthropicApiError {
    pub error_type: AnthropicErrorType,
    pub message: String,
}

impl AnthropicApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            error_type: AnthropicErrorType::InvalidRequest,
            message: msg.into(),
        }
    }

    pub fn overloaded(msg: impl Into<String>) -> Self {
        Self {
            error_type: AnthropicErrorType::Overloaded,
            message: msg.into(),
        }
    }

    pub fn api_error(msg: impl Into<String>) -> Self {
        Self {
            error_type: AnthropicErrorType::Api,
            message: msg.into(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "type": "error",
            "error": {
                "type": self.error_type.as_str(),
                "message": self.message,
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct AnthropicPreparedMeta {
    pub message_id: String,
    pub model: String,
    pub stream: bool,
}

/// Strips per-request `cch=<hex>;` tokens from `x-anthropic-billing-header:` lines
/// (matching `vLLM` `entrypoints/anthropic/serving.py`) so Claude Code requests
/// hit `MambaRadixCache` instead of invalidating the system prefix every turn.
pub fn strip_anthropic_billing_cch(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (idx, line) in text.split('\n').enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        if line.trim_start().starts_with("x-anthropic-billing-header:") {
            let cleaned = strip_cch_token_in_line(line);
            out.push_str(&cleaned);
        } else {
            out.push_str(line);
        }
    }
    out
}

fn strip_cch_token_in_line(line: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"cch=") {
            let mut j = i + 4;
            while j < bytes.len() && bytes[j].is_ascii_hexdigit() {
                j += 1;
            }
            if j > i + 4 && j < bytes.len() && bytes[j] == b';' {
                j += 1;
                if j < bytes.len() && bytes[j] == b' ' && result.ends_with(' ') {
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        let ch = line[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }
    result.trim_end().to_owned()
}

fn extract_system_prompt(system: Option<&Value>) -> Result<Option<String>, AnthropicApiError> {
    let Some(sys) = system.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let raw = match sys {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut chunks = Vec::with_capacity(blocks.len());
            for b in blocks {
                let obj = b.as_object().ok_or_else(|| {
                    AnthropicApiError::bad_request("system content block must be an object")
                })?;
                let btype = obj.get("type").and_then(Value::as_str).unwrap_or("text");
                if btype != "text" {
                    return Err(AnthropicApiError::bad_request(format!(
                        "unsupported system block type '{btype}'"
                    )));
                }
                let text = obj.get("text").and_then(Value::as_str).ok_or_else(|| {
                    AnthropicApiError::bad_request("system text block missing string 'text'")
                })?;
                if !text.is_empty() {
                    chunks.push(text.to_owned());
                }
            }
            chunks.join("\n\n")
        }
        _ => {
            return Err(AnthropicApiError::bad_request(
                "'system' must be a string or array of text blocks",
            ));
        }
    };
    let normalized = strip_anthropic_billing_cch(&raw);
    if normalized.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(normalized))
    }
}

fn convert_image_block(block_obj: &Map<String, Value>) -> Result<Value, AnthropicApiError> {
    let source = block_obj
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| AnthropicApiError::bad_request("image block missing 'source' object"))?;
    let stype = source
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| AnthropicApiError::bad_request("image.source missing string 'type'"))?;
    let url = match stype {
        "base64" => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    AnthropicApiError::bad_request("base64 image.source missing 'media_type'")
                })?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    AnthropicApiError::bad_request("base64 image.source missing 'data'")
                })?;
            format!("data:{media_type};base64,{data}")
        }
        "url" => source
            .get("url")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| AnthropicApiError::bad_request("url image.source missing 'url'"))?
            .to_owned(),
        other => {
            return Err(AnthropicApiError::bad_request(format!(
                "unsupported image.source.type '{other}'"
            )));
        }
    };
    Ok(json!({
        "type": "image_url",
        "image_url": {"url": url},
    }))
}

fn convert_messages(
    messages_val: &Value,
    system: Option<String>,
) -> Result<Vec<Value>, AnthropicApiError> {
    let arr = messages_val
        .as_array()
        .ok_or_else(|| AnthropicApiError::bad_request("'messages' must be an array"))?;
    if arr.is_empty() {
        return Err(AnthropicApiError::bad_request(
            "'messages' must not be empty",
        ));
    }

    let mut out = Vec::with_capacity(arr.len() + usize::from(system.is_some()));
    if let Some(sys) = system {
        out.push(json!({"role": "system", "content": sys}));
    }

    for msg in arr {
        let obj = msg
            .as_object()
            .ok_or_else(|| AnthropicApiError::bad_request("each message must be an object"))?;
        let role = obj
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| AnthropicApiError::bad_request("message missing string 'role'"))?;
        let content_val = obj
            .get("content")
            .ok_or_else(|| AnthropicApiError::bad_request("message missing 'content'"))?;

        match role {
            "user" => match content_val {
                Value::String(s) => {
                    out.push(json!({"role": "user", "content": s}));
                }
                Value::Array(blocks) => {
                    if blocks.is_empty() {
                        return Err(AnthropicApiError::bad_request(
                            "user message content array must not be empty",
                        ));
                    }
                    let mut tool_messages = Vec::new();
                    let mut user_parts = Vec::new();
                    let mut has_image = false;
                    let mut user_text = String::new();

                    for b in blocks {
                        let bobj = b.as_object().ok_or_else(|| {
                            AnthropicApiError::bad_request("content block must be an object")
                        })?;
                        let btype = bobj.get("type").and_then(Value::as_str).ok_or_else(|| {
                            AnthropicApiError::bad_request("content block missing string 'type'")
                        })?;
                        match btype {
                            "text" => {
                                let t =
                                    bobj.get("text").and_then(Value::as_str).ok_or_else(|| {
                                        AnthropicApiError::bad_request(
                                            "text block missing string 'text'",
                                        )
                                    })?;
                                user_text.push_str(t);
                                user_parts.push(json!({"type": "text", "text": t}));
                            }
                            "image" => {
                                has_image = true;
                                user_parts.push(convert_image_block(bobj)?);
                            }
                            "tool_result" => {
                                let tool_use_id = bobj
                                    .get("tool_use_id")
                                    .and_then(Value::as_str)
                                    .filter(|s| !s.is_empty())
                                    .ok_or_else(|| {
                                        AnthropicApiError::bad_request(
                                            "tool_result block missing non-empty 'tool_use_id'",
                                        )
                                    })?;
                                let is_error = bobj
                                    .get("is_error")
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false);
                                let mut tr_text = String::new();
                                match bobj.get("content") {
                                    None | Some(Value::Null) => {}
                                    Some(Value::String(s)) => tr_text.push_str(s),
                                    Some(Value::Array(inner)) => {
                                        for ib in inner {
                                            let iobj = ib.as_object().ok_or_else(|| {
                                                AnthropicApiError::bad_request(
                                                    "tool_result content block must be an object",
                                                )
                                            })?;
                                            match iobj.get("type").and_then(Value::as_str) {
                                                Some("text") => {
                                                    if let Some(t) =
                                                        iobj.get("text").and_then(Value::as_str)
                                                    {
                                                        if !tr_text.is_empty() {
                                                            tr_text.push('\n');
                                                        }
                                                        tr_text.push_str(t);
                                                    }
                                                }
                                                Some("image") => {
                                                    has_image = true;
                                                    user_parts.push(convert_image_block(iobj)?);
                                                }
                                                Some(other) => {
                                                    return Err(AnthropicApiError::bad_request(
                                                        format!(
                                                            "unsupported tool_result inner block type '{other}'"
                                                        ),
                                                    ));
                                                }
                                                None => {
                                                    return Err(AnthropicApiError::bad_request(
                                                        "tool_result inner block missing 'type'",
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                    Some(_) => {
                                        return Err(AnthropicApiError::bad_request(
                                            "tool_result 'content' must be a string or array",
                                        ));
                                    }
                                }
                                if is_error && !tr_text.starts_with("Error:") {
                                    tr_text = format!("Error: {tr_text}");
                                }
                                tool_messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": tr_text,
                                }));
                            }
                            other => {
                                return Err(AnthropicApiError::bad_request(format!(
                                    "unsupported user content block type '{other}'"
                                )));
                            }
                        }
                    }

                    out.extend(tool_messages);
                    if has_image {
                        out.push(json!({"role": "user", "content": user_parts}));
                    } else if !user_parts.is_empty() {
                        out.push(json!({"role": "user", "content": user_text}));
                    }
                }
                _ => {
                    return Err(AnthropicApiError::bad_request(
                        "user 'content' must be a string or array of blocks",
                    ));
                }
            },
            "assistant" => match content_val {
                Value::String(s) => {
                    out.push(json!({"role": "assistant", "content": s}));
                }
                Value::Array(blocks) => {
                    let mut thinking_text = String::new();
                    let mut assistant_text = String::new();
                    let mut tool_calls = Vec::new();

                    for b in blocks {
                        let bobj = b.as_object().ok_or_else(|| {
                            AnthropicApiError::bad_request("assistant block must be an object")
                        })?;
                        let btype = bobj.get("type").and_then(Value::as_str).ok_or_else(|| {
                            AnthropicApiError::bad_request("assistant block missing string 'type'")
                        })?;
                        match btype {
                            "thinking" => {
                                if let Some(t) = bobj.get("thinking").and_then(Value::as_str) {
                                    thinking_text.push_str(t);
                                }
                            }
                            "redacted_thinking" => {}
                            "text" => {
                                let t =
                                    bobj.get("text").and_then(Value::as_str).ok_or_else(|| {
                                        AnthropicApiError::bad_request(
                                            "assistant text block missing string 'text'",
                                        )
                                    })?;
                                assistant_text.push_str(t);
                            }
                            "tool_use" => {
                                let id = bobj
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .filter(|s| !s.is_empty())
                                    .ok_or_else(|| {
                                        AnthropicApiError::bad_request(
                                            "tool_use block missing non-empty 'id'",
                                        )
                                    })?;
                                let name = bobj
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .filter(|s| !s.is_empty())
                                    .ok_or_else(|| {
                                        AnthropicApiError::bad_request(
                                            "tool_use block missing non-empty 'name'",
                                        )
                                    })?;
                                let input = bobj.get("input").cloned().unwrap_or_else(|| json!({}));
                                if !input.is_object() {
                                    return Err(AnthropicApiError::bad_request(
                                        "tool_use 'input' must be a JSON object",
                                    ));
                                }
                                tool_calls.push(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": input,
                                    }
                                }));
                            }
                            other => {
                                return Err(AnthropicApiError::bad_request(format!(
                                    "unsupported assistant content block type '{other}'"
                                )));
                            }
                        }
                    }

                    let mut amsg = json!({
                        "role": "assistant",
                        "content": if assistant_text.is_empty() && !tool_calls.is_empty() {
                            Value::Null
                        } else {
                            Value::String(assistant_text)
                        },
                    });
                    if !thinking_text.is_empty() {
                        amsg["reasoning_content"] = json!(thinking_text);
                    }
                    if !tool_calls.is_empty() {
                        amsg["tool_calls"] = Value::Array(tool_calls);
                    }
                    out.push(amsg);
                }
                _ => {
                    return Err(AnthropicApiError::bad_request(
                        "assistant 'content' must be a string or array of blocks",
                    ));
                }
            },
            other => {
                return Err(AnthropicApiError::bad_request(format!(
                    "unsupported Anthropic message role '{other}': expected 'user' or 'assistant'"
                )));
            }
        }
    }
    Ok(out)
}

/// Translates an Anthropic Messages API request body into `(canonical_chat_request, meta)`.
pub fn translate_anthropic_to_chat(
    req: &Value,
    default_model: &str,
    request_seq_id: u64,
    require_max_tokens: bool,
) -> Result<(Value, AnthropicPreparedMeta), AnthropicApiError> {
    let obj = req
        .as_object()
        .ok_or_else(|| AnthropicApiError::bad_request("request body must be a JSON object"))?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(default_model)
        .to_owned();

    let max_tokens = match obj.get("max_tokens").filter(|v| !v.is_null()) {
        None if !require_max_tokens => 256usize,
        None => {
            return Err(AnthropicApiError::bad_request(
                "missing required field 'max_tokens'",
            ));
        }
        Some(v) => {
            let n = v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| AnthropicApiError::bad_request("invalid 'max_tokens'"))?;
            if n == 0 || n > 4096 {
                return Err(AnthropicApiError::bad_request(
                    "'max_tokens' must be in 1..=4096",
                ));
            }
            n
        }
    };

    let system = extract_system_prompt(obj.get("system"))?;
    let messages_val = obj
        .get("messages")
        .ok_or_else(|| AnthropicApiError::bad_request("missing required field 'messages'"))?;
    let messages = convert_messages(messages_val, system)?;

    // Tools
    let mut chat_tools = Vec::new();
    if let Some(tools_val) = obj.get("tools").filter(|v| !v.is_null()) {
        let arr = tools_val
            .as_array()
            .ok_or_else(|| AnthropicApiError::bad_request("'tools' must be an array"))?;
        for t in arr {
            let tobj = t
                .as_object()
                .ok_or_else(|| AnthropicApiError::bad_request("each tool must be an object"))?;
            if let Some(ttype) = tobj.get("type").and_then(Value::as_str)
                && ttype != "custom"
                && !tobj.contains_key("input_schema")
            {
                return Err(AnthropicApiError::bad_request(format!(
                    "Anthropic server tool type '{ttype}' without 'input_schema' is not supported"
                )));
            }
            let name = tobj
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| AnthropicApiError::bad_request("tool missing non-empty 'name'"))?;
            let schema = tobj
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            let mut fn_map = Map::new();
            fn_map.insert("name".into(), json!(name));
            if let Some(desc) = tobj.get("description").filter(|v| !v.is_null()) {
                fn_map.insert("description".into(), desc.clone());
            }
            fn_map.insert("parameters".into(), schema);
            chat_tools.push(json!({
                "type": "function",
                "function": Value::Object(fn_map),
            }));
        }
    }

    // Tool choice
    let mut is_forced_tool_choice = false;
    let chat_tool_choice = if let Some(tc_val) = obj.get("tool_choice").filter(|v| !v.is_null()) {
        let tc_obj = tc_val
            .as_object()
            .ok_or_else(|| AnthropicApiError::bad_request("'tool_choice' must be an object"))?;
        let tc_type = tc_obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| AnthropicApiError::bad_request("tool_choice missing string 'type'"))?;
        match tc_type {
            "auto" => Some(json!("auto")),
            "none" => Some(json!("none")),
            "any" => {
                is_forced_tool_choice = true;
                Some(json!("required"))
            }
            "tool" => {
                is_forced_tool_choice = true;
                let name = tc_obj
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        AnthropicApiError::bad_request(
                            "tool_choice type 'tool' requires non-empty 'name'",
                        )
                    })?;
                Some(json!({"type": "function", "function": {"name": name}}))
            }
            other => {
                return Err(AnthropicApiError::bad_request(format!(
                    "unsupported tool_choice.type '{other}'"
                )));
            }
        }
    } else {
        None
    };

    // Thinking config
    let mut enable_thinking: Option<bool> = None;
    if let Some(th_val) = obj.get("thinking").filter(|v| !v.is_null()) {
        let th_obj = th_val
            .as_object()
            .ok_or_else(|| AnthropicApiError::bad_request("'thinking' must be an object"))?;
        let th_type = th_obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| AnthropicApiError::bad_request("thinking missing string 'type'"))?;
        match th_type {
            "enabled" => {
                if is_forced_tool_choice {
                    return Err(AnthropicApiError::bad_request(
                        "thinking cannot be enabled when tool_choice forces tool use ('any' or 'tool')",
                    ));
                }
                let budget = th_obj
                    .get("budget_tokens")
                    .and_then(Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        AnthropicApiError::bad_request(
                            "thinking.type 'enabled' requires integer 'budget_tokens'",
                        )
                    })?;
                if budget < 1024 {
                    return Err(AnthropicApiError::bad_request(
                        "'thinking.budget_tokens' must be >= 1024",
                    ));
                }
                if require_max_tokens && budget >= max_tokens {
                    return Err(AnthropicApiError::bad_request(
                        "'thinking.budget_tokens' must be strictly less than 'max_tokens'",
                    ));
                }
                enable_thinking = Some(true);
            }
            "disabled" => {
                enable_thinking = Some(false);
            }
            "adaptive" => {
                enable_thinking = Some(!is_forced_tool_choice);
            }
            other => {
                return Err(AnthropicApiError::bad_request(format!(
                    "unsupported thinking.type '{other}'"
                )));
            }
        }
    } else if is_forced_tool_choice {
        enable_thinking = Some(false);
    }

    // output_config.effort & output_config.format -> response_format
    let mut response_format: Option<Value> = None;
    if let Some(oc_val) = obj.get("output_config").filter(|v| !v.is_null()) {
        let oc_obj = oc_val
            .as_object()
            .ok_or_else(|| AnthropicApiError::bad_request("'output_config' must be an object"))?;
        if let Some(effort_val) = oc_obj.get("effort").filter(|v| !v.is_null()) {
            let eff = effort_val.as_str().ok_or_else(|| {
                AnthropicApiError::bad_request("'output_config.effort' must be a string")
            })?;
            match eff {
                "low" | "medium" | "high" | "xhigh" | "max" => {
                    if enable_thinking.is_none() && !is_forced_tool_choice {
                        enable_thinking = Some(true);
                    }
                }
                _ => {
                    return Err(AnthropicApiError::bad_request(format!(
                        "unsupported output_config.effort '{eff}'"
                    )));
                }
            }
        }
        if let Some(fmt_val) = oc_obj.get("format").filter(|v| !v.is_null()) {
            let fmt_obj = fmt_val.as_object().ok_or_else(|| {
                AnthropicApiError::bad_request("'output_config.format' must be an object")
            })?;
            match fmt_obj.get("type").and_then(Value::as_str) {
                Some("json_schema") => {
                    let schema = fmt_obj.get("schema").ok_or_else(|| {
                        AnthropicApiError::bad_request("output_config.format missing 'schema'")
                    })?;
                    response_format = Some(json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": "anthropic_output_schema",
                            "strict": true,
                            "schema": schema,
                        }
                    }));
                }
                Some("text") | None => {}
                Some(other) => {
                    return Err(AnthropicApiError::bad_request(format!(
                        "unsupported output_config.format.type '{other}'"
                    )));
                }
            }
        }
    }

    let stream = obj
        .get("stream")
        .filter(|v| !v.is_null())
        .map_or(Ok(false), |v| {
            v.as_bool()
                .ok_or_else(|| AnthropicApiError::bad_request("'stream' must be a boolean"))
        })?;

    let mut chat_req = Map::new();
    chat_req.insert("model".into(), json!(model));
    chat_req.insert("messages".into(), Value::Array(messages));
    chat_req.insert("max_tokens".into(), json!(max_tokens));
    chat_req.insert("stream".into(), json!(stream));
    if let Some(temp) = obj.get("temperature").filter(|v| !v.is_null()) {
        chat_req.insert("temperature".into(), temp.clone());
    }
    if let Some(top_p) = obj.get("top_p").filter(|v| !v.is_null()) {
        chat_req.insert("top_p".into(), top_p.clone());
    }
    if let Some(top_k) = obj.get("top_k").filter(|v| !v.is_null()) {
        chat_req.insert("top_k".into(), top_k.clone());
    }
    if let Some(stops) = obj.get("stop_sequences").filter(|v| !v.is_null()) {
        chat_req.insert("stop".into(), stops.clone());
    }
    for key in [
        "lora",
        "adapter",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "stop_token_ids",
        "min_tokens",
        "ignore_eos",
    ] {
        if let Some(val) = obj.get(key).filter(|v| !v.is_null()) {
            chat_req.insert(key.into(), val.clone());
        }
    }
    if !chat_tools.is_empty() {
        chat_req.insert("tools".into(), Value::Array(chat_tools));
    }
    if let Some(tc) = chat_tool_choice {
        chat_req.insert("tool_choice".into(), tc);
    }
    if let Some(rf) = response_format {
        chat_req.insert("response_format".into(), rf);
    }
    let mut template_kwargs = match obj.get("chat_template_kwargs").filter(|v| !v.is_null()) {
        None => Map::new(),
        Some(v) => v.as_object().cloned().ok_or_else(|| {
            AnthropicApiError::bad_request("'chat_template_kwargs' must be an object")
        })?,
    };
    if let Some(thinking) = enable_thinking {
        template_kwargs.insert("enable_thinking".into(), json!(thinking));
    }
    if !template_kwargs.is_empty() {
        chat_req.insert(
            "chat_template_kwargs".into(),
            Value::Object(template_kwargs),
        );
    }

    let meta = AnthropicPreparedMeta {
        message_id: crate::responses_api::generate_unguessable_response_id(request_seq_id)
            .replacen("resp_", "msg_", 1),
        model,
        stream,
    };
    Ok((Value::Object(chat_req), meta))
}

fn tool_id_for(message_id: &str, tc: &ParsedToolCall, idx: usize) -> String {
    if let Some(rest) = tc.id.strip_prefix("toolu_") {
        return format!("toolu_{rest}");
    }
    let suffix = message_id.strip_prefix("msg_").unwrap_or(message_id);
    format!("toolu_{suffix}_{idx}")
}

fn parse_tool_input_object(arguments: &str) -> Value {
    match serde_json::from_str::<Value>(arguments) {
        Ok(v) if v.is_object() => v,
        _ => json!({}),
    }
}

pub fn anthropic_usage_json(
    prompt_tokens: usize,
    cached_tokens: usize,
    output_tokens: usize,
) -> Value {
    let cache_read = cached_tokens.min(prompt_tokens);
    let uncached_input = prompt_tokens.saturating_sub(cache_read);
    json!({
        "input_tokens": uncached_input,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output_tokens,
    })
}

pub fn resolve_stop_reason(
    turn: &ParsedAssistantTurn,
    finish_reason: &str,
    matched_stop: Option<&str>,
) -> (&'static str, Value) {
    if !turn.tool_calls.is_empty() {
        ("tool_use", Value::Null)
    } else if let Some(seq) = matched_stop {
        ("stop_sequence", json!(seq))
    } else if finish_reason == "length" {
        ("max_tokens", Value::Null)
    } else {
        ("end_turn", Value::Null)
    }
}

/// Builds the non-streaming Anthropic `type: "message"` JSON response.
pub fn build_anthropic_message(
    meta: &AnthropicPreparedMeta,
    turn: &ParsedAssistantTurn,
    finish_reason: &str,
    matched_stop: Option<&str>,
    prompt_tokens: usize,
    cached_tokens: usize,
    output_tokens: usize,
) -> Value {
    let mut content = Vec::new();
    if let Some(thinking) = turn.reasoning_content.as_deref().filter(|s| !s.is_empty()) {
        content.push(json!({
            "type": "thinking",
            "thinking": thinking,
            "signature": THINKING_SIGNATURE,
        }));
    }
    let text = turn.content.as_deref().unwrap_or("");
    if !text.is_empty() || (content.is_empty() && turn.tool_calls.is_empty()) {
        content.push(json!({
            "type": "text",
            "text": text,
        }));
    }
    for (idx, tc) in turn.tool_calls.iter().enumerate() {
        content.push(json!({
            "type": "tool_use",
            "id": tool_id_for(&meta.message_id, tc, idx),
            "name": tc.function.name,
            "input": parse_tool_input_object(&tc.function.arguments),
        }));
    }
    let (stop_reason, stop_sequence) = resolve_stop_reason(turn, finish_reason, matched_stop);
    json!({
        "id": meta.message_id,
        "type": "message",
        "role": "assistant",
        "model": meta.model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": anthropic_usage_json(prompt_tokens, cached_tokens, output_tokens),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveBlock {
    None,
    Thinking { index: usize },
    Text { index: usize },
}

/// Deterministic SSE state machine for `POST /v1/messages` (`stream: true`).
pub struct AnthropicSseFormatter {
    meta: AnthropicPreparedMeta,
    prompt_tokens: usize,
    started: bool,
    next_block_index: usize,
    next_tool_index: usize,
    active: ActiveBlock,
}

impl AnthropicSseFormatter {
    pub fn new(meta: AnthropicPreparedMeta, prompt_tokens: usize) -> Self {
        Self {
            meta,
            prompt_tokens,
            started: false,
            next_block_index: 0,
            next_tool_index: 0,
            active: ActiveBlock::None,
        }
    }

    pub fn ensure_started(&mut self, cached_tokens: usize) -> Vec<(String, Value)> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![(
            "message_start".into(),
            json!({
                "type": "message_start",
                "message": {
                    "id": self.meta.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.meta.model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": anthropic_usage_json(self.prompt_tokens, cached_tokens, 0),
                }
            }),
        )]
    }

    fn close_active(&mut self, out: &mut Vec<(String, Value)>) {
        match self.active {
            ActiveBlock::None => {}
            ActiveBlock::Thinking { index } => {
                out.push((
                    "content_block_delta".into(),
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": THINKING_SIGNATURE,
                        }
                    }),
                ));
                out.push((
                    "content_block_stop".into(),
                    json!({
                        "type": "content_block_stop",
                        "index": index,
                    }),
                ));
                self.active = ActiveBlock::None;
            }
            ActiveBlock::Text { index } => {
                out.push((
                    "content_block_stop".into(),
                    json!({
                        "type": "content_block_stop",
                        "index": index,
                    }),
                ));
                self.active = ActiveBlock::None;
            }
        }
    }

    fn emit_tool_block(&mut self, tc: &ParsedToolCall, out: &mut Vec<(String, Value)>) {
        self.close_active(out);
        let index = self.next_block_index;
        self.next_block_index += 1;
        let tool_idx = self.next_tool_index;
        self.next_tool_index += 1;
        let id = tool_id_for(&self.meta.message_id, tc, tool_idx);
        let partial_json =
            if serde_json::from_str::<Value>(&tc.function.arguments).is_ok_and(|v| v.is_object()) {
                tc.function.arguments.clone()
            } else {
                "{}".to_owned()
            };

        out.push((
            "content_block_start".into(),
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": id,
                    "name": tc.function.name,
                    "input": {},
                }
            }),
        ));
        out.push((
            "content_block_delta".into(),
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                }
            }),
        ));
        out.push((
            "content_block_stop".into(),
            json!({
                "type": "content_block_stop",
                "index": index,
            }),
        ));
    }

    pub fn push_delta(
        &mut self,
        delta: StreamParseDelta,
        cached_tokens: usize,
    ) -> Vec<(String, Value)> {
        let mut out = self.ensure_started(cached_tokens);

        if let Some(r_chunk) = delta.reasoning_delta.filter(|s| !s.is_empty()) {
            let index = match self.active {
                ActiveBlock::Thinking { index } => index,
                _ => {
                    self.close_active(&mut out);
                    let idx = self.next_block_index;
                    self.next_block_index += 1;
                    out.push((
                        "content_block_start".into(),
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "thinking",
                                "thinking": "",
                            }
                        }),
                    ));
                    self.active = ActiveBlock::Thinking { index: idx };
                    idx
                }
            };
            out.push((
                "content_block_delta".into(),
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": r_chunk,
                    }
                }),
            ));
        }

        if let Some(c_chunk) = delta.content_delta.filter(|s| !s.is_empty()) {
            let index = match self.active {
                ActiveBlock::Text { index } => index,
                _ => {
                    self.close_active(&mut out);
                    let idx = self.next_block_index;
                    self.next_block_index += 1;
                    out.push((
                        "content_block_start".into(),
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "text",
                                "text": "",
                            }
                        }),
                    ));
                    self.active = ActiveBlock::Text { index: idx };
                    idx
                }
            };
            out.push((
                "content_block_delta".into(),
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "text_delta",
                        "text": c_chunk,
                    }
                }),
            ));
        }

        for tc in &delta.completed_tool_calls {
            self.emit_tool_block(tc, &mut out);
        }

        out
    }

    pub fn finish(
        &mut self,
        turn: &ParsedAssistantTurn,
        finish_reason: &str,
        matched_stop: Option<&str>,
        cached_tokens: usize,
        output_tokens: usize,
    ) -> Vec<(String, Value)> {
        let mut out = self.ensure_started(cached_tokens);
        self.close_active(&mut out);

        if self.next_tool_index < turn.tool_calls.len() {
            for tc in &turn.tool_calls[self.next_tool_index..] {
                self.emit_tool_block(tc, &mut out);
            }
        }

        if self.next_block_index == 0 {
            out.push((
                "content_block_start".into(),
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }),
            ));
            out.push((
                "content_block_stop".into(),
                json!({"type": "content_block_stop", "index": 0}),
            ));
            self.next_block_index = 1;
        }

        let (stop_reason, stop_sequence) = resolve_stop_reason(turn, finish_reason, matched_stop);
        out.push((
            "message_delta".into(),
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                },
                "usage": {
                    "output_tokens": output_tokens,
                }
            }),
        ));
        out.push(("message_stop".into(), json!({"type": "message_stop"})));
        out
    }

    pub fn error_event(message: &str) -> (String, Value) {
        (
            "error".into(),
            AnthropicApiError::api_error(message).to_json(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Qwen3Parser;
    use crate::request_constraints::RequestConstraint;

    #[test]
    fn strips_claude_code_billing_cch_without_touching_body() {
        let s1 = "x-anthropic-billing-header: cc_version=2.1.63.b5c; cc_entrypoint=cli; cch=7019f;\nYou are Claude Code.";
        let s2 = "x-anthropic-billing-header: cc_version=2.1.63.b5c; cc_entrypoint=cli; cch=deadBEEF01;\nYou are Claude Code.";
        assert_eq!(
            strip_anthropic_billing_cch(s1),
            strip_anthropic_billing_cch(s2)
        );
        assert_eq!(
            strip_anthropic_billing_cch(s1),
            "x-anthropic-billing-header: cc_version=2.1.63.b5c; cc_entrypoint=cli;\nYou are Claude Code."
        );
    }

    #[test]
    fn validates_thinking_budget_and_translates_tools_and_multi_turn_blocks() {
        // Invalid budget (< 1024)
        let bad_budget = json!({
            "model": "qwen",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 512},
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert!(translate_anthropic_to_chat(&bad_budget, "qwen", 1, true).is_err());

        // Invalid: budget_tokens >= max_tokens
        let budget_ge_max = json!({
            "model": "qwen",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert!(translate_anthropic_to_chat(&budget_ge_max, "qwen", 2, true).is_err());

        // Invalid: thinking enabled + forced tool_choice ("any")
        let forced_conflict = json!({
            "model": "qwen",
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "ping", "input_schema": {"type": "object", "properties": {}}}],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert!(translate_anthropic_to_chat(&forced_conflict, "qwen", 3, true).is_err());

        // Valid multi-turn with tool_use + tool_result + tool_choice="none"
        let valid = json!({
            "model": "qwen",
            "max_tokens": 2048,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=abc123;"},
                {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}}
            ],
            "tools": [{
                "name": "read_file",
                "description": "Reads a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }],
            "tool_choice": {"type": "none"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [
                {"role": "user", "content": "Read src/lib.rs"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Reading file", "signature": "sig"},
                    {"type": "tool_use", "id": "toolu_01", "name": "read_file", "input": {"path": "src/lib.rs"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "pub mod server;"}
                ]}
            ]
        });
        let (chat, _) = translate_anthropic_to_chat(&valid, "qwen", 4, true).unwrap();
        assert_eq!(chat["tool_choice"], "none");
        assert!(RequestConstraint::parse(&chat).unwrap().is_none());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "system");
        assert!(!msgs[0]["content"].as_str().unwrap().contains("cch="));
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["reasoning_content"], "Reading file");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["content"], "pub mod server;");
    }

    #[test]
    fn anthropic_sse_and_usage_cached_subtraction_match_spec() {
        let req = json!({
            "model": "qwen",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let (_, meta) = translate_anthropic_to_chat(&req, "qwen", 9, true).unwrap();
        let mut parser = Qwen3Parser::new(true);
        let mut sse = AnthropicSseFormatter::new(meta.clone(), 100);
        let mut events = Vec::new();
        for chunk in [
            "Thinking...</think>",
            "Calling tool.\n<tool_call>\n<function=ping>\n<parameter=host>\n127.0.0.1\n</parameter>\n</function>\n</tool_call>",
        ] {
            events.extend(sse.push_delta(parser.push_str(chunk), 64));
        }
        let (tail, turn) = parser.finish();
        events.extend(sse.push_delta(tail, 64));
        events.extend(sse.finish(&turn, "tool_calls", None, 64, 19));

        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.first().copied(), Some("message_start"));
        assert_eq!(names.last().copied(), Some("message_stop"));
        assert!(names.contains(&"message_delta"));

        // Verify Anthropic usage invariant: total(100) = input_tokens(36) + cache_read_input_tokens(64)
        assert_eq!(events[0].1["message"]["usage"]["input_tokens"], 36);
        assert_eq!(
            events[0].1["message"]["usage"]["cache_read_input_tokens"],
            64
        );

        let sync_msg = build_anthropic_message(&meta, &turn, "tool_calls", None, 100, 64, 19);
        assert_eq!(sync_msg["stop_reason"], "tool_use");
        assert_eq!(sync_msg["usage"]["input_tokens"], 36);
        assert_eq!(sync_msg["usage"]["cache_read_input_tokens"], 64);
        assert_eq!(sync_msg["usage"]["output_tokens"], 19);
        assert_eq!(sync_msg["content"].as_array().unwrap().len(), 3);
        assert_eq!(sync_msg["content"][0]["type"], "thinking");
        assert_eq!(sync_msg["content"][1]["type"], "text");
        assert_eq!(sync_msg["content"][2]["type"], "tool_use");
        assert_eq!(sync_msg["content"][2]["input"]["host"], "127.0.0.1");
    }

    #[test]
    fn anthropic_error_constructors_and_serialization() {
        for (err_type, expected_str) in [
            (AnthropicErrorType::InvalidRequest, "invalid_request_error"),
            (AnthropicErrorType::Authentication, "authentication_error"),
            (AnthropicErrorType::NotFound, "not_found_error"),
            (AnthropicErrorType::Overloaded, "overloaded_error"),
            (AnthropicErrorType::Api, "api_error"),
        ] {
            assert_eq!(err_type.as_str(), expected_str);
        }
        let e1 = AnthropicApiError::bad_request("invalid param");
        assert_eq!(e1.to_json()["error"]["type"], "invalid_request_error");
        assert_eq!(e1.to_json()["error"]["message"], "invalid param");

        let e2 = AnthropicApiError::overloaded("high load");
        assert_eq!(e2.to_json()["error"]["type"], "overloaded_error");

        let e3 = AnthropicApiError::api_error("internal crash");
        assert_eq!(e3.to_json()["error"]["type"], "api_error");

        let (ev_name, ev_val) = AnthropicSseFormatter::error_event("fatal");
        assert_eq!(ev_name, "error");
        assert_eq!(ev_val["error"]["message"], "fatal");
    }

    #[test]
    fn translates_anthropic_multimodal_and_structured_output_and_stops() {
        // Base64 and URL image translation
        let img_req = json!({
            "model": "qwen-vl",
            "max_tokens": 128,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUg=="
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/test.jpg"
                        }
                    }
                ]
            }],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"ans": {"type": "string"}},
                        "required": ["ans"]
                    }
                }
            }
        });
        let (chat, _) = translate_anthropic_to_chat(&img_req, "qwen-vl", 11, false).unwrap();
        assert_eq!(chat["response_format"]["type"], "json_schema");
        let parts = chat["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        );
        assert_eq!(parts[2]["type"], "image_url");
        assert_eq!(parts[2]["image_url"]["url"], "https://example.com/test.jpg");

        // output_config format must be an object with type json_schema; non-object format is rejected
        let invalid_fmt_req = json!({
            "model": "qwen",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": "json"}
        });
        assert!(translate_anthropic_to_chat(&invalid_fmt_req, "qwen", 12, false).is_err());

        // resolve_stop_reason mapping
        let empty_turn = ParsedAssistantTurn {
            reasoning_content: None,
            content: Some("hi".into()),
            tool_calls: Vec::new(),
        };
        assert_eq!(
            resolve_stop_reason(&empty_turn, "stop", None),
            ("end_turn", Value::Null)
        );
        assert_eq!(
            resolve_stop_reason(&empty_turn, "length", None),
            ("max_tokens", Value::Null)
        );
        assert_eq!(
            resolve_stop_reason(&empty_turn, "stop", Some("###")),
            ("stop_sequence", json!("###"))
        );
    }

    #[test]
    fn validates_all_anthropic_bad_request_paths() {
        let bad_cases = [
            json!([]),
            json!({"model": "q"}),
            json!({"messages": []}),
            json!({"messages": "hi"}),
            json!({"messages": [123]}),
            json!({"messages": [{"content": "hi"}]}),
            json!({"messages": [{"role": "user"}]}),
            json!({"messages": [{"role": "system", "content": "hi"}]}),
            json!({"messages": [{"role": "user", "content": []}]}),
            json!({"messages": [{"role": "user", "content": [42]}]}),
            json!({"messages": [{"role": "user", "content": [{}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "text"}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "audio"}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image"}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "ftp"}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png"}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "abc"}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url"}}]}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "tool_result"}]}]}),
            json!({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text"}]}]}),
            json!({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "video"}]}]}),
            json!({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "tool_use", "name": "foo"}]}]}),
            json!({"system": 123, "messages": [{"role": "user", "content": "hi"}]}),
            json!({"system": [123], "messages": [{"role": "user", "content": "hi"}]}),
            json!({"system": [{"type": "image"}], "messages": [{"role": "user", "content": "hi"}]}),
            json!({"system": [{"type": "text"}], "messages": [{"role": "user", "content": "hi"}]}),
            json!({"tools": "tools", "messages": [{"role": "user", "content": "hi"}]}),
            json!({"tools": [1], "messages": [{"role": "user", "content": "hi"}]}),
            json!({"tools": [{"input_schema": {}}], "messages": [{"role": "user", "content": "hi"}]}),
            json!({"tools": [{"type": "bash_20241022", "name": "bash"}], "messages": [{"role": "user", "content": "hi"}]}),
            json!({"tool_choice": {"type": "unknown"}, "messages": [{"role": "user", "content": "hi"}]}),
            json!({"tool_choice": {"type": "tool"}, "messages": [{"role": "user", "content": "hi"}]}),
            json!({"thinking": "enabled", "messages": [{"role": "user", "content": "hi"}]}),
            json!({"thinking": {"type": "magic"}, "messages": [{"role": "user", "content": "hi"}]}),
            json!({"thinking": {"type": "enabled"}, "messages": [{"role": "user", "content": "hi"}]}),
            json!({"output_config": {"effort": "invalid"}, "messages": [{"role": "user", "content": "hi"}]}),
            json!({"chat_template_kwargs": "bad", "messages": [{"role": "user", "content": "hi"}]}),
        ];
        for case in &bad_cases {
            assert!(
                translate_anthropic_to_chat(case, "q", 1, false).is_err(),
                "expected bad-request error for payload: {case}"
            );
        }
        let (chat, _) = translate_anthropic_to_chat(
            &json!({
                "model": "q",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"effort": "high"},
                "seed": 99,
                "presence_penalty": 0.25,
                "chat_template_kwargs": {"preserve_thinking": true}
            }),
            "q",
            1,
            true,
        )
        .unwrap();
        assert_eq!(chat["seed"], 99);
        assert_eq!(chat["presence_penalty"], 0.25);
        assert_eq!(chat["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(chat["chat_template_kwargs"]["preserve_thinking"], true);
    }
}

//! OpenAI Responses API (`/v1/responses`, `/v1/responses/input_tokens`) translation,
//! bounded in-memory `ResponseStore`, sync response builder, and 0-indexed
//! `sequence_number` SSE state machine.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use serde_json::{Map, Value, json};

use crate::parser::{ParsedAssistantTurn, ParsedToolCall, StreamParseDelta};

const DEFAULT_STORE_CAPACITY: usize = 256;
const DEFAULT_PER_SCOPE_CAPACITY: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResponsesErrorKind {
    InvalidRequest,
    NotFound,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponsesApiError {
    pub kind: ResponsesErrorKind,
    pub message: String,
}

impl ResponsesApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            kind: ResponsesErrorKind::InvalidRequest,
            message: msg.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            kind: ResponsesErrorKind::NotFound,
            message: msg.into(),
        }
    }

    pub fn to_openai_json(&self) -> Value {
        let err_type = match self.kind {
            ResponsesErrorKind::InvalidRequest => "invalid_request_error",
            ResponsesErrorKind::NotFound => "not_found_error",
        };
        json!({
            "error": {
                "message": self.message,
                "type": err_type,
                "param": Value::Null,
                "code": Value::Null,
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct StoredResponse {
    pub id: String,
    pub owner_scope: String,
    pub response: Value,
    pub history_messages: Vec<Value>,
}

/// Generates a 128-bit cryptographically random response identifier (`resp_<32 hex>`) from OS
/// CSPRNG (`/dev/urandom`), preventing response-ID enumeration across callers.
pub fn generate_unguessable_response_id(seq_fallback: u64) -> String {
    let mut buf = [0u8; 16];
    let ok = std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut buf))
        .is_ok();
    if !ok {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let rs1 = RandomState::new();
        let rs2 = RandomState::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0u128, |d| d.as_nanos());
        let mut h1 = rs1.build_hasher();
        h1.write_u128(now);
        h1.write_u64(seq_fallback);
        let mut h2 = rs2.build_hasher();
        h2.write_u64(seq_fallback.rotate_left(17));
        h2.write_u128(now.rotate_right(29));
        buf[..8].copy_from_slice(&h1.finish().to_le_bytes());
        buf[8..].copy_from_slice(&h2.finish().to_le_bytes());
    }
    let mut out = String::with_capacity(5 + 32);
    out.push_str("resp_");
    for b in buf {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// Bounded store for explicit opt-in (`"store": true`) responses and active cancellation handles.
///
/// Storing, retrieving, chaining (`previous_response_id`), deleting, or cancelling a response
/// requires a non-empty `Authorization` or `x-api-key` credential header; entries and eviction
/// quotas are partitioned per credential scope alongside 128-bit CSPRNG `resp_<32 hex>` IDs.
pub struct ResponseStore {
    capacity: usize,
    per_scope_capacity: usize,
    scope_hasher: std::collections::hash_map::RandomState,
    entries: Mutex<VecDeque<StoredResponse>>,
    active_cancels: Mutex<HashMap<String, (String, Arc<AtomicBool>)>>,
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_STORE_CAPACITY)
    }
}

impl ResponseStore {
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let per_scope = DEFAULT_PER_SCOPE_CAPACITY.min(cap).max(1);
        Self {
            capacity: cap,
            per_scope_capacity: per_scope,
            scope_hasher: std::collections::hash_map::RandomState::new(),
            entries: Mutex::new(VecDeque::with_capacity(cap)),
            active_cancels: Mutex::new(HashMap::new()),
        }
    }

    /// Derives an opaque per-credential scope token from an HTTP `Authorization` or `x-api-key`
    /// header value (`None` when unauthenticated). Strips an optional case-insensitive `Bearer`
    /// prefix and surrounding ASCII whitespace so `Authorization: Bearer <token>` and
    /// `x-api-key: <token>` map to the same scope.
    pub fn scope_for_credential(&self, raw_credential: Option<&[u8]>) -> Option<String> {
        use std::hash::{BuildHasher, Hasher};
        let mut trimmed = raw_credential?;
        while let Some((&b, rest)) = trimmed.split_first() {
            if b.is_ascii_whitespace() {
                trimmed = rest;
            } else {
                break;
            }
        }
        while let Some((&b, rest)) = trimmed.split_last() {
            if b.is_ascii_whitespace() {
                trimmed = rest;
            } else {
                break;
            }
        }
        if trimmed.len() > 6
            && trimmed[..6].eq_ignore_ascii_case(b"bearer")
            && trimmed[6].is_ascii_whitespace()
        {
            trimmed = &trimmed[6..];
            while let Some((&b, rest)) = trimmed.split_first() {
                if b.is_ascii_whitespace() {
                    trimmed = rest;
                } else {
                    break;
                }
            }
        }
        if trimmed.is_empty() {
            return None;
        }
        let mut h = self.scope_hasher.build_hasher();
        h.write(b"w34-responses-scope:");
        h.write(trimmed);
        Some(format!("cred_{:016x}", h.finish()))
    }

    pub fn insert(&self, entry: StoredResponse) {
        let mut guard = self.entries.lock().unwrap();
        if let Some(pos) = guard.iter().position(|e| e.id == entry.id) {
            guard.remove(pos);
        } else {
            let scope_count = guard
                .iter()
                .filter(|e| e.owner_scope == entry.owner_scope)
                .count();
            if scope_count >= self.per_scope_capacity {
                if let Some(pos) = guard
                    .iter()
                    .position(|e| e.owner_scope == entry.owner_scope)
                {
                    guard.remove(pos);
                }
            } else if guard.len() >= self.capacity {
                // Evict the oldest entry from whichever scope currently occupies the most slots,
                // preventing one noisy caller from displacing other scopes' stored conversations.
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for e in guard.iter() {
                    *counts.entry(e.owner_scope.as_str()).or_insert(0) += 1;
                }
                let largest_scope = counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(s, _)| s.to_owned());
                if let Some(target_scope) = largest_scope
                    && let Some(pos) = guard.iter().position(|e| e.owner_scope == target_scope)
                {
                    guard.remove(pos);
                } else {
                    guard.pop_front();
                }
            }
        }
        guard.push_back(entry);
    }

    pub fn get_response(&self, id: &str, caller_scope: &str) -> Option<Value> {
        let guard = self.entries.lock().unwrap();
        guard
            .iter()
            .find(|e| e.id == id && e.owner_scope == caller_scope)
            .map(|e| e.response.clone())
    }

    pub fn get_history(&self, id: &str, caller_scope: &str) -> Option<Vec<Value>> {
        let guard = self.entries.lock().unwrap();
        guard
            .iter()
            .find(|e| e.id == id && e.owner_scope == caller_scope)
            .map(|e| e.history_messages.clone())
    }

    pub fn delete(&self, id: &str, caller_scope: &str) -> bool {
        let mut guard = self.entries.lock().unwrap();
        if let Some(pos) = guard
            .iter()
            .position(|e| e.id == id && e.owner_scope == caller_scope)
        {
            guard.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn register_active(&self, id: String, owner_scope: String, flag: Arc<AtomicBool>) {
        self.active_cancels
            .lock()
            .unwrap()
            .insert(id, (owner_scope, flag));
    }

    pub fn unregister_active(&self, id: &str) {
        self.active_cancels.lock().unwrap().remove(id);
    }

    pub fn cancel(&self, id: &str, caller_scope: &str) -> Option<Value> {
        let matched = self
            .active_cancels
            .lock()
            .unwrap()
            .get(id)
            .filter(|(owner, _)| owner == caller_scope)
            .map(|(_, flag)| flag.clone());
        if let Some(flag) = matched {
            flag.store(true, Ordering::Release);
            let now = unix_timestamp_secs();
            let cancelled = json!({
                "id": id,
                "object": "response",
                "created_at": now,
                "status": "cancelled",
                "error": Value::Null,
                "incomplete_details": {"reason": "cancelled"},
                "output": [],
                "output_text": "",
            });
            return Some(cancelled);
        }
        let mut guard = self.entries.lock().unwrap();
        if let Some(entry) = guard
            .iter_mut()
            .find(|e| e.id == id && e.owner_scope == caller_scope)
        {
            if entry.response["status"] == "in_progress" {
                entry.response["status"] = json!("cancelled");
            }
            return Some(entry.response.clone());
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct ResponsesPreparedMeta {
    pub response_id: String,
    pub created_at: u64,
    pub model: String,
    pub instructions: Value,
    pub previous_response_id: Value,
    pub max_output_tokens: usize,
    pub parallel_tool_calls: bool,
    pub store: bool,
    pub stream: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub text_echo: Value,
    pub tool_choice_echo: Value,
    pub tools_echo: Value,
    pub reasoning_echo: Value,
    pub truncation: String,
    pub metadata: Value,
    pub input_messages: Vec<Value>,
}

fn unix_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn normalize_tool_def(tool: &Value) -> Result<(Value, Value), ResponsesApiError> {
    let obj = tool
        .as_object()
        .ok_or_else(|| ResponsesApiError::bad_request("each tool must be a JSON object"))?;
    let tool_type = obj
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("function");
    if tool_type != "function" {
        return Err(ResponsesApiError::bad_request(format!(
            "unsupported Responses tool type '{tool_type}': only 'function' tools are supported"
        )));
    }
    let (name, description, parameters, strict) =
        if let Some(func) = obj.get("function").and_then(Value::as_object) {
            (
                func.get("name").and_then(Value::as_str),
                func.get("description").cloned(),
                func.get("parameters").cloned(),
                func.get("strict").cloned(),
            )
        } else {
            (
                obj.get("name").and_then(Value::as_str),
                obj.get("description").cloned(),
                obj.get("parameters").cloned(),
                obj.get("strict").cloned(),
            )
        };
    let name = name
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| ResponsesApiError::bad_request("function tool missing non-empty 'name'"))?;
    let params = parameters.unwrap_or_else(
        || json!({"type": "object", "properties": {}, "additionalProperties": false}),
    );

    let mut fn_obj = Map::new();
    fn_obj.insert("name".into(), json!(name));
    if let Some(desc) = description.clone().filter(|v| !v.is_null()) {
        fn_obj.insert("description".into(), desc);
    }
    fn_obj.insert("parameters".into(), params.clone());
    if let Some(st) = strict.clone().filter(|v| !v.is_null()) {
        fn_obj.insert("strict".into(), st);
    }
    let chat_tool = json!({
        "type": "function",
        "function": Value::Object(fn_obj),
    });

    let mut resp_tool = Map::new();
    resp_tool.insert("type".into(), json!("function"));
    resp_tool.insert("name".into(), json!(name));
    resp_tool.insert("description".into(), description.unwrap_or(Value::Null));
    resp_tool.insert("parameters".into(), params);
    resp_tool.insert("strict".into(), strict.unwrap_or(json!(true)));
    Ok((chat_tool, Value::Object(resp_tool)))
}

fn normalize_tool_choice(
    choice: Option<&Value>,
) -> Result<(Option<Value>, Value), ResponsesApiError> {
    let Some(v) = choice.filter(|v| !v.is_null()) else {
        return Ok((None, json!("auto")));
    };
    if let Some(s) = v.as_str() {
        return match s {
            "none" | "auto" | "required" => Ok((Some(json!(s)), json!(s))),
            other => Err(ResponsesApiError::bad_request(format!(
                "unsupported tool_choice '{other}'"
            ))),
        };
    }
    if let Some(obj) = v.as_object() {
        let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");
        if kind == "function" {
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| v.pointer("/function/name").and_then(Value::as_str))
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ResponsesApiError::bad_request("function tool_choice requires non-empty 'name'")
                })?;
            return Ok((
                Some(json!({"type": "function", "function": {"name": name}})),
                json!({"type": "function", "name": name}),
            ));
        }
    }
    Err(ResponsesApiError::bad_request(
        "tool_choice must be 'none', 'auto', 'required', or {\"type\": \"function\", \"name\": \"...\"}",
    ))
}

fn convert_content_parts(content: &Value, role: &str) -> Result<Value, ResponsesApiError> {
    match content {
        Value::String(s) => Ok(Value::String(s.clone())),
        Value::Array(parts) => {
            let mut chat_parts = Vec::with_capacity(parts.len());
            let mut has_image = false;
            let mut text_only = String::new();
            for part in parts {
                let Some(pobj) = part.as_object() else {
                    return Err(ResponsesApiError::bad_request(
                        "message content part must be an object",
                    ));
                };
                let ptype = pobj
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("input_text");
                match ptype {
                    "input_text" | "output_text" | "text" => {
                        let t = pobj.get("text").and_then(Value::as_str).ok_or_else(|| {
                            ResponsesApiError::bad_request(format!(
                                "content part '{ptype}' missing string 'text'"
                            ))
                        })?;
                        text_only.push_str(t);
                        chat_parts.push(json!({"type": "text", "text": t}));
                    }
                    "refusal" => {
                        let t = pobj
                            .get("refusal")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        text_only.push_str(t);
                        chat_parts.push(json!({"type": "text", "text": t}));
                    }
                    "input_image" | "image_url" => {
                        if matches!(role, "system" | "developer") {
                            return Err(ResponsesApiError::bad_request(
                                "system/developer messages cannot contain images",
                            ));
                        }
                        let url = pobj
                            .get("image_url")
                            .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)))
                            .ok_or_else(|| {
                                ResponsesApiError::bad_request(
                                    "input_image part requires string 'image_url' (data URL or HTTP(S) URL)",
                                )
                            })?;
                        has_image = true;
                        chat_parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    }
                    other => {
                        return Err(ResponsesApiError::bad_request(format!(
                            "unsupported Responses content part type '{other}'"
                        )));
                    }
                }
            }
            if has_image {
                Ok(Value::Array(chat_parts))
            } else {
                Ok(Value::String(text_only))
            }
        }
        _ => Err(ResponsesApiError::bad_request(
            "message content must be a string or array of content parts",
        )),
    }
}

fn parse_input_items(
    input: &Value,
    system_chunks: &mut Vec<String>,
    turns: &mut Vec<Value>,
) -> Result<(), ResponsesApiError> {
    match input {
        Value::String(s) => {
            turns.push(json!({"role": "user", "content": s}));
            Ok(())
        }
        Value::Array(items) => {
            if items.is_empty() {
                return Err(ResponsesApiError::bad_request(
                    "'input' array must not be empty",
                ));
            }
            let mut pending_reasoning: Option<String> = None;
            for item in items {
                let obj = item.as_object().ok_or_else(|| {
                    ResponsesApiError::bad_request("each item in 'input' must be an object")
                })?;
                let item_type = obj.get("type").and_then(Value::as_str).unwrap_or("message");
                match item_type {
                    "message" => {
                        let raw_role =
                            obj.get("role").and_then(Value::as_str).ok_or_else(|| {
                                ResponsesApiError::bad_request("message item missing string 'role'")
                            })?;
                        let content_val = obj.get("content").ok_or_else(|| {
                            ResponsesApiError::bad_request("message item missing 'content'")
                        })?;
                        let content = convert_content_parts(content_val, raw_role)?;
                        match raw_role {
                            "system" | "developer" => {
                                if let Some(s) = content.as_str()
                                    && !s.trim().is_empty()
                                {
                                    system_chunks.push(s.to_owned());
                                }
                            }
                            "user" => {
                                pending_reasoning = None;
                                turns.push(json!({"role": "user", "content": content}));
                            }
                            "assistant" => {
                                let mut msg = json!({"role": "assistant", "content": content});
                                if let Some(r) = pending_reasoning.take() {
                                    msg["reasoning_content"] = json!(r);
                                }
                                turns.push(msg);
                            }
                            "tool" => {
                                pending_reasoning = None;
                                turns.push(json!({"role": "tool", "content": content}));
                            }
                            other => {
                                return Err(ResponsesApiError::bad_request(format!(
                                    "unsupported message role '{other}'"
                                )));
                            }
                        }
                    }
                    "reasoning" => {
                        let mut text = String::new();
                        if let Some(arr) = obj.get("content").and_then(Value::as_array) {
                            for p in arr {
                                if let Some(t) = p.get("text").and_then(Value::as_str) {
                                    text.push_str(t);
                                }
                            }
                        } else if let Some(arr) = obj.get("summary").and_then(Value::as_array) {
                            for p in arr {
                                if let Some(t) = p.get("text").and_then(Value::as_str) {
                                    text.push_str(t);
                                }
                            }
                        }
                        if !text.is_empty() {
                            if let Some(last) = turns.last_mut().filter(|m| {
                                m.get("role").and_then(Value::as_str) == Some("assistant")
                            }) {
                                last["reasoning_content"] = json!(text);
                            } else {
                                pending_reasoning = Some(text);
                            }
                        }
                    }
                    "function_call" => {
                        let name = obj
                            .get("name")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                ResponsesApiError::bad_request(
                                    "function_call item missing non-empty 'name'",
                                )
                            })?;
                        let args_val = obj.get("arguments").ok_or_else(|| {
                            ResponsesApiError::bad_request("function_call item missing 'arguments'")
                        })?;
                        let args_str = match args_val {
                            Value::String(s) => s.clone(),
                            Value::Object(_) => args_val.to_string(),
                            _ => {
                                return Err(ResponsesApiError::bad_request(
                                    "function_call 'arguments' must be a JSON string or object",
                                ));
                            }
                        };
                        // Validate JSON object shape early so chatml serialization succeeds cleanly.
                        let parsed_args: Value = serde_json::from_str(&args_str).map_err(|e| {
                            ResponsesApiError::bad_request(format!(
                                "function_call 'arguments' must be valid JSON object: {e}"
                            ))
                        })?;
                        if !parsed_args.is_object() {
                            return Err(ResponsesApiError::bad_request(
                                "function_call 'arguments' must decode to a JSON object",
                            ));
                        }
                        let call_id = obj
                            .get("call_id")
                            .or_else(|| obj.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or("call_0");
                        let tc = json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": parsed_args,
                            }
                        });
                        if let Some(last) = turns
                            .last_mut()
                            .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
                        {
                            if !last.get("tool_calls").is_some_and(Value::is_array) {
                                last["tool_calls"] = json!([]);
                            }
                            last["tool_calls"].as_array_mut().unwrap().push(tc);
                        } else {
                            let mut msg = json!({
                                "role": "assistant",
                                "content": Value::Null,
                                "tool_calls": [tc],
                            });
                            if let Some(r) = pending_reasoning.take() {
                                msg["reasoning_content"] = json!(r);
                            }
                            turns.push(msg);
                        }
                    }
                    "function_call_output" => {
                        pending_reasoning = None;
                        let call_id = obj
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or("call_0");
                        let output_val = obj.get("output").ok_or_else(|| {
                            ResponsesApiError::bad_request(
                                "function_call_output item missing 'output'",
                            )
                        })?;
                        let output_text = match output_val {
                            Value::String(s) => s.clone(),
                            Value::Array(_) => {
                                let converted = convert_content_parts(output_val, "user")?;
                                match converted {
                                    Value::String(s) => s,
                                    other => other.to_string(),
                                }
                            }
                            other => other.to_string(),
                        };
                        turns.push(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output_text,
                        }));
                    }
                    "input_text" => {
                        let t = obj.get("text").and_then(Value::as_str).ok_or_else(|| {
                            ResponsesApiError::bad_request("input_text item missing string 'text'")
                        })?;
                        turns.push(json!({"role": "user", "content": t}));
                    }
                    other => {
                        return Err(ResponsesApiError::bad_request(format!(
                            "unsupported Responses input item type '{other}'"
                        )));
                    }
                }
            }
            Ok(())
        }
        _ => Err(ResponsesApiError::bad_request(
            "'input' must be a string or non-empty array of input items",
        )),
    }
}

fn parse_reasoning_and_text_config(
    obj: &Map<String, Value>,
) -> Result<(Option<bool>, Value, Option<Value>, Value), ResponsesApiError> {
    let mut enable_thinking: Option<bool> = None;
    let reasoning_echo = if let Some(r_val) = obj.get("reasoning").filter(|v| !v.is_null()) {
        let r_obj = r_val
            .as_object()
            .ok_or_else(|| ResponsesApiError::bad_request("'reasoning' must be an object"))?;
        if let Some(effort) = r_obj.get("effort").filter(|v| !v.is_null()) {
            let eff = effort.as_str().ok_or_else(|| {
                ResponsesApiError::bad_request("'reasoning.effort' must be a string")
            })?;
            match eff {
                "none" | "minimal" => enable_thinking = Some(false),
                "low" | "medium" | "high" | "xhigh" | "max" => enable_thinking = Some(true),
                other => {
                    return Err(ResponsesApiError::bad_request(format!(
                        "unsupported reasoning.effort '{other}'"
                    )));
                }
            }
        }
        json!({
            "effort": r_obj.get("effort").cloned().unwrap_or(Value::Null),
            "summary": r_obj.get("summary").cloned().unwrap_or(Value::Null),
        })
    } else {
        json!({"effort": Value::Null, "summary": Value::Null})
    };

    let mut response_format: Option<Value> = None;
    let text_echo = if let Some(text_val) = obj.get("text").filter(|v| !v.is_null()) {
        let text_obj = text_val
            .as_object()
            .ok_or_else(|| ResponsesApiError::bad_request("'text' must be an object"))?;
        if let Some(fmt_val) = text_obj.get("format").filter(|v| !v.is_null()) {
            let fmt_obj = fmt_val
                .as_object()
                .ok_or_else(|| ResponsesApiError::bad_request("'text.format' must be an object"))?;
            let fmt_type = fmt_obj.get("type").and_then(Value::as_str).ok_or_else(|| {
                ResponsesApiError::bad_request("'text.format.type' must be a string")
            })?;
            match fmt_type {
                "text" => {}
                "json_object" => {
                    response_format = Some(json!({"type": "json_object"}));
                }
                "json_schema" => {
                    let schema = fmt_obj.get("schema").ok_or_else(|| {
                        ResponsesApiError::bad_request(
                            "text.format type 'json_schema' requires 'schema'",
                        )
                    })?;
                    let name = fmt_obj
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("response_schema");
                    response_format = Some(json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": name,
                            "strict": true,
                            "schema": schema,
                        }
                    }));
                }
                other => {
                    return Err(ResponsesApiError::bad_request(format!(
                        "unsupported text.format.type '{other}'"
                    )));
                }
            }
            json!({"format": fmt_val})
        } else {
            json!({"format": {"type": "text"}})
        }
    } else {
        json!({"format": {"type": "text"}})
    };
    Ok((enable_thinking, reasoning_echo, response_format, text_echo))
}

/// Translates an OpenAI Responses API request body into `(canonical_chat_request, meta)`.
pub fn translate_responses_to_chat(
    req: &Value,
    default_model: &str,
    store: &ResponseStore,
    caller_scope: Option<&str>,
    request_seq_id: u64,
) -> Result<(Value, ResponsesPreparedMeta), ResponsesApiError> {
    let obj = req
        .as_object()
        .ok_or_else(|| ResponsesApiError::bad_request("request body must be a JSON object"))?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or(default_model)
        .to_owned();

    let mut system_chunks: Vec<String> = Vec::new();
    let instructions_val = obj.get("instructions").cloned().unwrap_or(Value::Null);
    if !instructions_val.is_null() {
        let instr = instructions_val.as_str().ok_or_else(|| {
            ResponsesApiError::bad_request("'instructions' must be a string or null")
        })?;
        if !instr.trim().is_empty() {
            system_chunks.push(instr.to_owned());
        }
    }

    let mut turns: Vec<Value> = Vec::new();
    let prev_id_val = obj
        .get("previous_response_id")
        .cloned()
        .unwrap_or(Value::Null);
    if !prev_id_val.is_null() {
        let prev_id = prev_id_val.as_str().ok_or_else(|| {
            ResponsesApiError::bad_request("'previous_response_id' must be a string")
        })?;
        let scope = caller_scope.ok_or_else(|| {
            ResponsesApiError::not_found(format!("previous_response_id '{prev_id}' not found"))
        })?;
        let history = store.get_history(prev_id, scope).ok_or_else(|| {
            ResponsesApiError::not_found(format!("previous_response_id '{prev_id}' not found"))
        })?;
        for msg in history {
            if msg.get("role").and_then(Value::as_str) == Some("system") {
                if system_chunks.is_empty()
                    && let Some(s) = msg.get("content").and_then(Value::as_str)
                    && !s.trim().is_empty()
                {
                    system_chunks.push(s.to_owned());
                }
            } else {
                turns.push(msg);
            }
        }
    }

    let input = obj
        .get("input")
        .ok_or_else(|| ResponsesApiError::bad_request("missing required field 'input'"))?;
    parse_input_items(input, &mut system_chunks, &mut turns)?;

    let mut messages = Vec::with_capacity(turns.len() + usize::from(!system_chunks.is_empty()));
    if !system_chunks.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": system_chunks.join("\n\n"),
        }));
    }
    messages.extend(turns);

    // Tools & tool_choice
    let mut chat_tools = Vec::new();
    let mut resp_tools = Vec::new();
    if let Some(tools_val) = obj.get("tools").filter(|v| !v.is_null()) {
        let arr = tools_val
            .as_array()
            .ok_or_else(|| ResponsesApiError::bad_request("'tools' must be an array"))?;
        for t in arr {
            let (ct, rt) = normalize_tool_def(t)?;
            chat_tools.push(ct);
            resp_tools.push(rt);
        }
    }
    let (chat_tool_choice, tool_choice_echo) = normalize_tool_choice(obj.get("tool_choice"))?;
    let (enable_thinking, reasoning_echo, response_format, text_echo) =
        parse_reasoning_and_text_config(obj)?;

    let max_output_tokens = match obj
        .get("max_output_tokens")
        .or_else(|| obj.get("max_tokens"))
        .filter(|v| !v.is_null())
    {
        None => 256usize,
        Some(v) => {
            let n = v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| ResponsesApiError::bad_request("invalid max_output_tokens"))?;
            if n == 0 || n > 4096 {
                return Err(ResponsesApiError::bad_request(
                    "max_output_tokens must be in 1..=4096",
                ));
            }
            n
        }
    };

    let stream = obj
        .get("stream")
        .filter(|v| !v.is_null())
        .map_or(Ok(false), |v| {
            v.as_bool()
                .ok_or_else(|| ResponsesApiError::bad_request("'stream' must be a boolean"))
        })?;
    // Persistence is explicit opt-in (`"store": true`) so omitted `store` never retains prompts
    // in shared in-memory state unintentionally.
    let store_flag = obj
        .get("store")
        .filter(|v| !v.is_null())
        .map_or(Ok(false), |v| {
            v.as_bool()
                .ok_or_else(|| ResponsesApiError::bad_request("'store' must be a boolean"))
        })?;
    if store_flag && caller_scope.is_none() {
        return Err(ResponsesApiError::bad_request(
            "persisting responses with 'store': true requires an Authorization or x-api-key header",
        ));
    }
    let parallel_tool_calls = obj
        .get("parallel_tool_calls")
        .filter(|v| !v.is_null())
        .map_or(Ok(true), |v| {
            v.as_bool().ok_or_else(|| {
                ResponsesApiError::bad_request("'parallel_tool_calls' must be a boolean")
            })
        })?;
    let temperature = obj
        .get("temperature")
        .filter(|v| !v.is_null())
        .map_or(Ok(0.0f64), |v| {
            v.as_f64()
                .ok_or_else(|| ResponsesApiError::bad_request("invalid temperature"))
        })?;
    let top_p = obj
        .get("top_p")
        .filter(|v| !v.is_null())
        .map_or(Ok(1.0f64), |v| {
            v.as_f64()
                .ok_or_else(|| ResponsesApiError::bad_request("invalid top_p"))
        })?;
    let truncation = obj
        .get("truncation")
        .and_then(Value::as_str)
        .unwrap_or("disabled")
        .to_owned();
    let metadata = match obj.get("metadata").filter(|v| !v.is_null()) {
        None => json!({}),
        Some(v) if v.is_object() => v.clone(),
        Some(_) => {
            return Err(ResponsesApiError::bad_request(
                "'metadata' must be a JSON object",
            ));
        }
    };

    let mut chat_req = Map::new();
    chat_req.insert("model".into(), json!(model));
    chat_req.insert("messages".into(), Value::Array(messages.clone()));
    chat_req.insert("max_tokens".into(), json!(max_output_tokens));
    chat_req.insert("stream".into(), json!(stream));
    chat_req.insert("parallel_tool_calls".into(), json!(parallel_tool_calls));
    chat_req.insert("temperature".into(), json!(temperature));
    chat_req.insert("top_p".into(), json!(top_p));
    if let Some(top_k) = obj.get("top_k").filter(|v| !v.is_null()) {
        chat_req.insert("top_k".into(), top_k.clone());
    }
    for key in [
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "stop",
        "stop_token_ids",
        "min_tokens",
        "ignore_eos",
        "logit_bias",
        "logprobs",
        "top_logprobs",
    ] {
        if let Some(val) = obj.get(key).filter(|v| !v.is_null()) {
            chat_req.insert(key.into(), val.clone());
        }
    }
    if chat_req
        .get("top_logprobs")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 0)
        && !chat_req.contains_key("logprobs")
    {
        chat_req.insert("logprobs".into(), json!(true));
    }
    if !chat_tools.is_empty() {
        chat_req.insert("tools".into(), Value::Array(chat_tools));
    }
    if let Some(tc) = chat_tool_choice {
        chat_req.insert("tool_choice".into(), tc);
    }
    if let Some(so) = obj.get("structured_outputs").filter(|v| !v.is_null()) {
        if response_format.is_some() {
            return Err(ResponsesApiError::bad_request(
                "cannot specify both structured_outputs and non-text text.format",
            ));
        }
        chat_req.insert("structured_outputs".into(), so.clone());
    }
    if let Some(rf) = response_format {
        chat_req.insert("response_format".into(), rf);
    }
    let mut template_kwargs = match obj.get("chat_template_kwargs").filter(|v| !v.is_null()) {
        None => Map::new(),
        Some(v) => v.as_object().cloned().ok_or_else(|| {
            ResponsesApiError::bad_request("'chat_template_kwargs' must be an object")
        })?,
    };
    if let Some(thinking) = enable_thinking {
        template_kwargs
            .entry("enable_thinking")
            .or_insert_with(|| json!(thinking));
    }
    if !template_kwargs.is_empty() {
        chat_req.insert(
            "chat_template_kwargs".into(),
            Value::Object(template_kwargs),
        );
    }

    let meta = ResponsesPreparedMeta {
        response_id: generate_unguessable_response_id(request_seq_id),
        created_at: unix_timestamp_secs(),
        model,
        instructions: instructions_val,
        previous_response_id: prev_id_val,
        max_output_tokens,
        parallel_tool_calls,
        store: store_flag,
        stream,
        temperature,
        top_p,
        text_echo,
        tool_choice_echo,
        tools_echo: Value::Array(resp_tools),
        reasoning_echo,
        truncation,
        metadata,
        input_messages: messages,
    };
    Ok((Value::Object(chat_req), meta))
}

#[derive(Clone, Debug)]
struct ItemIds {
    reasoning_id: String,
    message_id: String,
}

impl ItemIds {
    fn for_response(response_id: &str) -> Self {
        let suffix = response_id.strip_prefix("resp_").unwrap_or(response_id);
        Self {
            reasoning_id: format!("rs_{suffix}"),
            message_id: format!("msg_{suffix}"),
        }
    }
}

fn build_response_envelope(
    meta: &ResponsesPreparedMeta,
    status: &str,
    output: Vec<Value>,
    output_text: &str,
    incomplete_details: Value,
    error: Value,
    usage: Value,
) -> Value {
    json!({
        "id": meta.response_id,
        "object": "response",
        "created_at": meta.created_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": meta.instructions,
        "max_output_tokens": meta.max_output_tokens,
        "model": meta.model,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": meta.parallel_tool_calls,
        "previous_response_id": meta.previous_response_id,
        "reasoning": meta.reasoning_echo,
        "store": meta.store,
        "temperature": meta.temperature,
        "text": meta.text_echo,
        "tool_choice": meta.tool_choice_echo,
        "tools": meta.tools_echo,
        "top_p": meta.top_p,
        "truncation": meta.truncation,
        "usage": usage,
        "metadata": meta.metadata,
    })
}

fn build_usage_json(
    input_tokens: usize,
    cached_tokens: usize,
    output_tokens: usize,
    reasoning_tokens: usize,
) -> Value {
    json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens.min(output_tokens),
        },
        "total_tokens": input_tokens + output_tokens,
    })
}

/// Builds the completed/incomplete `response` JSON object and updated history messages.
pub fn build_response_object(
    meta: &ResponsesPreparedMeta,
    turn: &ParsedAssistantTurn,
    finish_reason: &str,
    input_tokens: usize,
    cached_tokens: usize,
    output_tokens: usize,
    reasoning_tokens: usize,
) -> (Value, Vec<Value>) {
    let ids = ItemIds::for_response(&meta.response_id);
    let suffix = meta
        .response_id
        .strip_prefix("resp_")
        .unwrap_or(&meta.response_id);
    let mut output = Vec::new();

    if let Some(reasoning) = turn.reasoning_content.as_deref().filter(|s| !s.is_empty()) {
        output.push(json!({
            "id": ids.reasoning_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [{
                "type": "reasoning_text",
                "text": reasoning,
            }],
        }));
    }

    let text = turn.content.as_deref().unwrap_or("");
    if !text.is_empty() || turn.tool_calls.is_empty() {
        output.push(json!({
            "id": ids.message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        }));
    }

    for (idx, tc) in turn.tool_calls.iter().enumerate() {
        let call_id = if tc.id.is_empty() {
            format!("call_{suffix}_{idx}")
        } else {
            tc.id.clone()
        };
        output.push(json!({
            "id": format!("fc_{suffix}_{idx}"),
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        }));
    }

    let is_incomplete = finish_reason == "length";
    let status = if is_incomplete {
        "incomplete"
    } else {
        "completed"
    };
    let incomplete_details = if is_incomplete {
        json!({"reason": "max_output_tokens"})
    } else {
        Value::Null
    };
    let usage = build_usage_json(input_tokens, cached_tokens, output_tokens, reasoning_tokens);
    let response = build_response_envelope(
        meta,
        status,
        output,
        text,
        incomplete_details,
        Value::Null,
        usage,
    );

    let mut history = meta.input_messages.clone();
    let mut assistant_msg = json!({
        "role": "assistant",
        "content": turn.content,
    });
    if let Some(r) = &turn.reasoning_content {
        assistant_msg["reasoning_content"] = json!(r);
    }
    if !turn.tool_calls.is_empty() {
        assistant_msg["tool_calls"] = json!(turn.tool_calls);
    }
    history.push(assistant_msg);

    (response, history)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveItem {
    None,
    Reasoning { output_index: usize },
    Message { output_index: usize },
}

/// Deterministic SSE event state machine for `POST /v1/responses` (`stream: true`).
/// Guarantees strictly monotonic 0-indexed `sequence_number` across all emitted events.
pub struct ResponsesSseFormatter {
    meta: ResponsesPreparedMeta,
    ids: ItemIds,
    next_seq: u64,
    next_output_index: usize,
    next_msg_index: usize,
    next_fc_index: usize,
    active: ActiveItem,
    reasoning_accum: String,
    current_msg_id: String,
    current_msg_accum: String,
    text_accum: String,
    emitted_items: Vec<Value>,
}

impl ResponsesSseFormatter {
    pub fn new(meta: ResponsesPreparedMeta) -> Self {
        let ids = ItemIds::for_response(&meta.response_id);
        Self {
            meta,
            ids,
            next_seq: 0,
            next_output_index: 0,
            next_msg_index: 0,
            next_fc_index: 0,
            active: ActiveItem::None,
            reasoning_accum: String::new(),
            current_msg_id: String::new(),
            current_msg_accum: String::new(),
            text_accum: String::new(),
            emitted_items: Vec::new(),
        }
    }

    fn emit(&mut self, event_type: &str, mut payload: Map<String, Value>) -> (String, Value) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let mut out = Map::with_capacity(payload.len() + 2);
        out.insert("type".into(), json!(event_type));
        out.insert("sequence_number".into(), json!(seq));
        out.append(&mut payload);
        (event_type.to_owned(), Value::Object(out))
    }

    pub fn start(&mut self) -> Vec<(String, Value)> {
        let initial = build_response_envelope(
            &self.meta,
            "in_progress",
            Vec::new(),
            "",
            Value::Null,
            Value::Null,
            Value::Null,
        );
        let mut m1 = Map::new();
        m1.insert("response".into(), initial.clone());
        let ev1 = self.emit("response.created", m1);

        let mut m2 = Map::new();
        m2.insert("response".into(), initial);
        let ev2 = self.emit("response.in_progress", m2);
        vec![ev1, ev2]
    }

    fn close_active(&mut self, out: &mut Vec<(String, Value)>) {
        match self.active {
            ActiveItem::None => {}
            ActiveItem::Reasoning { output_index } => {
                let text = self.reasoning_accum.clone();
                let item_id = self.ids.reasoning_id.clone();

                let mut d1 = Map::new();
                d1.insert("item_id".into(), json!(item_id));
                d1.insert("output_index".into(), json!(output_index));
                d1.insert("content_index".into(), json!(0));
                d1.insert("text".into(), json!(text));
                out.push(self.emit("response.reasoning_text.done", d1));

                let part = json!({"type": "reasoning_text", "text": text});
                let mut d2 = Map::new();
                d2.insert("item_id".into(), json!(item_id));
                d2.insert("output_index".into(), json!(output_index));
                d2.insert("content_index".into(), json!(0));
                d2.insert("part".into(), part.clone());
                out.push(self.emit("response.reasoning_part.done", d2));

                let item = json!({
                    "id": item_id,
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [],
                    "content": [part],
                });
                let mut d3 = Map::new();
                d3.insert("output_index".into(), json!(output_index));
                d3.insert("item".into(), item.clone());
                out.push(self.emit("response.output_item.done", d3));

                self.emitted_items.push(item);
                self.active = ActiveItem::None;
            }
            ActiveItem::Message { output_index } => {
                let text = self.current_msg_accum.clone();
                let item_id = self.current_msg_id.clone();

                let mut d1 = Map::new();
                d1.insert("item_id".into(), json!(item_id));
                d1.insert("output_index".into(), json!(output_index));
                d1.insert("content_index".into(), json!(0));
                d1.insert("text".into(), json!(text));
                d1.insert("logprobs".into(), json!([]));
                out.push(self.emit("response.output_text.done", d1));

                let part = json!({"type": "output_text", "text": text, "annotations": []});
                let mut d2 = Map::new();
                d2.insert("item_id".into(), json!(item_id));
                d2.insert("output_index".into(), json!(output_index));
                d2.insert("content_index".into(), json!(0));
                d2.insert("part".into(), part.clone());
                out.push(self.emit("response.content_part.done", d2));

                let item = json!({
                    "id": item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [part],
                });
                let mut d3 = Map::new();
                d3.insert("output_index".into(), json!(output_index));
                d3.insert("item".into(), item.clone());
                out.push(self.emit("response.output_item.done", d3));

                self.emitted_items.push(item);
                self.current_msg_accum.clear();
                self.active = ActiveItem::None;
            }
        }
    }

    fn emit_tool_call(&mut self, tc: &ParsedToolCall, out: &mut Vec<(String, Value)>) {
        self.close_active(out);
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        let fc_idx = self.next_fc_index;
        self.next_fc_index += 1;

        let suffix = self
            .meta
            .response_id
            .strip_prefix("resp_")
            .unwrap_or(&self.meta.response_id);
        let item_id = format!("fc_{suffix}_{fc_idx}");
        let call_id = if tc.id.is_empty() {
            format!("call_{suffix}_{fc_idx}")
        } else {
            tc.id.clone()
        };

        let added_item = json!({
            "id": item_id,
            "type": "function_call",
            "status": "in_progress",
            "call_id": call_id,
            "name": tc.function.name,
            "arguments": "",
        });
        let mut a = Map::new();
        a.insert("output_index".into(), json!(output_index));
        a.insert("item".into(), added_item);
        out.push(self.emit("response.output_item.added", a));

        let mut d = Map::new();
        d.insert("item_id".into(), json!(item_id));
        d.insert("output_index".into(), json!(output_index));
        d.insert("call_id".into(), json!(call_id));
        d.insert("delta".into(), json!(tc.function.arguments));
        out.push(self.emit("response.function_call_arguments.delta", d));

        let mut done_args = Map::new();
        done_args.insert("item_id".into(), json!(item_id));
        done_args.insert("output_index".into(), json!(output_index));
        done_args.insert("call_id".into(), json!(call_id));
        done_args.insert("name".into(), json!(tc.function.name));
        done_args.insert("arguments".into(), json!(tc.function.arguments));
        out.push(self.emit("response.function_call_arguments.done", done_args));

        let done_item = json!({
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        });
        let mut done_ev = Map::new();
        done_ev.insert("output_index".into(), json!(output_index));
        done_ev.insert("item".into(), done_item.clone());
        out.push(self.emit("response.output_item.done", done_ev));

        self.emitted_items.push(done_item);
    }

    pub fn push_delta(&mut self, delta: StreamParseDelta) -> Vec<(String, Value)> {
        let mut out = Vec::new();

        if let Some(r_chunk) = delta.reasoning_delta.filter(|s| !s.is_empty()) {
            let output_index = match self.active {
                ActiveItem::Reasoning { output_index } => output_index,
                _ => {
                    self.close_active(&mut out);
                    let idx = self.next_output_index;
                    self.next_output_index += 1;
                    let item_id = self.ids.reasoning_id.clone();

                    let mut a1 = Map::new();
                    a1.insert("output_index".into(), json!(idx));
                    a1.insert(
                        "item".into(),
                        json!({
                            "id": item_id,
                            "type": "reasoning",
                            "status": "in_progress",
                            "summary": [],
                            "content": [],
                        }),
                    );
                    out.push(self.emit("response.output_item.added", a1));

                    let mut a2 = Map::new();
                    a2.insert("item_id".into(), json!(self.ids.reasoning_id));
                    a2.insert("output_index".into(), json!(idx));
                    a2.insert("content_index".into(), json!(0));
                    a2.insert("part".into(), json!({"type": "reasoning_text", "text": ""}));
                    out.push(self.emit("response.reasoning_part.added", a2));

                    self.active = ActiveItem::Reasoning { output_index: idx };
                    idx
                }
            };
            self.reasoning_accum.push_str(&r_chunk);
            let mut dm = Map::new();
            dm.insert("item_id".into(), json!(self.ids.reasoning_id));
            dm.insert("output_index".into(), json!(output_index));
            dm.insert("content_index".into(), json!(0));
            dm.insert("delta".into(), json!(r_chunk));
            out.push(self.emit("response.reasoning_text.delta", dm));
        }

        if let Some(raw_chunk) = delta.content_delta.filter(|s| !s.is_empty()) {
            let c_chunk = if matches!(self.active, ActiveItem::Message { .. }) {
                Some(raw_chunk)
            } else {
                let trimmed = raw_chunk.trim_start_matches('\n');
                (!trimmed.is_empty()).then(|| trimmed.to_owned())
            };
            if let Some(c_chunk) = c_chunk {
                let output_index = match self.active {
                    ActiveItem::Message { output_index } => output_index,
                    _ => {
                        self.close_active(&mut out);
                        let idx = self.next_output_index;
                        self.next_output_index += 1;
                        let msg_idx = self.next_msg_index;
                        self.next_msg_index += 1;
                        let item_id = if msg_idx == 0 {
                            self.ids.message_id.clone()
                        } else {
                            format!("{}_{msg_idx}", self.ids.message_id)
                        };
                        self.current_msg_id = item_id.clone();
                        self.current_msg_accum.clear();

                        let mut a1 = Map::new();
                        a1.insert("output_index".into(), json!(idx));
                        a1.insert(
                            "item".into(),
                            json!({
                                "id": item_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            }),
                        );
                        out.push(self.emit("response.output_item.added", a1));

                        let mut a2 = Map::new();
                        a2.insert("item_id".into(), json!(self.current_msg_id));
                        a2.insert("output_index".into(), json!(idx));
                        a2.insert("content_index".into(), json!(0));
                        a2.insert(
                            "part".into(),
                            json!({"type": "output_text", "text": "", "annotations": []}),
                        );
                        out.push(self.emit("response.content_part.added", a2));

                        self.active = ActiveItem::Message { output_index: idx };
                        idx
                    }
                };
                self.current_msg_accum.push_str(&c_chunk);
                self.text_accum.push_str(&c_chunk);
                let mut dm = Map::new();
                dm.insert("item_id".into(), json!(self.current_msg_id));
                dm.insert("output_index".into(), json!(output_index));
                dm.insert("content_index".into(), json!(0));
                dm.insert("delta".into(), json!(c_chunk));
                dm.insert("logprobs".into(), json!([]));
                out.push(self.emit("response.output_text.delta", dm));
            }
        }

        for tc in &delta.completed_tool_calls {
            self.emit_tool_call(tc, &mut out);
        }

        out
    }

    pub fn finish(
        &mut self,
        turn: &ParsedAssistantTurn,
        finish_reason: &str,
        input_tokens: usize,
        cached_tokens: usize,
        output_tokens: usize,
        reasoning_tokens: usize,
    ) -> (Vec<(String, Value)>, Value, Vec<Value>) {
        let mut out = Vec::new();
        self.close_active(&mut out);

        // Emit any tool calls that were not yet emitted incrementally
        if self.next_fc_index < turn.tool_calls.len() {
            for tc in &turn.tool_calls[self.next_fc_index..] {
                self.emit_tool_call(tc, &mut out);
            }
        }

        // If neither message nor tool_call was emitted, open and close an empty message item
        let has_msg_or_fc = self.emitted_items.iter().any(|it| {
            matches!(
                it.get("type").and_then(Value::as_str),
                Some("message" | "function_call")
            )
        });
        if !has_msg_or_fc {
            let idx = self.next_output_index;
            self.next_output_index += 1;
            let msg_idx = self.next_msg_index;
            self.next_msg_index += 1;
            let item_id = if msg_idx == 0 {
                self.ids.message_id.clone()
            } else {
                format!("{}_{msg_idx}", self.ids.message_id)
            };
            self.current_msg_id = item_id.clone();
            self.current_msg_accum.clear();

            let mut a1 = Map::new();
            a1.insert("output_index".into(), json!(idx));
            a1.insert(
                "item".into(),
                json!({
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }),
            );
            out.push(self.emit("response.output_item.added", a1));

            let mut a2 = Map::new();
            a2.insert("item_id".into(), json!(item_id));
            a2.insert("output_index".into(), json!(idx));
            a2.insert("content_index".into(), json!(0));
            a2.insert(
                "part".into(),
                json!({"type": "output_text", "text": "", "annotations": []}),
            );
            out.push(self.emit("response.content_part.added", a2));

            self.active = ActiveItem::Message { output_index: idx };
            self.close_active(&mut out);
        }

        let is_incomplete = finish_reason == "length";
        let status = if is_incomplete {
            "incomplete"
        } else {
            "completed"
        };
        let incomplete_details = if is_incomplete {
            json!({"reason": "max_output_tokens"})
        } else {
            Value::Null
        };
        let usage = build_usage_json(input_tokens, cached_tokens, output_tokens, reasoning_tokens);
        let response = build_response_envelope(
            &self.meta,
            status,
            self.emitted_items.clone(),
            &self.text_accum,
            incomplete_details,
            Value::Null,
            usage,
        );

        let term_type = if is_incomplete {
            "response.incomplete"
        } else {
            "response.completed"
        };
        let mut tm = Map::new();
        tm.insert("response".into(), response.clone());
        out.push(self.emit(term_type, tm));

        let (_, history) = build_response_object(
            &self.meta,
            turn,
            finish_reason,
            input_tokens,
            cached_tokens,
            output_tokens,
            reasoning_tokens,
        );
        (out, response, history)
    }

    pub fn fail(&mut self, message: &str) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        let err_obj = json!({
            "code": "server_error",
            "message": message,
        });
        let failed = build_response_envelope(
            &self.meta,
            "failed",
            self.emitted_items.clone(),
            &self.text_accum,
            Value::Null,
            err_obj,
            Value::Null,
        );
        let mut fm = Map::new();
        fm.insert("response".into(), failed);
        out.push(self.emit("response.failed", fm));
        out
    }
}

#[cfg(test)]
mod tests;

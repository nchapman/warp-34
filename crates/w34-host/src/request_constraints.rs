//! HTTP structured-output controls, validated before model dispatch.
use serde_json::Value;

use crate::{
    constraints::{ConstraintFactory, StructuralTag, TokenBitmaskConstraint},
    tool_constraints::{ToolChoice, ToolFormat},
};

pub const MAX_CONSTRAINT_SPEC_BYTES: usize = 64 * 1024;
pub const MAX_TOOLS_SPEC_BYTES: usize = 512 * 1024;

fn check_spec_size_limit(name: &str, value: &Value, limit: usize) -> Result<(), String> {
    let bytes = serde_json::to_vec(value)
        .map(|v| v.len())
        .unwrap_or(limit + 1);
    if bytes > limit {
        return Err(format!("{name} exceeds {limit} byte limit ({bytes} bytes)"));
    }
    Ok(())
}

fn check_spec_size(name: &str, value: &Value) -> Result<(), String> {
    check_spec_size_limit(name, value, MAX_CONSTRAINT_SPEC_BYTES)
}

pub enum RequestConstraint {
    JsonObject,
    JsonSchema(Value),
    Regex(String),
    Choice(Vec<String>),
    Grammar(String),
    StructuralTags {
        tags: Vec<StructuralTag>,
        triggers: Vec<String>,
    },
    Tools {
        tools: Value,
        choice: ToolChoice,
        format: ToolFormat,
        parallel_tool_calls: bool,
    },
}

fn parse_choices(value: &Value) -> Result<Vec<String>, String> {
    let arr = value
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or("choice must be a non-empty array of strings")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let s = item
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("choice entries must be non-empty strings")?;
        out.push(s.to_string());
    }
    Ok(out)
}

pub fn parse_request_tool_format(request: &Value) -> Result<ToolFormat, String> {
    let raw = request
        .get("chat_template_kwargs")
        .and_then(|v| v.get("tool_call_format"))
        .or_else(|| request.get("tool_call_format"))
        .filter(|v| !v.is_null());
    match raw {
        None => Ok(ToolFormat::QwenXml),
        Some(Value::String(s))
            if s.eq_ignore_ascii_case("xml") || s.eq_ignore_ascii_case("qwen_xml") =>
        {
            Ok(ToolFormat::QwenXml)
        }
        Some(Value::String(s))
            if s.eq_ignore_ascii_case("json") || s.eq_ignore_ascii_case("hermes") =>
        {
            Ok(ToolFormat::HermesJson)
        }
        _ => Err("tool_call_format must be 'xml' or 'json'".into()),
    }
}

impl RequestConstraint {
    pub fn parse(request: &Value) -> Result<Option<Self>, String> {
        let mut output = None;
        if let Some(format) = request.get("response_format").filter(|v| !v.is_null()) {
            check_spec_size("response_format", format)?;
            output = match format.get("type").and_then(Value::as_str) {
                Some("text") => None,
                Some("json_object") => Some(Self::JsonObject),
                Some("json_schema") => {
                    let spec = format
                        .get("json_schema")
                        .ok_or("missing response_format.json_schema")?;
                    match spec.get("strict") {
                        None | Some(Value::Null) | Some(Value::Bool(true)) => {}
                        _ => return Err("json_schema requires strict: true".into()),
                    }
                    Some(Self::JsonSchema(
                        spec.get("schema")
                            .ok_or("missing json_schema.schema")?
                            .clone(),
                    ))
                }
                Some("structural_tag") => Some(Self::tags(format)?),
                _ => return Err("unsupported response_format.type".into()),
            };
        }
        if let Some(structured) = request.get("structured_outputs").filter(|v| !v.is_null()) {
            check_spec_size("structured_outputs", structured)?;
            if output.is_some() {
                return Err("response_format and structured_outputs cannot be combined".into());
            }
            let object = structured
                .as_object()
                .ok_or("structured_outputs must be an object")?;
            if object.len() != 1 {
                return Err("structured_outputs requires exactly one of json, regex, choice, grammar, structural_tag".into());
            }
            let (kind, value) = object.iter().next().unwrap();
            output = Some(match kind.as_str() {
                "json" => Self::JsonSchema(value.clone()),
                "regex" => Self::Regex(value.as_str().ok_or("regex must be a string")?.into()),
                "choice" => Self::Choice(parse_choices(value)?),
                "grammar" => Self::Grammar(
                    value
                        .as_str()
                        .ok_or("grammar must be a Lark string")?
                        .into(),
                ),
                "structural_tag" => Self::tags(value)?,
                _ => return Err("unsupported structured_outputs field".into()),
            });
        }
        for guided_key in [
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
        ] {
            if let Some(val) = request.get(guided_key).filter(|v| !v.is_null()) {
                check_spec_size(guided_key, val)?;
                if output.is_some() {
                    return Err("multiple structured output constraints cannot be combined".into());
                }
                output = Some(match guided_key {
                    "guided_json" => {
                        let schema = if let Some(s) = val.as_str() {
                            serde_json::from_str::<Value>(s)
                                .map_err(|e| format!("invalid guided_json string: {e}"))?
                        } else {
                            val.clone()
                        };
                        Self::JsonSchema(schema)
                    }
                    "guided_regex" => {
                        Self::Regex(val.as_str().ok_or("guided_regex must be a string")?.into())
                    }
                    "guided_choice" => Self::Choice(parse_choices(val)?),
                    "guided_grammar" => Self::Grammar(
                        val.as_str()
                            .ok_or("guided_grammar must be a Lark string")?
                            .into(),
                    ),
                    _ => unreachable!(),
                });
            }
        }
        let tool_format = parse_request_tool_format(request)?;
        let parallel_tool_calls = match request.get("parallel_tool_calls") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            _ => return Err("parallel_tool_calls must be a boolean".into()),
        };
        let tools = match request.get("tools") {
            None | Some(Value::Null) => Vec::new(),
            Some(value) => {
                check_spec_size_limit("tools", value, MAX_TOOLS_SPEC_BYTES)?;
                value.as_array().ok_or("tools must be an array")?.clone()
            }
        };
        let choice = request
            .get("tool_choice")
            .filter(|v| !v.is_null())
            .map(|v| {
                check_spec_size("tool_choice", v)?;
                ToolChoice::parse(v)
            })
            .transpose()?;
        if !tools.is_empty() || choice.is_some() {
            let choice = choice.unwrap_or(ToolChoice::Auto);
            if tools.is_empty() {
                match choice {
                    ToolChoice::Auto => return Ok(output),
                    ToolChoice::None => {}
                    ToolChoice::Required | ToolChoice::Named(_) => {
                        return Err("tool_choice requires nonempty tools".into());
                    }
                }
            }
            if output.is_some() {
                if choice == ToolChoice::None {
                    return Ok(output);
                }
                return Err(
                    "tool_choice/tools and output grammars cannot currently be combined".into(),
                );
            }
            output = Some(Self::Tools {
                tools: Value::Array(tools),
                choice,
                format: tool_format,
                parallel_tool_calls,
            });
        }
        Ok(output)
    }

    /// Whether the caller explicitly requested a mandatory output constraint that
    /// must fail closed (`400`) if the active execution path cannot enforce masks.
    /// Opportunistic `tools` with default/`"auto"` choice degrades cleanly on paths
    /// without bitmask sampling instead of breaking standard tool-enabled prompts.
    pub fn is_mandatory(&self) -> bool {
        !matches!(
            self,
            Self::Tools {
                choice: ToolChoice::Auto | ToolChoice::None,
                ..
            }
        )
    }

    fn tags(value: &Value) -> Result<Self, String> {
        let tags = value
            .get("structures")
            .and_then(Value::as_array)
            .ok_or("structural_tag requires structures array")?
            .iter()
            .map(|tag| {
                Ok(StructuralTag {
                    begin: tag
                        .get("begin")
                        .and_then(Value::as_str)
                        .ok_or("structure.begin must be a string")?
                        .into(),
                    end: tag
                        .get("end")
                        .and_then(Value::as_str)
                        .ok_or("structure.end must be a string")?
                        .into(),
                    schema: tag
                        .get("schema")
                        .ok_or("structure.schema is required")?
                        .clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let triggers = value
            .get("triggers")
            .and_then(Value::as_array)
            .ok_or("structural_tag requires triggers array")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "trigger must be a string".into())
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self::StructuralTags { tags, triggers })
    }

    pub fn compile(
        &self,
        factory: &ConstraintFactory,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        match self {
            Self::JsonObject => factory.json_object(thinking),
            Self::JsonSchema(schema) => factory.json_schema(schema.clone(), thinking),
            Self::Regex(regex) => factory.regex(regex, thinking),
            Self::Choice(choices) => factory.choice(choices, thinking),
            Self::Grammar(grammar) => factory.lark(grammar, thinking),
            Self::StructuralTags { tags, triggers } => {
                factory.structural_tags(tags, triggers, thinking)
            }
            Self::Tools {
                tools,
                choice,
                format,
                parallel_tool_calls,
            } => factory.tools_with_options(tools, choice, *format, *parallel_tool_calls, thinking),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn null_optional_fields_and_auto_tool_optionality() {
        let parsed = RequestConstraint::parse(&json!({
            "response_format": null,
            "structured_outputs": null,
            "tools": null,
            "tool_choice": null
        }))
        .unwrap();
        assert!(parsed.is_none());

        let auto_tools = RequestConstraint::parse(&json!({
            "tools": [{"type":"function","function":{"name":"ping"}}],
            "tool_choice": null
        }))
        .unwrap()
        .unwrap();
        assert!(!auto_tools.is_mandatory());

        let none_tools = RequestConstraint::parse(&json!({
            "tools": [{"type":"function","function":{"name":"ping"}}],
            "tool_choice": "none"
        }))
        .unwrap()
        .unwrap();
        assert!(!none_tools.is_mandatory());

        assert!(
            RequestConstraint::parse(&json!({
                "tools": [],
                "tool_choice": "auto"
            }))
            .unwrap()
            .is_none()
        );

        let required_tools = RequestConstraint::parse(&json!({
            "tools": [{"type":"function","function":{"name":"ping"}}],
            "tool_choice": "required"
        }))
        .unwrap()
        .unwrap();
        assert!(required_tools.is_mandatory());
    }

    #[test]
    fn oversized_specs_are_rejected_before_grammar_compilation() {
        let huge = "a".repeat(MAX_CONSTRAINT_SPEC_BYTES + 1);
        let err = RequestConstraint::parse(&json!({
            "structured_outputs": { "regex": huge }
        }))
        .err()
        .expect("oversized regex must fail");
        assert!(err.contains("byte limit"), "{err}");
    }

    #[test]
    fn parses_all_response_format_and_structured_outputs_variants() {
        assert!(
            RequestConstraint::parse(&json!({"response_format": {"type": "text"}}))
                .unwrap()
                .is_none()
        );
        let jo = RequestConstraint::parse(&json!({"response_format": {"type": "json_object"}}))
            .unwrap()
            .unwrap();
        assert!(jo.is_mandatory());

        let js = RequestConstraint::parse(&json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "strict": true,
                    "schema": {"type": "object"}
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert!(js.is_mandatory());
        assert!(
            RequestConstraint::parse(&json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"strict": false, "schema": {"type": "object"}}
                }
            }))
            .is_err()
        );

        let st = RequestConstraint::parse(&json!({
            "response_format": {
                "type": "structural_tag",
                "structures": [{"begin": "<a>", "end": "</a>", "schema": {"type": "object"}}],
                "triggers": ["<a>"]
            }
        }))
        .unwrap()
        .unwrap();
        assert!(st.is_mandatory());

        assert!(
            RequestConstraint::parse(&json!({"structured_outputs": {"json": {"type": "object"}}}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({"structured_outputs": {"regex": "[0-9]+"}}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({"structured_outputs": {"grammar": "start: \"ok\""}}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({
                "structured_outputs": {
                    "structural_tag": {
                        "structures": [{"begin": "<a>", "end": "</a>", "schema": {"type": "object"}}],
                        "triggers": ["<a>"]
                    }
                }
            }))
            .unwrap()
            .is_some()
        );

        let js_default_strict = RequestConstraint::parse(&json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "implicit_strict",
                    "schema": {"type": "object"}
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert!(js_default_strict.is_mandatory());

        assert!(
            RequestConstraint::parse(&json!({"structured_outputs": {"choice": ["yes", "no"]}}))
                .unwrap()
                .is_some()
        );
        assert!(RequestConstraint::parse(&json!({"structured_outputs": {"choice": []}})).is_err());
    }

    #[test]
    fn parses_guided_aliases_and_mutual_exclusion() {
        assert!(
            RequestConstraint::parse(&json!({"guided_json": {"type": "object"}}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({"guided_json": "{\"type\":\"object\"}"}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({"guided_regex": "[a-z]+"}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({"guided_choice": ["a", "b"]}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({"guided_grammar": "start: \"ok\""}))
                .unwrap()
                .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({
                "guided_json": {"type": "object"},
                "guided_regex": "a"
            }))
            .is_err()
        );
        assert!(
            RequestConstraint::parse(&json!({
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "parallel_tool_calls": false,
                "chat_template_kwargs": {"tool_call_format": "json"}
            }))
            .unwrap()
            .is_some()
        );
        assert!(
            RequestConstraint::parse(&json!({
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "parallel_tool_calls": "no"
            }))
            .is_err()
        );
        assert!(
            RequestConstraint::parse(&json!({
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_call_format": "yaml"
            }))
            .is_err()
        );

        // Mutual exclusion & validation errors
        assert!(
            RequestConstraint::parse(&json!({
                "response_format": {"type": "json_object"},
                "structured_outputs": {"regex": "a"}
            }))
            .is_err()
        );
        assert!(
            RequestConstraint::parse(&json!({
                "response_format": {"type": "json_object"},
                "tools": [{"type": "function", "function": {"name": "f"}}]
            }))
            .is_err()
        );
        // `tool_choice: "none"` and empty `tools: []` coexist with output constraints
        assert!(matches!(
            RequestConstraint::parse(&json!({
                "response_format": {"type": "json_object"},
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": "none"
            }))
            .unwrap(),
            Some(RequestConstraint::JsonObject)
        ));
        assert!(matches!(
            RequestConstraint::parse(&json!({
                "response_format": {"type": "json_object"},
                "tools": []
            }))
            .unwrap(),
            Some(RequestConstraint::JsonObject)
        ));
        assert!(
            RequestConstraint::parse(&json!({"tools": [], "tool_choice": "required"})).is_err()
        );
    }
}

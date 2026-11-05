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
    Grammar(String),
    StructuralTags {
        tags: Vec<StructuralTag>,
        triggers: Vec<String>,
    },
    Tools {
        tools: Value,
        choice: ToolChoice,
    },
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
                    if spec.get("strict") != Some(&Value::Bool(true)) {
                        return Err("json_schema requires strict: true".into());
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
                return Err("structured_outputs requires exactly one of json, regex, grammar, structural_tag".into());
            }
            let (kind, value) = object.iter().next().unwrap();
            output = Some(match kind.as_str() {
                "json" => Self::JsonSchema(value.clone()),
                "regex" => Self::Regex(value.as_str().ok_or("regex must be a string")?.into()),
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
            if output.is_some() {
                return Err(
                    "tool_choice/tools and output grammars cannot currently be combined".into(),
                );
            }
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
            output = Some(Self::Tools {
                tools: Value::Array(tools),
                choice,
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
            Self::Grammar(grammar) => factory.lark(grammar, thinking),
            Self::StructuralTags { tags, triggers } => {
                factory.structural_tags(tags, triggers, thinking)
            }
            Self::Tools { tools, choice } => {
                factory.tools(tools, choice, ToolFormat::QwenXml, thinking)
            }
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
        assert!(
            RequestConstraint::parse(&json!({"tools": [], "tool_choice": "required"})).is_err()
        );
    }
}

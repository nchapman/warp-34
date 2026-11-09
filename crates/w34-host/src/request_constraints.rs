//! HTTP structured-output controls, validated before model dispatch.
use serde_json::Value;

use crate::{
    constraints::{ConstraintFactory, TokenBitmaskConstraint},
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
    StructuralTag(Value),
    Tools {
        tools: Value,
        choice: ToolChoice,
        parallel: bool,
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
            let active: Vec<_> = object.iter().filter(|(_, v)| !v.is_null()).collect();
            if active.len() != 1 {
                return Err("structured_outputs requires exactly one of json, json_object, regex, grammar, structural_tag, choice".into());
            }
            let (kind, value) = active[0];
            output = Some(match kind.as_str() {
                "json" => Self::JsonSchema(if let Some(text) = value.as_str() {
                    serde_json::from_str(text).map_err(|e| format!("invalid JSON schema: {e}"))?
                } else {
                    value.clone()
                }),
                "json_object" if value == &Value::Bool(true) => Self::JsonObject,
                "choice" => {
                    let choices = value
                        .as_array()
                        .filter(|v| !v.is_empty())
                        .ok_or("choice must be a nonempty array of strings")?;
                    let elements = choices
                        .iter()
                        .map(|v| {
                            v.as_str()
                                .map(|s| serde_json::json!({"type":"const_string","value":s}))
                                .ok_or("choice entries must be strings")
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Self::StructuralTag(serde_json::json!({"type":"structural_tag",
                        "format":{"type":"or","elements":elements}}))
                }
                "regex" => Self::Regex(value.as_str().ok_or("regex must be a string")?.into()),
                "grammar" => Self::Grammar(
                    value
                        .as_str()
                        .ok_or("grammar must be a Lark or EBNF string")?
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
            let choice = choice.unwrap_or(ToolChoice::Auto);
            if choice == ToolChoice::None {
                return Ok(output);
            }
            if tools.is_empty() {
                if choice == ToolChoice::Auto {
                    return Ok(output);
                }
                return Err("tool_choice requires nonempty tools".into());
            }
            if output.is_some() {
                return Err(
                    "tool_choice/tools and output grammars cannot currently be combined".into(),
                );
            }
            let parallel = request
                .get("parallel_tool_calls")
                .filter(|v| !v.is_null())
                .map(|v| v.as_bool().ok_or("parallel_tool_calls must be a boolean"))
                .transpose()?
                .unwrap_or(true);
            output = Some(Self::Tools {
                tools: Value::Array(tools),
                choice,
                parallel,
            });
        }
        Ok(output)
    }

    fn tags(value: &Value) -> Result<Self, String> {
        let value: Value = if let Some(text) = value.as_str() {
            serde_json::from_str(text).map_err(|e| format!("invalid structural_tag JSON: {e}"))?
        } else {
            value.clone()
        };
        if value.get("format").is_some() {
            let mut native = value
                .as_object()
                .ok_or("structural_tag must be an object")?
                .clone();
            native.insert("type".into(), Value::String("structural_tag".into()));
            return Ok(Self::StructuralTag(Value::Object(native)));
        }
        let tags = value
            .get("structures")
            .and_then(Value::as_array)
            .ok_or("structural_tag requires format or structures")?
            .iter()
            .map(|tag| {
                let begin = tag
                    .get("begin")
                    .and_then(Value::as_str)
                    .ok_or("structure.begin must be a string")?;
                let end = tag
                    .get("end")
                    .and_then(Value::as_str)
                    .ok_or("structure.end must be a string")?;
                let schema = tag.get("schema").ok_or("structure.schema is required")?;
                crate::constraints::validate_json_schema(schema)?;
                Ok(serde_json::json!({"type":"tag","begin":begin,"end":end,
                    "content":{"type":"json_schema","json_schema":schema}}))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let triggers = value
            .get("triggers")
            .and_then(Value::as_array)
            .ok_or("structural_tag requires triggers array")?;
        if triggers
            .iter()
            .any(|v| v.as_str().is_none_or(str::is_empty))
        {
            return Err("triggers must be nonempty strings".into());
        }
        Ok(Self::StructuralTag(
            serde_json::json!({"type":"structural_tag",
            "format":{"type":"triggered_tags","tags":tags,"triggers":triggers}}),
        ))
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
            Self::StructuralTag(value) => factory.structural_tag(value.clone(), thinking),
            Self::Tools {
                tools,
                choice,
                parallel,
            } => {
                factory.tools_with_parallel(tools, choice, ToolFormat::QwenXml, thinking, *parallel)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accepts_vllm_output_forms() {
        for request in [
            json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}),
            json!({"response_format":{"type":"json_schema","json_schema":{"strict":false,"schema":true}}}),
            json!({"structured_outputs":{"json_object":true}}),
            json!({"structured_outputs":{"json":"{\"type\":\"object\"}"}}),
            json!({"structured_outputs":{"regex":"[a-z]+"}}),
            json!({"structured_outputs":{"grammar":"root ::= \"ok\""}}),
            json!({"structured_outputs":{"grammar":"start: \"ok\""}}),
            json!({"structured_outputs":{"choice":["yes","no"]}}),
            json!({"structured_outputs":{"structural_tag":{"format":{"type":"const_string","value":"ok"}}}}),
            json!({"response_format":{"type":"structural_tag","structures":[{"begin":"<a>","end":"</a>","schema":{"type":"object"}}],"triggers":["<a>"]}}),
        ] {
            let parsed = RequestConstraint::parse(&request).unwrap().unwrap();
            if let RequestConstraint::StructuralTag(descriptor) = parsed {
                xgrammar::Grammar::from_structural_tag(&descriptor.to_string()).unwrap();
            }
        }
    }

    #[test]
    fn none_is_unconstrained_and_conflicts_fail() {
        assert!(RequestConstraint::parse(&json!({"tools":[{"type":"function","function":{"name":"ping"}}],"tool_choice":"none"})).unwrap().is_none());
        assert!(
            RequestConstraint::parse(
                &json!({"tools":null,"response_format":null,"structured_outputs":null})
            )
            .unwrap()
            .is_none()
        );
        assert!(RequestConstraint::parse(&json!({"tools":[],"tool_choice":"required"})).is_err());
        assert!(RequestConstraint::parse(&json!({"response_format":{"type":"json_object"},"structured_outputs":{"regex":"x"}})).is_err());
        assert!(
            RequestConstraint::parse(
                &json!({"structured_outputs":{"regex":"x","grammar":"start: x"}})
            )
            .is_err()
        );
        assert!(
            RequestConstraint::parse(
                &json!({"structured_outputs":{"regex":"a".repeat(MAX_CONSTRAINT_SPEC_BYTES + 1)}})
            )
            .is_err()
        );
    }
}

//! Schema-enforced Qwen XML and Hermes JSON tool-call grammars.
//!
//! XML parameter values are JSON literals, including quoted strings. Qwen's
//! argument converter parses these losslessly (unquoted `true`/`123` strings
//! would otherwise be mistaken for booleans/numbers).

use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
pub enum ToolFormat {
    QwenXml,
    HermesJson,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

impl ToolChoice {
    pub fn parse(value: &Value) -> Result<Self, String> {
        match value.as_str() {
            Some("none") => Ok(Self::None),
            Some("auto") => Ok(Self::Auto),
            Some("required") => Ok(Self::Required),
            _ if value["type"] == "function" => value["function"]["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(|s| Self::Named(s.into()))
                .ok_or_else(|| "named tool_choice requires a nonempty function.name".into()),
            _ => Err("tool_choice must be none, auto, required, or a named function".into()),
        }
    }
}

fn literal(value: &str) -> String {
    serde_json::to_string(value).unwrap()
}
fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && !value
            .chars()
            .any(|c| c.is_whitespace() || "<>=/".contains(c))
}

/// Returns a Lark grammar; unsupported object-level XML constraints fail closed.
/// XML emits only declared properties in schema order, even if additional
/// properties are permitted; this intentionally chooses a schema-valid subset.
/// Use Hermes for open-ended argument dictionaries or cross-property constraints.
/// Hermes supports the full llguidance JSON-schema subset without XML lowering.
pub fn tool_grammar(
    tools: &Value,
    choice: &ToolChoice,
    format: ToolFormat,
) -> Result<String, String> {
    if *choice == ToolChoice::None {
        return Ok("start: TEXT\nTEXT: /(?s:.*)/ & ~/(?s:.*)<tool_call>(?s:.*)/".into());
    }
    let tools = tools
        .as_array()
        .filter(|t| !t.is_empty())
        .ok_or("tools must be a nonempty array")?;
    let mut bodies = Vec::new();
    let mut definitions = String::new();
    let mut names = std::collections::HashSet::new();
    for (index, tool) in tools.iter().enumerate() {
        if tool["type"] != "function" {
            return Err("only function tools are supported".into());
        }
        let function = &tool["function"];
        let name = function["name"]
            .as_str()
            .filter(|s| valid_name(s))
            .ok_or("invalid tool name")?;
        if !names.insert(name) {
            return Err(format!("duplicate tool name: {name}"));
        }
        if let ToolChoice::Named(selected) = choice
            && selected != name
        {
            continue;
        }
        let empty = json!({"type":"object","properties":{},"additionalProperties":false});
        let schema = function.get("parameters").unwrap_or(&empty);
        let body = match format {
            ToolFormat::HermesJson => {
                let schema = json!({"type":"object","properties":{
                    "name":{"const":name},"arguments":schema
                },"required":["name","arguments"],"additionalProperties":false});
                format!("%json {schema}")
            }
            ToolFormat::QwenXml => {
                if schema["type"] != "object" {
                    return Err("XML tools require an object parameter schema".into());
                }
                let schema_obj = schema
                    .as_object()
                    .ok_or("tool parameters must be a schema object")?;
                // Cross-property constraints cannot be enforced by independent XML
                // value grammars. Reject them rather than silently dropping semantics.
                for key in schema_obj.keys() {
                    if !matches!(
                        key.as_str(),
                        "type"
                            | "properties"
                            | "required"
                            | "additionalProperties"
                            | "title"
                            | "description"
                            | "$schema"
                            | "default"
                            | "examples"
                            | "deprecated"
                    ) {
                        return Err(format!(
                            "XML object keyword '{key}' is unsupported; use Hermes JSON tools"
                        ));
                    }
                }
                let properties = match schema.get("properties") {
                    None => serde_json::Map::new(),
                    Some(Value::Object(props)) => props.clone(),
                    _ => return Err("tool properties must be an object".into()),
                };
                let required: Vec<&str> = match schema.get("required") {
                    None => Vec::new(),
                    Some(Value::Array(items)) => items
                        .iter()
                        .map(|v| v.as_str().ok_or("required entries must be strings"))
                        .collect::<Result<_, _>>()?,
                    _ => return Err("required must be an array".into()),
                };
                if required.iter().any(|key| !properties.contains_key(*key)) {
                    return Err(
                        "XML required parameters must have declared property schemas".into(),
                    );
                }
                let mut body = format!("{} WS", literal(&format!("<function={name}>")));
                for (property_index, (key, value_schema)) in properties.iter().enumerate() {
                    if !valid_name(key) {
                        return Err("invalid XML parameter name".into());
                    }
                    let rule = format!("param_{index}_{property_index}");
                    definitions.push_str(&format!(
                        "{rule}: {} WS %json {value_schema} WS {} WS\n",
                        literal(&format!("<parameter={key}>")),
                        literal("</parameter>")
                    ));
                    body.push_str(&format!(
                        " {rule}{}",
                        if required.contains(&key.as_str()) {
                            ""
                        } else {
                            "?"
                        }
                    ));
                }
                body.push_str(" \"</function>\"");
                body
            }
        };
        let rule = format!("call_body_{index}");
        definitions.push_str(&format!("{rule}: {body}\n"));
        bodies.push(rule);
    }
    if bodies.is_empty() {
        return Err("named tool_choice does not name a declared tool".into());
    }
    let mut grammar = if *choice == ToolChoice::Auto {
        // StructTag's lazy trigger forces the tagged branch, preventing TEXT
        // from swallowing an invalid tool call as ordinary assistant content.
        "start: call* tail\ntail: TEXT\nhead[lazy]: TEXT \"<tool_call>\"\nTEXT: /(?s:.*)/\n"
            .to_string()
    } else {
        "start: call+ WS\nhead: WS \"<tool_call>\"\n".to_string()
    };
    grammar.push_str(&format!(
        "call: head WS ({}) WS \"</tool_call>\"\nWS: /[ \\t\\r\\n]*/\n",
        bodies.join(" | ")
    ));
    grammar.push_str(&definitions);
    Ok(grammar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_choice_parsing_and_validation() {
        assert_eq!(ToolChoice::parse(&json!("none")).unwrap(), ToolChoice::None);
        assert_eq!(ToolChoice::parse(&json!("auto")).unwrap(), ToolChoice::Auto);
        assert_eq!(
            ToolChoice::parse(&json!("required")).unwrap(),
            ToolChoice::Required
        );
        assert_eq!(
            ToolChoice::parse(&json!({"type": "function", "function": {"name": "search"}}))
                .unwrap(),
            ToolChoice::Named("search".into())
        );
        assert!(ToolChoice::parse(&json!({"type": "function", "function": {"name": ""}})).is_err());
        assert!(ToolChoice::parse(&json!("invalid")).is_err());
    }

    #[test]
    fn tool_grammar_xml_and_hermes_and_rejections() {
        let tools = json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "units": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ping"
                }
            }
        ]);

        let none_g = tool_grammar(&tools, &ToolChoice::None, ToolFormat::QwenXml).unwrap();
        assert!(none_g.contains("~/(?s:.*)<tool_call>"));

        let xml_auto = tool_grammar(&tools, &ToolChoice::Auto, ToolFormat::QwenXml).unwrap();
        assert!(xml_auto.contains("head[lazy]"));
        assert!(xml_auto.contains("<function=get_weather>"));
        assert!(xml_auto.contains("param_0_0"));

        let hermes_req =
            tool_grammar(&tools, &ToolChoice::Required, ToolFormat::HermesJson).unwrap();
        assert!(hermes_req.contains("start: call+ WS"));
        assert!(hermes_req.contains("%json"));

        let named_g = tool_grammar(
            &tools,
            &ToolChoice::Named("ping".into()),
            ToolFormat::QwenXml,
        )
        .unwrap();
        assert!(named_g.contains("<function=ping>"));
        assert!(!named_g.contains("get_weather"));

        // Validation error paths
        assert!(tool_grammar(&json!([]), &ToolChoice::Auto, ToolFormat::QwenXml).is_err());
        assert!(
            tool_grammar(
                &json!([{"type": "custom", "function": {"name": "x"}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([{"type": "function", "function": {"name": "bad name"}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([
                    {"type": "function", "function": {"name": "dup"}},
                    {"type": "function", "function": {"name": "dup"}}
                ]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &tools,
                &ToolChoice::Named("missing".into()),
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "array"}}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "anyOf": []}}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": "bad"}}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "required": "bad"}}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}, "required": ["missing"]}}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
        assert!(
            tool_grammar(
                &json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"bad=key": {"type": "string"}}}}}]),
                &ToolChoice::Auto,
                ToolFormat::QwenXml
            )
            .is_err()
        );
    }
}

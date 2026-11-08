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

fn contains_xml_tool_delimiter(s: &str) -> bool {
    [
        "</parameter>",
        "</tool_call>",
        "</function>",
        "<parameter=",
        "<function=",
        "<tool_call>",
    ]
    .iter()
    .any(|delim| s.contains(delim))
}

fn is_safe_unquoted_xml_string(s: &str) -> bool {
    !s.is_empty()
        && s == s.trim()
        && !contains_xml_tool_delimiter(s)
        && !s.starts_with(['"', '{', '['])
        && serde_json::from_str::<Value>(s).is_err()
}

fn attach_root_defs(value_schema: &Value, root_schema: &serde_json::Map<String, Value>) -> Value {
    let Some(obj) = value_schema.as_object() else {
        return value_schema.clone();
    };
    let mut cloned = obj.clone();
    for def_key in ["$defs", "definitions"] {
        if !cloned.contains_key(def_key)
            && let Some(defs) = root_schema.get(def_key)
        {
            cloned.insert(def_key.to_string(), defs.clone());
        }
    }
    Value::Object(cloned)
}

fn xml_param_rule(
    rule: &str,
    key: &str,
    value_schema: &Value,
    root_schema: &serde_json::Map<String, Value>,
    uses_unquoted_str: &mut bool,
) -> String {
    let open_tag = literal(&format!("<parameter={key}>"));
    let close_tag = literal("</parameter>");
    let schema_with_defs = attach_root_defs(value_schema, root_schema);
    let json_branch = format!("%json {schema_with_defs}");
    let default_rule = format!("{rule}: {open_tag} WS {json_branch} WS {close_tag} WS\n");
    let Some(obj) = value_schema.as_object() else {
        return default_rule;
    };

    let only_meta_or = |allowed: &[&str]| {
        obj.keys().all(|k| {
            matches!(
                k.as_str(),
                "title"
                    | "description"
                    | "default"
                    | "examples"
                    | "deprecated"
                    | "$defs"
                    | "definitions"
            ) || allowed.contains(&k.as_str())
        })
    };

    let type_allows_string = match obj.get("type") {
        None => true,
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("string")),
        _ => false,
    };
    if !type_allows_string {
        return default_rule;
    }

    if let Some(Value::String(s)) = obj.get("const")
        && only_meta_or(&["type", "const"])
        && is_safe_unquoted_xml_string(s)
    {
        let val_expr = format!("({json_branch} | {})", literal(s));
        return format!("{rule}: {open_tag} WS {val_expr} WS {close_tag} WS\n");
    }

    if let Some(Value::Array(items)) = obj.get("enum")
        && only_meta_or(&["type", "enum"])
    {
        let mut unquoted = Vec::new();
        for item in items {
            if let Value::String(s) = item
                && is_safe_unquoted_xml_string(s)
            {
                let lit = literal(s);
                if !unquoted.contains(&lit) {
                    unquoted.push(lit);
                }
            }
        }
        if !unquoted.is_empty() {
            let val_expr = format!("({json_branch} | {})", unquoted.join(" | "));
            return format!("{rule}: {open_tag} WS {val_expr} WS {close_tag} WS\n");
        }
    }

    let is_plain_string = obj.get("type").and_then(Value::as_str) == Some("string")
        && only_meta_or(&["type", "minLength"])
        && obj
            .get("minLength")
            .and_then(Value::as_u64)
            .is_none_or(|n| n <= 1);
    if is_plain_string {
        *uses_unquoted_str = true;
        return format!(
            "{rule}: {open_tag} WS ({json_branch} WS {close_tag} | unquoted_xml_param_end) WS\n"
        );
    }

    default_rule
}

/// Returns a Lark grammar; unsupported object-level XML constraints fail closed.
/// XML emits declared properties in any order (up to 6 properties; declaration
/// order above 6) and enforces required properties without duplicate tags.
/// Use Hermes for open-ended argument dictionaries or cross-property constraints.
/// Hermes supports the full llguidance JSON-schema subset without XML lowering.
pub fn tool_grammar(
    tools: &Value,
    choice: &ToolChoice,
    format: ToolFormat,
) -> Result<String, String> {
    tool_grammar_with_options(tools, choice, format, true)
}

pub fn tool_grammar_with_options(
    tools: &Value,
    choice: &ToolChoice,
    format: ToolFormat,
    parallel_tool_calls: bool,
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
    let mut uses_unquoted_str = false;
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
                            | "$defs"
                            | "definitions"
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
                let prop_entries: Vec<(&String, &Value)> = properties.iter().collect();
                for (property_index, (key, value_schema)) in prop_entries.iter().enumerate() {
                    if !valid_name(key) {
                        return Err("invalid XML parameter name".into());
                    }
                    let rule = format!("param_{index}_{property_index}");
                    definitions.push_str(&xml_param_rule(
                        &rule,
                        key,
                        value_schema,
                        schema_obj,
                        &mut uses_unquoted_str,
                    ));
                }
                let num_props = prop_entries.len();
                if num_props > 0 && num_props <= 6 {
                    let full_mask = (1usize << num_props) - 1;
                    let mut req_mask = 0usize;
                    for (p, (key, _)) in prop_entries.iter().enumerate() {
                        if required.contains(&key.as_str()) {
                            req_mask |= 1usize << p;
                        }
                    }
                    for mask in 0..full_mask {
                        let mut branches = Vec::with_capacity(num_props);
                        for p in 0..num_props {
                            if (mask & (1usize << p)) != 0 {
                                continue;
                            }
                            let next_mask = mask | (1usize << p);
                            let p_rule = format!("param_{index}_{p}");
                            if next_mask == full_mask {
                                branches.push(p_rule);
                            } else if (next_mask & req_mask) == req_mask {
                                branches.push(format!("{p_rule} params_{index}_{next_mask}?"));
                            } else {
                                branches.push(format!("{p_rule} params_{index}_{next_mask}"));
                            }
                        }
                        definitions.push_str(&format!(
                            "params_{index}_{mask}: {}\n",
                            branches.join(" | ")
                        ));
                    }
                    if req_mask == 0 {
                        body.push_str(&format!(" params_{index}_0?"));
                    } else {
                        body.push_str(&format!(" params_{index}_0"));
                    }
                } else {
                    for (property_index, (key, _)) in prop_entries.iter().enumerate() {
                        let rule = format!("param_{index}_{property_index}");
                        body.push_str(&format!(
                            " {rule}{}",
                            if required.contains(&key.as_str()) {
                                ""
                            } else {
                                "?"
                            }
                        ));
                    }
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
    let body_union = bodies.join(" | ");
    let mut grammar = if *choice == ToolChoice::Auto {
        // StructTag's lazy trigger forces the tagged branch, preventing TEXT
        // from swallowing an invalid tool call as ordinary assistant content,
        // and forbids non-whitespace suffix text after the tool call(s).
        if parallel_tool_calls {
            format!(
                "start: call_turn | tail\ncall_turn: call (WS next_call)* WS\ncall: head WS ({body_union}) WS \"</tool_call>\"\nnext_call: \"<tool_call>\" WS ({body_union}) WS \"</tool_call>\"\ntail: TEXT_NO_TOOL\nhead[lazy]: TEXT \"<tool_call>\"\nTEXT: /(?s:.*)/\nTEXT_NO_TOOL: /(?s:.*)/ & ~/(?s:.*)<tool_call>(?s:.*)/\nWS: /[ \\t\\r\\n]*/\n"
            )
        } else {
            format!(
                "start: call_turn | tail\ncall_turn: call WS\ncall: head WS ({body_union}) WS \"</tool_call>\"\ntail: TEXT_NO_TOOL\nhead[lazy]: TEXT \"<tool_call>\"\nTEXT: /(?s:.*)/\nTEXT_NO_TOOL: /(?s:.*)/ & ~/(?s:.*)<tool_call>(?s:.*)/\nWS: /[ \\t\\r\\n]*/\n"
            )
        }
    } else if parallel_tool_calls {
        format!(
            "start: call+ WS\nhead: WS \"<tool_call>\"\ncall: head WS ({body_union}) WS \"</tool_call>\"\nWS: /[ \\t\\r\\n]*/\n"
        )
    } else {
        format!(
            "start: call WS\nhead: WS \"<tool_call>\"\ncall: head WS ({body_union}) WS \"</tool_call>\"\nWS: /[ \\t\\r\\n]*/\n"
        )
    };
    if uses_unquoted_str {
        grammar.push_str(
            "unquoted_xml_param_end[lazy]: UNQUOTED_XML_BODY \"</parameter>\"\nUNQUOTED_XML_BODY: (/[^ \\t\\r\\n\"\\[{](?s:.*[^ \\t\\r\\n])?/ & ~(\"true\" | \"false\" | \"null\" | /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/ | /(?s:.*)(<\\/tool_call>|<\\/function>|<parameter=|<function=|<tool_call>)(?s:.*)/)) /[ \\t\\r\\n]*/\n",
        );
    }
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

        // >6 properties uses linear declaration-order rule without combinatorial explosion
        let seven_prop_tool = json!([{
            "type": "function",
            "function": {
                "name": "many_args",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "p0": {"type": "integer"},
                        "p1": {"type": "integer"},
                        "p2": {"type": "integer"},
                        "p3": {"type": "integer"},
                        "p4": {"type": "integer"},
                        "p5": {"type": "integer"},
                        "p6": {"type": "integer"}
                    },
                    "required": ["p0", "p6"]
                }
            }
        }]);
        let seven_g =
            tool_grammar(&seven_prop_tool, &ToolChoice::Required, ToolFormat::QwenXml).unwrap();
        assert!(seven_g.contains("param_0_0 param_0_1? param_0_2? param_0_3? param_0_4? param_0_5? param_0_6 \"</function>\""));

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

//! Native XGrammar tool-call structural descriptors.
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

/// Build the native descriptor; `none` imposes no textual restrictions.
/// Parallel calls are allowed by default, as in the upstream builders.
pub fn tool_structural_tag(
    tools: &Value,
    choice: &ToolChoice,
    format: ToolFormat,
) -> Result<Option<Value>, String> {
    tool_structural_tag_with_parallel(tools, choice, format, true)
}

pub fn tool_structural_tag_with_parallel(
    tools: &Value,
    choice: &ToolChoice,
    format: ToolFormat,
    parallel: bool,
) -> Result<Option<Value>, String> {
    if *choice == ToolChoice::None {
        return Ok(None);
    }
    let tools = tools
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("tools must be a nonempty array")?;
    let mut names = std::collections::HashSet::new();
    let mut tags = Vec::new();
    for tool in tools {
        if tool["type"] != "function" {
            return Err("only function tools are supported".into());
        }
        let function = &tool["function"];
        let name = function["name"]
            .as_str()
            .filter(|s| {
                !s.is_empty()
                    && s.chars()
                        .all(|c| c.is_ascii_alphanumeric() || "_-".contains(c))
            })
            .ok_or("invalid tool name")?;
        if !names.insert(name) {
            return Err(format!("duplicate tool name: {name}"));
        }
        let empty = json!({"type":"object","properties":{},"additionalProperties":false});
        let schema = function.get("parameters").unwrap_or(&empty);
        if let ToolChoice::Named(selected) = choice
            && selected != name
        {
            continue;
        }
        crate::constraints::validate_json_schema(schema)?;
        match format {
            ToolFormat::QwenXml => tags.push(json!({
                "type":"tag", "begin":format!("<tool_call>\n<function={name}>\n"),
                "content":{"type":"json_schema","json_schema":schema,"style":"qwen_xml","max_whitespace_cnt":1},
                "end":"\n</function>\n</tool_call>"
            })),
            ToolFormat::HermesJson => {
                for (begin, end) in [
                    ("<tool_call>\n", "}\n</tool_call>"),
                    ("<tool_call>", "}</tool_call>"),
                ] {
                    tags.push(json!({
                        "type":"tag",
                        "begin":format!("{begin}{{\"name\": \"{name}\", \"arguments\": "),
                        "content":{"type":"json_schema","json_schema":schema},
                        "end":end
                    }));
                }
            }
        }
    }
    if tags.is_empty() {
        return Err("named tool_choice does not name a declared tool".into());
    }
    let body = match (format, choice) {
        (ToolFormat::QwenXml, ToolChoice::Named(_)) => tags.remove(0),
        (ToolFormat::QwenXml, _) => json!({
            "type":"triggered_tags", "triggers":["<tool_call>\n<function="],
            "tags":tags, "at_least_one":*choice == ToolChoice::Required,
            "stop_after_first":!parallel, "excludes":["<think>","</think>"]
        }),
        (ToolFormat::HermesJson, ToolChoice::Auto) => json!({
            "type":"triggered_tags", "triggers":["<tool_call>"], "tags":tags,
            "stop_after_first":!parallel
        }),
        (ToolFormat::HermesJson, _) => json!({
            "type":"tags_with_separator", "tags":tags, "separator":"",
            "at_least_one":true, "stop_after_first":!parallel || matches!(choice, ToolChoice::Named(_))
        }),
    };
    Ok(Some(json!({"type":"structural_tag","format":body})))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_tool_descriptors() {
        let tools = json!([
            {"type":"function","function":{"name":"weather","parameters":{
                "type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
            {"type":"function","function":{"name":"ping"}}
        ]);
        assert!(
            tool_structural_tag(&Value::Null, &ToolChoice::None, ToolFormat::QwenXml)
                .unwrap()
                .is_none()
        );
        for format in [ToolFormat::QwenXml, ToolFormat::HermesJson] {
            let auto = tool_structural_tag(&tools, &ToolChoice::Auto, format)
                .unwrap()
                .unwrap();
            assert_eq!(auto["format"]["type"], "triggered_tags");
            let required = tool_structural_tag(&tools, &ToolChoice::Required, format)
                .unwrap()
                .unwrap();
            assert_eq!(required["format"]["at_least_one"], true);
            let named = tool_structural_tag(&tools, &ToolChoice::Named("ping".into()), format)
                .unwrap()
                .unwrap();
            assert!(!named.to_string().contains("weather"));
            for descriptor in [&auto, &required, &named] {
                xgrammar::Grammar::from_structural_tag(&descriptor.to_string()).unwrap();
            }
            assert!(
                tool_structural_tag(&tools, &ToolChoice::Named("missing".into()), format).is_err()
            );
        }
        let xml = tool_structural_tag(&tools, &ToolChoice::Auto, ToolFormat::QwenXml)
            .unwrap()
            .unwrap();
        assert_eq!(xml["format"]["tags"][0]["content"]["style"], "qwen_xml");
        assert_eq!(
            xml["format"]["tags"][0]["content"]["json_schema"],
            tools[0]["function"]["parameters"]
        );
    }
    #[test]
    fn choices_and_invalid_tools() {
        assert_eq!(ToolChoice::parse(&json!("none")).unwrap(), ToolChoice::None);
        assert!(ToolChoice::parse(&json!("invalid")).is_err());
        for tools in [
            json!([]),
            json!([{"type":"custom"}]),
            json!([
                {"type":"function","function":{"name":"x"}},
                {"type":"function","function":{"name":"x"}}
            ]),
        ] {
            assert!(tool_structural_tag(&tools, &ToolChoice::Auto, ToolFormat::QwenXml).is_err());
        }
    }
}

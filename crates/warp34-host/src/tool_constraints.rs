//! Native XGrammar tool-call structural descriptors.
use serde_json::{Value, json};

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
pub fn tool_structural_tag(tools: &Value, choice: &ToolChoice) -> Result<Option<Value>, String> {
    tool_structural_tag_with_parallel(tools, choice, true)
}

pub fn tool_structural_tag_with_parallel(
    tools: &Value,
    choice: &ToolChoice,
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
        tags.push(json!({
            "type":"tag", "begin":format!("<tool_call>\n<function={name}>\n"),
            "content":{"type":"json_schema","json_schema":schema,"style":"qwen_xml","max_whitespace_cnt":1},
            "end":"\n</function>\n</tool_call>"
        }));
    }
    if tags.is_empty() {
        return Err("named tool_choice does not name a declared tool".into());
    }
    let body = match choice {
        ToolChoice::Named(_) => tags.remove(0),
        _ => json!({
            "type":"triggered_tags", "triggers":["<tool_call>\n<function="],
            "tags":tags, "at_least_one":*choice == ToolChoice::Required,
            "stop_after_first":!parallel, "excludes":["<think>","</think>"]
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
            tool_structural_tag(&Value::Null, &ToolChoice::None)
                .unwrap()
                .is_none()
        );
        let auto = tool_structural_tag(&tools, &ToolChoice::Auto)
            .unwrap()
            .unwrap();
        assert_eq!(auto["format"]["type"], "triggered_tags");
        let required = tool_structural_tag(&tools, &ToolChoice::Required)
            .unwrap()
            .unwrap();
        assert_eq!(required["format"]["at_least_one"], true);
        let named = tool_structural_tag(&tools, &ToolChoice::Named("ping".into()))
            .unwrap()
            .unwrap();
        assert!(!named.to_string().contains("weather"));
        for descriptor in [&auto, &required, &named] {
            xgrammar::Grammar::from_structural_tag(&descriptor.to_string()).unwrap();
        }
        assert!(tool_structural_tag(&tools, &ToolChoice::Named("missing".into())).is_err());
        let xml = tool_structural_tag(&tools, &ToolChoice::Auto)
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
            assert!(tool_structural_tag(&tools, &ToolChoice::Auto).is_err());
        }
    }
}

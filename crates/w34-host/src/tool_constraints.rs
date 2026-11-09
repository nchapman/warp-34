//! Schema-enforced Qwen XML and Hermes JSON tool-call grammars.
//!
//! XML parameter values are JSON literals, including quoted strings. Qwen's
//! argument converter parses these losslessly (unquoted `true`/`123` strings
//! would otherwise be mistaken for booleans/numbers).

use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
                .or_else(|| value["name"].as_str())
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
            .any(|c| c.is_whitespace() || "<>=".contains(c))
}

pub(crate) fn is_schema_metadata_key(key: &str) -> bool {
    matches!(
        key,
        "$defs"
            | "definitions"
            | "$schema"
            | "$id"
            | "id"
            | "$anchor"
            | "$comment"
            | "title"
            | "description"
            | "default"
            | "examples"
            | "example"
            | "deprecated"
            | "readOnly"
            | "writeOnly"
            | "nullable"
            | "contentMediaType"
            | "contentEncoding"
            | "xml"
            | "externalDocs"
            | "discriminator"
    ) || (key.starts_with("x-") && key != "x-guidance")
}

fn contains_xml_tool_delimiter(s: &str) -> bool {
    let mut rest = s;
    while let Some(idx) = rest.find('<') {
        let after_lt = rest[idx + 1..].trim_start();
        if let Some(after_slash) = after_lt.strip_prefix('/') {
            let after_slash = after_slash.trim_start();
            for tag in ["parameter", "tool_call", "function"] {
                if let Some(after_tag) = after_slash.strip_prefix(tag)
                    && after_tag.trim_start().starts_with('>')
                {
                    return true;
                }
            }
        } else {
            for tag in ["parameter", "function"] {
                if let Some(after_tag) = after_lt.strip_prefix(tag)
                    && after_tag.trim_start().starts_with('=')
                {
                    return true;
                }
            }
            if let Some(after_tag) = after_lt.strip_prefix("tool_call")
                && after_tag.trim_start().starts_with('>')
            {
                return true;
            }
        }
        rest = &rest[idx + 1..];
    }
    false
}

fn is_safe_unquoted_xml_string(s: &str, allow_json_primitives: bool) -> bool {
    !s.is_empty()
        && s == s.trim()
        && s != "null"
        && !contains_xml_tool_delimiter(s)
        && !s.starts_with('"')
        && (allow_json_primitives
            || (!s.starts_with(['{', '[']) && serde_json::from_str::<Value>(s).is_err()))
}

fn rewrite_local_property_refs(value: &mut Value, prop_prefix: &str) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(r)) = map.get_mut("$ref")
                && let Some(rest) = r.strip_prefix(prop_prefix)
            {
                *r = format!("#/{rest}");
            }
            for v in map.values_mut() {
                rewrite_local_property_refs(v, prop_prefix);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                rewrite_local_property_refs(v, prop_prefix);
            }
        }
        _ => {}
    }
}

fn attach_root_defs(
    prop_key: Option<&str>,
    value_schema: &Value,
    root_schema: &serde_json::Map<String, Value>,
) -> Value {
    let Some(obj) = value_schema.as_object() else {
        return value_schema.clone();
    };
    let mut cloned = obj.clone();
    for def_key in ["$defs", "definitions"] {
        let Some(root_defs) = root_schema.get(def_key).and_then(Value::as_object) else {
            continue;
        };
        if let Some(Value::Object(local_defs)) = cloned.get_mut(def_key) {
            for (k, v) in root_defs {
                local_defs.entry(k.clone()).or_insert_with(|| v.clone());
            }
        } else if !cloned.contains_key(def_key) {
            cloned.insert(def_key.to_string(), Value::Object(root_defs.clone()));
        }
    }
    let mut out = Value::Object(cloned);
    if let Some(key) = prop_key {
        let prefix = format!("#/properties/{key}/");
        rewrite_local_property_refs(&mut out, &prefix);
    }
    out
}

#[derive(Default)]
struct UnquotedXmlBranches {
    has_plain_string: bool,
    plain_string_min_length: Option<u64>,
    plain_string_max_length: Option<u64>,
    plain_string_pattern: Option<String>,
    plain_string_format: Option<String>,
    has_null: bool,
    only_string_and_null: bool,
    raw_literals: Vec<String>,
    literals: Vec<String>,
}

fn collect_unquoted_xml_string_branches(
    schema: &Value,
    root_schema: &Value,
) -> UnquotedXmlBranches {
    let mut out = UnquotedXmlBranches {
        only_string_and_null: true,
        ..Default::default()
    };
    collect_unquoted_xml_inner(schema, root_schema, 0, &mut out);
    for s in &out.raw_literals {
        if is_safe_unquoted_xml_string(s, out.only_string_and_null) {
            let lit = literal(s);
            if !out.literals.contains(&lit) {
                out.literals.push(lit);
            }
        } else if out.only_string_and_null {
            // Unsafe literal (e.g. padded with whitespace or `"null"`) must remain JSON-quoted;
            // keep the other safe literals available unquoted.
        }
    }
    out
}

fn merge_plain_min_length(dst: &mut Option<u64>, min_len: u64) {
    *dst = Some(dst.map_or(min_len, |prev| prev.min(min_len)));
}

fn merge_plain_max_length(dst: &mut Option<u64>, max_len: Option<u64>, first_plain: bool) {
    if first_plain {
        *dst = max_len;
    } else {
        *dst = match (*dst, max_len) {
            (Some(a), Some(b)) => Some(a.max(b)),
            _ => None,
        };
    }
}

fn collect_unquoted_xml_inner(
    schema: &Value,
    root_schema: &Value,
    depth: usize,
    out: &mut UnquotedXmlBranches,
) {
    if depth > 8 {
        out.only_string_and_null = false;
        return;
    }
    let Some(obj) = schema.as_object() else {
        out.only_string_and_null = false;
        return;
    };
    let only_meta_or = |allowed: &[&str]| {
        obj.keys()
            .all(|k| is_schema_metadata_key(k.as_str()) || allowed.contains(&k.as_str()))
    };
    if let Some(Value::String(ref_str)) = obj.get("$ref")
        && only_meta_or(&["$ref"])
        && let Some(ptr) = ref_str.strip_prefix('#')
        && let Some(target) = root_schema.pointer(ptr)
    {
        collect_unquoted_xml_inner(target, root_schema, depth + 1, out);
        return;
    }
    if let Some(Value::Array(items)) = obj.get("allOf")
        && items.len() == 1
        && only_meta_or(&["allOf"])
    {
        collect_unquoted_xml_inner(&items[0], root_schema, depth + 1, out);
        return;
    }
    if let Some(Value::Array(items)) = obj.get("anyOf")
        && only_meta_or(&["anyOf"])
    {
        for sub in items {
            collect_unquoted_xml_inner(sub, root_schema, depth + 1, out);
        }
        return;
    }
    if obj.get("type").and_then(Value::as_str) == Some("null") && only_meta_or(&["type"]) {
        out.has_null = true;
        return;
    }
    let type_allows_string = match obj.get("type") {
        None => true,
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(arr)) => {
            if arr
                .iter()
                .all(|v| matches!(v.as_str(), Some("string" | "null")))
            {
                if arr.iter().any(|v| v.as_str() == Some("null")) {
                    out.has_null = true;
                }
            } else {
                out.only_string_and_null = false;
            }
            arr.iter().any(|v| v.as_str() == Some("string"))
        }
        _ => false,
    };
    if !type_allows_string {
        out.only_string_and_null = false;
        return;
    }
    if let Some(Value::String(s)) = obj.get("const")
        && only_meta_or(&["type", "const"])
    {
        if !out.raw_literals.contains(s) {
            out.raw_literals.push(s.clone());
        }
        return;
    }
    if let Some(Value::Array(items)) = obj.get("enum")
        && only_meta_or(&["type", "enum"])
    {
        for item in items {
            if let Value::String(s) = item {
                if !out.raw_literals.contains(s) {
                    out.raw_literals.push(s.clone());
                }
            } else if item.is_null() {
                out.has_null = true;
            } else {
                out.only_string_and_null = false;
            }
        }
        return;
    }
    let explicit_string_type = match obj.get("type") {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("string")),
        _ => false,
    };
    if explicit_string_type
        && only_meta_or(&["type", "minLength", "maxLength", "pattern", "format"])
    {
        let first_plain = !out.has_plain_string;
        let min_len = obj.get("minLength").and_then(Value::as_u64).unwrap_or(0);
        let max_len = obj.get("maxLength").and_then(Value::as_u64);
        let pat = obj.get("pattern").and_then(Value::as_str);
        let fmt = obj.get("format").and_then(Value::as_str);
        if max_len.is_some_and(|max| max < min_len) {
            out.only_string_and_null = false;
            return;
        }
        if !first_plain
            && (pat.is_some()
                || fmt.is_some()
                || out.plain_string_pattern.is_some()
                || out.plain_string_format.is_some())
        {
            out.only_string_and_null = false;
            return;
        }
        if let Some(f) = fmt
            && !matches!(
                f,
                "date" | "time" | "date-time" | "uuid" | "ipv4" | "email" | "uri" | "hostname"
            )
        {
            out.only_string_and_null = false;
            return;
        }
        out.has_plain_string = true;
        merge_plain_min_length(&mut out.plain_string_min_length, min_len);
        merge_plain_max_length(&mut out.plain_string_max_length, max_len, first_plain);
        if let Some(p) = pat {
            out.plain_string_pattern = Some(p.to_string());
        }
        if let Some(f) = fmt {
            out.plain_string_format = Some(f.to_string());
        }
    } else {
        out.only_string_and_null = false;
    }
}

fn json_pattern_to_lark_regex(pat: &str) -> String {
    let (anchored_start, body) = match pat.strip_prefix('^') {
        Some(rest) => (true, rest),
        None => (false, pat),
    };
    let anchored_end = body.ends_with('$') && !body.ends_with("\\$");
    let core = if anchored_end {
        &body[..body.len() - 1]
    } else {
        body
    };
    let mut escaped = String::with_capacity(core.len() + 16);
    let mut prev_backslash = false;
    for ch in core.chars() {
        if ch == '/' && !prev_backslash {
            escaped.push_str("\\/");
        } else {
            escaped.push(ch);
        }
        prev_backslash = ch == '\\' && !prev_backslash;
    }
    let prefix = if anchored_start { "" } else { "(?s:.*)" };
    let suffix = if anchored_end { "" } else { "(?s:.*)" };
    format!("/{prefix}({escaped}){suffix}/")
}

fn format_to_lark_regex(fmt: &str) -> Option<&'static str> {
    match fmt {
        "date" => Some("/[0-9]{4}-[0-9]{2}-[0-9]{2}/"),
        "time" => Some("/[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})?/"),
        "date-time" => Some(
            "/[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ][0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})?/",
        ),
        "uuid" => {
            Some("/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/")
        }
        "ipv4" => Some("/([0-9]{1,3}\\.){3}[0-9]{1,3}/"),
        "email" => Some("/[^ \\t\\r\\n@<>]+@[^ \\t\\r\\n@<>]+\\.[^ \\t\\r\\n@<>]+/"),
        "uri" => Some("/[a-zA-Z][a-zA-Z0-9+.\\-]*:[^\\r\\n \\t]*/"),
        "hostname" => Some(
            "/[a-zA-Z0-9]([a-zA-Z0-9\\-]*[a-zA-Z0-9])?(\\.[a-zA-Z0-9]([a-zA-Z0-9\\-]*[a-zA-Z0-9])?)*/",
        ),
        _ => None,
    }
}

const XML_DELIM_BAN_REGEX: &str = "/(?s:.*)(<[ \\t\\r\\n]*\\/[ \\t\\r\\n]*(parameter|tool_call|function)[ \\t\\r\\n]*>|<[ \\t\\r\\n]*(parameter|function)[ \\t\\r\\n]*=|<[ \\t\\r\\n]*tool_call[ \\t\\r\\n]*>)(?s:.*)/";

fn xml_param_rule(
    rule: &str,
    key: &str,
    value_schema: &Value,
    root_schema: &serde_json::Map<String, Value>,
    uses_unquoted_str: &mut bool,
    uses_unquoted_nullable_str: &mut bool,
    extra_defs: &mut String,
) -> String {
    let open_tag = literal(&format!("<parameter={key}>"));
    let close_tag = literal("</parameter>");
    let mut schema_with_defs = attach_root_defs(Some(key), value_schema, root_schema);
    crate::constraints::normalize_schema_for_llguidance(&mut schema_with_defs);
    let json_branch = format!("%json {schema_with_defs}");
    let default_rule = format!("{rule}: {open_tag} WS {json_branch} WS {close_tag} WS\n");
    if !schema_with_defs.is_object() {
        return default_rule;
    }

    let branches = collect_unquoted_xml_string_branches(&schema_with_defs, &schema_with_defs);

    if branches.has_plain_string && branches.only_string_and_null && branches.literals.is_empty() {
        let min_len = branches.plain_string_min_length.unwrap_or(0);
        let max_len = branches.plain_string_max_length;
        let empty_alt = if min_len == 0 {
            format!(" | {close_tag}")
        } else {
            String::new()
        };
        let null_alt = if branches.has_null {
            *uses_unquoted_nullable_str = true;
            " | unquoted_xml_null_end"
        } else {
            ""
        };
        if min_len <= 1
            && max_len.is_none()
            && branches.plain_string_pattern.is_none()
            && branches.plain_string_format.is_none()
        {
            let str_json = if min_len == 1 {
                "%json {\"type\":\"string\",\"minLength\":1}"
            } else {
                "%json {\"type\":\"string\"}"
            };
            *uses_unquoted_str = true;
            return format!(
                "{rule}: {open_tag} WS ({str_json} WS {close_tag} | unquoted_xml_param_end{null_alt}{empty_alt}) WS\n"
            );
        }
        // Constrained length (`minLength > 1` or `maxLength`), `pattern`, or `format` on plain string:
        let mut str_schema = serde_json::Map::new();
        str_schema.insert("type".into(), Value::String("string".into()));
        if min_len > 0 {
            str_schema.insert("minLength".into(), Value::from(min_len));
        }
        if let Some(max) = max_len {
            str_schema.insert("maxLength".into(), Value::from(max));
        }
        if let Some(ref pat) = branches.plain_string_pattern {
            str_schema.insert("pattern".into(), Value::String(pat.clone()));
        }
        if let Some(ref fmt) = branches.plain_string_format {
            str_schema.insert("format".into(), Value::String(fmt.clone()));
        }
        let str_json = format!("%json {}", Value::Object(str_schema));
        if max_len == Some(0) {
            return format!(
                "{rule}: {open_tag} WS ({str_json} WS {close_tag}{null_alt}{empty_alt}) WS\n"
            );
        }
        let mut extra_intersects = String::new();
        if min_len > 1 || max_len.is_some() {
            let eff_min = min_len.max(1);
            let len_rx = match max_len {
                Some(max) => format!("/(?s:.{{{eff_min},{max}}})/"),
                None => format!("/(?s:.{{{eff_min},}})/"),
            };
            extra_intersects.push_str(&format!(" & {len_rx}"));
        }
        if let Some(ref pat) = branches.plain_string_pattern {
            let pat_rx = json_pattern_to_lark_regex(pat);
            extra_intersects.push_str(&format!(" & {pat_rx}"));
        }
        if let Some(ref fmt) = branches.plain_string_format
            && let Some(fmt_rx) = format_to_lark_regex(fmt)
        {
            extra_intersects.push_str(&format!(" & {fmt_rx}"));
        }
        let end_rule = format!("unquoted_end_{rule}");
        let body_tok = format!("UNQUOTED_BODY_{}", rule.to_uppercase());
        extra_defs.push_str(&format!(
            "{end_rule}[lazy]: {body_tok} \"</parameter>\"\n{body_tok}: (/[^ \\t\\r\\n\"](?s:.*[^ \\t\\r\\n])?/{extra_intersects} & ~(\"null\" | {XML_DELIM_BAN_REGEX})) /[ \\t\\r\\n]*/\n"
        ));
        return format!(
            "{rule}: {open_tag} WS ({str_json} WS {close_tag} | {end_rule}{null_alt}{empty_alt}) WS\n"
        );
    }
    if !branches.literals.is_empty() && !branches.has_plain_string {
        let lits = branches.literals.join(" | ");
        return format!(
            "{rule}: {open_tag} /\\r?\\n/? ({json_branch} WS {close_tag} | /[ \\t\\r\\n]+/ {json_branch} WS {close_tag} | ({lits}) /\\r?\\n/? {close_tag}) WS\n"
        );
    }

    default_rule
}

fn build_qwen_xml_tool_body(
    index: usize,
    name: &str,
    schema: &Value,
    definitions: &mut String,
    extra_defs: &mut String,
    uses_unquoted_str: &mut bool,
    uses_unquoted_nullable_str: &mut bool,
) -> Result<String, String> {
    let schema_obj = schema
        .as_object()
        .ok_or("tool parameters must be a schema object")?;
    let is_object_type = match schema_obj.get("type") {
        None => {
            !schema_obj.contains_key("$ref")
                && !schema_obj.contains_key("anyOf")
                && !schema_obj.contains_key("oneOf")
                && !schema_obj.contains_key("allOf")
        }
        Some(Value::String(s)) => s == "object",
        Some(Value::Array(arr)) => arr.len() == 1 && arr[0].as_str() == Some("object"),
        _ => false,
    };
    if !is_object_type {
        return Err("XML tools require an object parameter schema".into());
    }
    // Cross-property constraints cannot be enforced by independent XML
    // value grammars. Reject them rather than silently dropping semantics.
    for key in schema_obj.keys() {
        if !matches!(
            key.as_str(),
            "type" | "properties" | "required" | "additionalProperties"
        ) && !is_schema_metadata_key(key.as_str())
        {
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
        return Err("XML required parameters must have declared property schemas".into());
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
            uses_unquoted_str,
            uses_unquoted_nullable_str,
            extra_defs,
        ));
    }
    let num_props = prop_entries.len();
    if num_props == 0 {
        let addl_schema = match schema_obj.get("additionalProperties") {
            Some(Value::Bool(true)) => Some(json!({})),
            Some(Value::Object(map)) => Some(Value::Object(map.clone())),
            _ => None,
        };
        if let Some(val_schema) = addl_schema {
            let dyn_val_rule = format!("dyn_val_{index}");
            let mut val_with_defs = attach_root_defs(None, &val_schema, schema_obj);
            crate::constraints::normalize_schema_for_llguidance(&mut val_with_defs);
            let branches = collect_unquoted_xml_string_branches(&val_with_defs, &val_with_defs);
            if branches.has_plain_string
                && branches.only_string_and_null
                && branches.literals.is_empty()
            {
                *uses_unquoted_str = true;
                definitions.push_str(&format!(
                    "{dyn_val_rule}: \"<parameter=\" /[a-zA-Z0-9_.\\-\\/]+/ \">\" WS (%json {{\"type\":\"string\"}} WS \"</parameter>\" | unquoted_xml_param_end) WS\n"
                ));
            } else {
                definitions.push_str(&format!(
                    "{dyn_val_rule}: \"<parameter=\" /[a-zA-Z0-9_.\\-\\/]+/ \">\" WS %json {val_with_defs} WS \"</parameter>\" WS\n"
                ));
            }
            body.push_str(&format!(" ({dyn_val_rule})*"));
        }
    } else if num_props <= 6 {
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
    Ok(body)
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
    let mut extra_defs = String::new();
    let mut uses_unquoted_str = false;
    let mut uses_unquoted_nullable_str = false;
    let mut names = std::collections::HashSet::new();
    for (index, tool) in tools.iter().enumerate() {
        if tool
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function")
            != "function"
        {
            return Err("only function tools are supported".into());
        }
        let function = tool.get("function").unwrap_or(tool);
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
        let mut norm_schema = function
            .get("parameters")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or(empty);
        crate::constraints::normalize_schema_for_llguidance(&mut norm_schema);
        let schema = &norm_schema;
        let body = match format {
            ToolFormat::HermesJson => {
                let mut wrapper = json!({"type":"object","properties":{
                    "name":{"const":name},"arguments":schema
                },"required":["name","arguments"],"additionalProperties":false});
                if let Some(schema_obj) = schema.as_object() {
                    wrapper = attach_root_defs(None, &wrapper, schema_obj);
                    if let Some(Value::Object(props)) = schema_obj.get("properties") {
                        for prop_val in props.values() {
                            if let Some(prop_obj) = prop_val.as_object() {
                                wrapper = attach_root_defs(None, &wrapper, prop_obj);
                            }
                        }
                    }
                }
                format!("%json {wrapper}")
            }
            ToolFormat::QwenXml => build_qwen_xml_tool_body(
                index,
                name,
                schema,
                &mut definitions,
                &mut extra_defs,
                &mut uses_unquoted_str,
                &mut uses_unquoted_nullable_str,
            )?,
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
        grammar.push_str(&format!(
            "unquoted_xml_param_end[lazy]: UNQUOTED_XML_BODY \"</parameter>\"\nUNQUOTED_XML_BODY: (/[^ \\t\\r\\n\"](?s:.*[^ \\t\\r\\n])?/ & ~(\"null\" | {XML_DELIM_BAN_REGEX})) /[ \\t\\r\\n]*/\n"
        ));
    }
    if uses_unquoted_nullable_str {
        grammar
            .push_str("unquoted_xml_null_end[lazy]: \"null\" /[ \\t\\r\\n]*/ \"</parameter>\"\n");
    }
    grammar.push_str(&extra_defs);
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

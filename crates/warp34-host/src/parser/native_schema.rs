//! Request-owned native grammars for disambiguating XML parameter values.
use std::sync::Arc;

use serde_json::{Value, json};
use xgrammar::{Grammar, functor::grammar_optimizer, parser::EarleyParser};

pub(super) type ValueGrammar = Arc<Grammar>;

/// Keep references relative to the original tool schema, including recursive
/// nested objects. Traverse schema positions only, never const/enum data.
fn relocate_refs(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };
    let mut result = object.clone();
    if let Some(reference) = object.get("$ref").and_then(Value::as_str)
        && let Some(pointer) = reference.strip_prefix('#')
    {
        result.insert("$ref".into(), json!(format!("#/$defs/source{pointer}")));
    }
    for key in [
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ] {
        if let Some(entries) = object.get(key).and_then(Value::as_object) {
            result.insert(
                key.into(),
                Value::Object(
                    entries
                        .iter()
                        .map(|(name, value)| (name.clone(), relocate_refs(value)))
                        .collect(),
                ),
            );
        }
    }
    for key in [
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
    ] {
        if let Some(value) = object.get(key) {
            result.insert(
                key.into(),
                if let Some(items) = value.as_array() {
                    Value::Array(items.iter().map(relocate_refs).collect())
                } else {
                    relocate_refs(value)
                },
            );
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(items) = object.get(key).and_then(Value::as_array) {
            result.insert(
                key.into(),
                Value::Array(items.iter().map(relocate_refs).collect()),
            );
        }
    }
    Value::Object(result)
}

pub(super) fn compile(source: &Value, value: &Value) -> Result<ValueGrammar, String> {
    let schema = json!({"$ref":"#/$defs/value", "$defs":{
        "source":relocate_refs(source), "value":relocate_refs(value)
    }});
    // serde_json's preserve_order feature retains the generated object order.
    // Keep native object requirements rather than using relaxed any-order rules.
    let grammar = Grammar::from_json_schema(&schema.to_string(), true, None, None, true, None)
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(
        grammar_optimizer(&grammar).map_err(|e| e.to_string())?,
    ))
}

pub(super) fn accepts(value: &Value, grammar: Option<&ValueGrammar>) -> bool {
    let Some(grammar) = grammar else {
        return true;
    };
    let mut parser = EarleyParser::new(Arc::clone(grammar), None);
    value.to_string().bytes().all(|byte| parser.advance(byte)) && parser.is_completed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_references_keep_the_original_root() {
        let source = json!({"$defs":{"choice":{"anyOf":[{"const":"42"},{"type":"integer","minimum":1000}]}},
            "properties":{"value":{"$ref":"#/$defs/choice"}}});
        let grammar = compile(&source, &source["properties"]["value"]).unwrap();
        assert!(accepts(&json!("42"), Some(&grammar)));
        assert!(accepts(&json!(1000), Some(&grammar)));
        assert!(!accepts(&json!(42), Some(&grammar)));
    }

    #[test]
    fn schema_relocation_does_not_rewrite_literal_ref_data() {
        let schema = json!({"const":{"$ref":"#/literal"}});
        let grammar = compile(&schema, &schema).unwrap();
        assert!(accepts(&json!({"$ref":"#/literal"}), Some(&grammar)));
        assert!(!accepts(
            &json!({"$ref":"#/$defs/source/literal"}),
            Some(&grammar)
        ));
    }
}

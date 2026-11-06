//! Strict HGPTQ metadata. Keep patterns as strings so configuration equality is semantic.
use super::*;
use fancy_regex::{Regex, RegexBuilder};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedTensorRules {
    ignore: Vec<CompiledPattern>,
    groups: Vec<QuantRule>,
    transforms: Vec<TransformRule>,
}

#[derive(Clone, Debug)]
enum CompiledPattern {
    Literal(String),
    Regex { raw: String, re: Regex },
}

impl PartialEq for CompiledPattern {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Literal(a), Self::Literal(b)) => a == b,
            (Self::Regex { raw: a, .. }, Self::Regex { raw: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl Eq for CompiledPattern {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct QuantRule {
    targets: Vec<CompiledPattern>,
    bits: usize,
    group_size: usize,
    symmetric: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TransformRule {
    targets: Vec<CompiledPattern>,
    ignore: Vec<CompiledPattern>,
    head_dim: usize,
    offline: bool,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn regex(pattern: &str) -> io::Result<Regex> {
    if pattern.len() > 8192 {
        return Err(invalid("HGPTQ regex exceeds size limit"));
    }
    RegexBuilder::new(&format!(r"\A(?:{pattern})"))
        .backtrack_limit(100_000)
        .build()
        .map_err(|e| invalid(format!("invalid HGPTQ regex: {e}")))
}

fn compile_pattern(s: &str) -> io::Result<CompiledPattern> {
    if s.is_empty() || s.len() > 8192 {
        return Err(invalid("HGPTQ pattern must be a bounded nonempty string"));
    }
    if let Some(p) = s.strip_prefix("re:") {
        Ok(CompiledPattern::Regex {
            raw: s.to_owned(),
            re: regex(p)?,
        })
    } else {
        Ok(CompiledPattern::Literal(s.to_owned()))
    }
}

fn patterns(value: Option<&Json>, required: bool) -> io::Result<Vec<CompiledPattern>> {
    let Some(value) = value else {
        return if required {
            Err(invalid("missing HGPTQ targets"))
        } else {
            Ok(vec![])
        };
    };
    let array = value
        .as_array()
        .ok_or_else(|| invalid("HGPTQ targets/ignore must be arrays"))?;
    if required && array.is_empty() {
        return Err(invalid("empty HGPTQ targets"));
    }
    if array.len() > 64 {
        return Err(invalid("HGPTQ targets/ignore exceeds pattern count limit"));
    }
    array
        .iter()
        .map(|v| {
            let s = v
                .as_str()
                .ok_or_else(|| invalid("HGPTQ pattern must be a nonempty string"))?;
            compile_pattern(s)
        })
        .collect()
}

fn matches(patterns: &[CompiledPattern], name: &str) -> io::Result<bool> {
    if name.len() > 16384 {
        return Err(invalid("HGPTQ tensor name exceeds size limit"));
    }
    // Python re.match semantics: expressions are anchored at the start, not searched.
    for pattern in patterns {
        let matched = match pattern {
            CompiledPattern::Regex { re, .. } => re
                .find(name)
                .map_err(|e| invalid(format!("HGPTQ regex matching failed: {e}")))?
                .is_some_and(|m| m.start() == 0),
            CompiledPattern::Literal(lit) => {
                lit == "Linear"
                    || name == lit
                    || name
                        .strip_suffix(lit.as_str())
                        .is_some_and(|prefix| prefix.ends_with('.'))
            }
        };
        if matched {
            return Ok(true);
        }
    }
    Ok(false)
}

impl CompressedTensorRules {
    pub(super) fn parse(q: &Json) -> io::Result<Option<Self>> {
        let Some(config) = q.get("transform_config").filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        let obj = config
            .as_object()
            .ok_or_else(|| invalid("transform_config must be an object"))?;
        if obj.is_empty() {
            return Ok(None);
        }
        if obj.keys().any(|k| k != "config_groups") {
            return Err(invalid("unsupported transform_config metadata"));
        }
        let groups = config
            .get("config_groups")
            .and_then(Json::as_object)
            .ok_or_else(|| invalid("missing transform config_groups"))?;
        if groups.is_empty() {
            return Ok(None);
        }
        if groups.len() > 32 {
            return Err(invalid("transform config_groups exceeds group count limit"));
        }
        let ignore = patterns(q.get("ignore"), false)?;
        let mut transforms = vec![];
        for group in groups.values() {
            let obj = group
                .as_object()
                .ok_or_else(|| invalid("invalid transform group"))?;
            if obj.keys().any(|k| {
                !matches!(
                    k.as_str(),
                    "type" | "head_dim" | "randomize" | "requires_grad" | "precision" | "apply"
                )
            }) || group.get("type").and_then(Json::as_str) != Some("hadamard")
                || group.get("randomize").and_then(Json::as_bool) != Some(false)
                || group.get("requires_grad").is_some_and(|v| v != false)
                || group
                    .get("precision")
                    .is_some_and(|v| v != "torch.float32" && v != "float32")
            {
                return Err(invalid(
                    "unsupported non-deterministic HGPTQ transform metadata",
                ));
            }
            let head_dim = group
                .get("head_dim")
                .and_then(Json::as_u64)
                .ok_or_else(|| invalid("missing Hadamard head_dim"))?
                as usize;
            if ![16, 64].contains(&head_dim) {
                return Err(invalid("unsupported Hadamard head_dim"));
            }
            let apply = group
                .get("apply")
                .and_then(Json::as_array)
                .filter(|a| !a.is_empty())
                .ok_or_else(|| invalid("missing transform apply rules"))?;
            for rule in apply {
                let obj = rule
                    .as_object()
                    .ok_or_else(|| invalid("invalid transform apply rule"))?;
                if obj
                    .keys()
                    .any(|k| !matches!(k.as_str(), "targets" | "ignore" | "location" | "inverse"))
                {
                    return Err(invalid("unsupported transform apply metadata"));
                }
                let offline = match (
                    rule.get("location").and_then(Json::as_str),
                    rule.get("inverse").and_then(Json::as_bool),
                ) {
                    (Some("input"), Some(false)) => false,
                    (Some("weight_input"), Some(true)) => true,
                    _ => {
                        return Err(invalid(
                            "unsupported transform location/inverse (only input and offline inverse weight_input supported)",
                        ));
                    }
                };
                transforms.push(TransformRule {
                    targets: patterns(rule.get("targets"), true)?,
                    ignore: patterns(rule.get("ignore"), false)?,
                    head_dim,
                    offline,
                });
            }
        }
        let mut quant_groups = vec![];
        let quant_map = q
            .get("config_groups")
            .and_then(Json::as_object)
            .ok_or_else(|| invalid("missing quantization groups"))?;
        if quant_map.len() > 32 {
            return Err(invalid(
                "quantization config_groups exceeds group count limit",
            ));
        }
        for (key, group) in quant_map {
            let (bits, group_size, symmetric, _) =
                ModelConfig::parse_compressed_tensors_group(key, group)?;
            let w = &group["weights"];
            if bits != 4
                || !symmetric
                || ![0, 64].contains(&group_size)
                || w.get("dynamic").is_some_and(|v| v != false)
                || (group_size == 0
                    && (w.get("strategy").and_then(Json::as_str) != Some("channel")
                        || w.get("group_size").is_some_and(|v| !v.is_null())))
            {
                return Err(invalid(
                    "HGPTQ requires static symmetric W4 g64 or channel weights",
                ));
            }
            quant_groups.push(QuantRule {
                targets: patterns(group.get("targets"), true)?,
                bits,
                group_size,
                symmetric,
            });
        }
        Ok(Some(Self {
            ignore,
            groups: quant_groups,
            transforms,
        }))
    }

    fn resolve(&self, name: &str) -> io::Result<(Option<&QuantRule>, Option<usize>)> {
        if matches(&self.ignore, name)? {
            return Ok((None, None));
        }
        let mut selected: Option<&QuantRule> = None;
        for group in &self.groups {
            if matches(&group.targets, name)? {
                if selected.is_some_and(|old| {
                    (old.bits, old.group_size, old.symmetric)
                        != (group.bits, group.group_size, group.symmetric)
                }) {
                    return Err(invalid(format!(
                        "conflicting quantization rules for {name}"
                    )));
                }
                selected = Some(group);
            }
        }
        let mut input = None;
        let mut offline = None;
        for rule in &self.transforms {
            if matches(&rule.targets, name)? && !matches(&rule.ignore, name)? {
                if selected.is_none() {
                    return Err(invalid(format!(
                        "transform targets unquantized linear {name}"
                    )));
                }
                let slot = if rule.offline {
                    &mut offline
                } else {
                    &mut input
                };
                if slot.is_some() {
                    return Err(invalid(format!("overlapping transform rules for {name}")));
                }
                *slot = Some(rule.head_dim);
            }
        }
        if offline != input {
            return Err(invalid(format!(
                "inconsistent offline/input transform for {name}"
            )));
        }
        Ok((selected, input))
    }
}

impl QuantScheme {
    /// Resolve against the actual checkpoint tensor base name, without removing prefixes.
    pub fn for_linear(&self, name: &str) -> io::Result<Self> {
        if let Self::CompressedTensors {
            rules: Some(rules),
            ignore,
            ..
        } = self
        {
            let (group, _) = rules.resolve(name)?;
            return Ok(match group {
                Some(g) => Self::CompressedTensors {
                    num_bits: g.bits,
                    group_size: g.group_size,
                    symmetric: g.symmetric,
                    ignore: ignore.clone(),
                    rules: Some(rules.clone()),
                },
                None => Self::Unquantized,
            });
        }
        Ok(self.clone())
    }

    /// Online input transform only; offline weight_input transforms must never run again.
    pub fn input_hadamard(&self, name: &str) -> io::Result<Option<usize>> {
        match self {
            Self::CompressedTensors {
                rules: Some(rules), ..
            } => Ok(rules.resolve(name)?.1),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
#[path = "config_rules_tests.rs"]
mod tests;

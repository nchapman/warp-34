//! Native XGrammar compilation and rollback-safe sampler state.
//!
//! Reasoning is deliberately outside the grammar. Only the bounded delimiter
//! prefix is retained; matcher rollback counts are recorded per emitted token.
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use xgrammar::{CompiledGrammar, GrammarCompiler, GrammarMatcher, TokenizerInfo};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct StructuralTag {
    pub begin: String,
    pub end: String,
    pub schema: Value,
}

// Inspect syntax only: never retry invalid EBNF as a different language.
// Lark's native lexer recognizes double-quoted strings, /regex/, and # / // comments.
fn has_ebnf_rule_operator(grammar: &str) -> bool {
    let bytes = grammar.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'#' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'"' | b'/' => {
                let delimiter = bytes[i];
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i = (i + 2).min(bytes.len());
                    } else if bytes[i] == delimiter {
                        i += 1;
                        break;
                    } else {
                        i += 1;
                    }
                }
            }
            b':' if bytes[i..].starts_with(b"::=") => return true,
            _ => i += 1,
        }
    }
    false
}

pub struct ConstraintFactory {
    compiler: Arc<GrammarCompiler>,
    bytes: Arc<Vec<Vec<u8>>>,
    valid_mask: Arc<Vec<u32>>,
    vocab_size: usize,
    eos_ids: Vec<i32>,
}
impl ConstraintFactory {
    pub fn new(
        tokenizer: tokenizers::Tokenizer,
        vocab_size: usize,
        eos_ids: &[u32],
    ) -> Result<Self, String> {
        let vocab = tokenizer.get_vocab(true);
        if vocab_size > i32::MAX as usize
            || eos_ids.is_empty()
            || eos_ids
                .iter()
                .any(|id| *id as usize >= vocab_size || !vocab.values().any(|v| v == id))
        {
            return Err(
                "grammar EOS IDs must be nonempty and inside the tokenizer/model vocabulary".into(),
            );
        }
        if vocab.values().any(|id| *id as usize >= vocab_size) {
            return Err("model vocabulary is smaller than tokenizer vocabulary".into());
        }
        let eos_ids: Vec<i32> = eos_ids.iter().map(|id| *id as i32).collect();
        let info = TokenizerInfo::from_huggingface(&tokenizer, Some(vocab_size), Some(&eos_ids))?;
        let bytes = Arc::new(info.decoded_vocab().to_vec());
        let mut valid_mask = vec![0; vocab_size.div_ceil(32)];
        for id in vocab.values() {
            valid_mask[*id as usize / 32] |= 1 << (*id as usize % 32);
        }
        Ok(Self {
            compiler: Arc::new(GrammarCompiler::new(info, 8, true, 512 * 1024 * 1024)),
            bytes,
            valid_mask: Arc::new(valid_mask),
            vocab_size,
            eos_ids,
        })
    }
    /// Reuse compiled grammars while overriding request-specific termination IDs,
    /// like vLLM's all_stop_token_ids. Stop tokens remain masked until completion.
    pub fn with_stop_tokens(&self, stop_ids: &[i32]) -> Result<Self, String> {
        let mut eos_ids = self.eos_ids.clone();
        for &id in stop_ids {
            if id < 0
                || id as usize >= self.vocab_size
                || self.valid_mask[id as usize / 32] & (1 << (id as usize % 32)) == 0
            {
                return Err("stop token is not in the tokenizer vocabulary".into());
            }
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        Ok(Self {
            compiler: Arc::clone(&self.compiler),
            bytes: Arc::clone(&self.bytes),
            valid_mask: Arc::clone(&self.valid_mask),
            vocab_size: self.vocab_size,
            eos_ids,
        })
    }

    pub fn json_object(&self, thinking: bool) -> Result<TokenBitmaskConstraint, String> {
        self.json_schema(json!({"type":"object"}), thinking)
    }
    pub fn json_schema(
        &self,
        schema: Value,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        validate_json_schema(&schema)?;
        let g = self
            .compiler
            .compile_json_schema(&schema.to_string(), true, None, None, true, None)
            .map_err(|e| e.to_string())?;
        self.wrap(g, thinking, false)
    }
    pub fn regex(&self, regex: &str, thinking: bool) -> Result<TokenBitmaskConstraint, String> {
        if regex.contains('\0') {
            return Err("regex must not contain NUL".into());
        }
        self.wrap(
            self.compiler
                .compile_regex(regex)
                .map_err(|e| e.to_string())?,
            thinking,
            false,
        )
    }
    /// Accept native Lark (start rule) or EBNF (root rule).
    pub fn lark(&self, grammar: &str, thinking: bool) -> Result<TokenBitmaskConstraint, String> {
        let g = if has_ebnf_rule_operator(grammar) {
            self.compiler.compile_grammar_ebnf(grammar, "root")
        } else {
            self.compiler.compile_lark(grammar, &[])
        }
        .map_err(|e| e.to_string())?;
        self.wrap(g, thinking, false)
    }
    pub fn structural_tags(
        &self,
        tags: &[StructuralTag],
        triggers: &[String],
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        let mut native = Vec::new();
        for tag in tags {
            validate_json_schema(&tag.schema)?;
            native.push(json!({"type":"tag", "begin":tag.begin,"end":tag.end,"content":{"type":"json_schema","json_schema":tag.schema}}));
        }
        self.structural_tag(json!({"type":"structural_tag","format":{"type":"triggered_tags","triggers":triggers,"tags":native}}), thinking)
    }
    pub fn structural_tag(
        &self,
        value: Value,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        validate_tag_schemas(&value)?;
        self.wrap(
            self.compiler
                .compile_structural_tag(&value.to_string())
                .map_err(|e| e.to_string())?,
            thinking,
            false,
        )
    }
    pub fn tools(
        &self,
        tools: &Value,
        choice: &crate::tool_constraints::ToolChoice,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        self.tools_with_parallel(tools, choice, thinking, true)
    }

    pub fn tools_with_parallel(
        &self,
        tools: &Value,
        choice: &crate::tool_constraints::ToolChoice,
        thinking: bool,
        parallel: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        let Some(value) =
            crate::tool_constraints::tool_structural_tag_with_parallel(tools, choice, parallel)?
        else {
            let mut c = TokenBitmaskConstraint::unconstrained(self.vocab_size)
                .with_terminal_eos_ids(self.eos_ids.clone());
            c.mask = self.valid_mask.as_ref().clone();
            c.base_mask = Arc::clone(&self.valid_mask);
            return Ok(c);
        };
        validate_tag_schemas(&value)?;
        let g = self
            .compiler
            .compile_structural_tag(&value.to_string())
            .map_err(|e| e.to_string())?;
        self.wrap(g, thinking, true)
    }
    fn wrap(
        &self,
        g: CompiledGrammar,
        thinking: bool,
        implicit_tool: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        let mut c = TokenBitmaskConstraint::unconstrained(self.vocab_size);
        c.matcher = Some(
            GrammarMatcher::with_options(&g, Some(self.eos_ids.clone()), false, None)
                .map_err(|e| e.to_string())?,
        );
        c.bytes = Arc::clone(&self.bytes);
        c.base_mask = Arc::clone(&self.valid_mask);
        c.eos_ids = self.eos_ids.clone();
        c.gate = thinking.then(Vec::new);
        c.implicit_tool = implicit_tool;
        c.refresh()?;
        Ok(c)
    }
}

fn validate_tag_schemas(v: &Value) -> Result<(), String> {
    if let Some(schema) = v.get("json_schema") {
        validate_json_schema(schema)?;
    }
    match v {
        Value::Object(o) => {
            for (k, v) in o {
                if k != "json_schema" {
                    validate_tag_schemas(v)?;
                }
            }
        }
        Value::Array(a) => {
            for v in a {
                validate_tag_schemas(v)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Reject known silently weakened constraints, while leaving schema syntax and
/// supported extensions to the native compiler. Traverse schema positions only
/// (a property named `contains`, or an enum object, is not a keyword).
pub(crate) fn validate_json_schema(v: &Value) -> Result<(), String> {
    let Some(o) = v.as_object() else {
        return Ok(());
    };
    if let Some(reference) = o.get("$ref").and_then(Value::as_str)
        && reference != "#"
        && !reference.starts_with("#/")
    {
        return Err("only local JSON schema references are supported".into());
    }
    // Native 0.2.8 otherwise warns and lowers overlapping oneOf to anyOf.
    // Accept the common provably-disjoint primitive-type case; reject other
    // cases conservatively rather than silently weaken exclusivity.
    if let Some(branches) = o.get("oneOf").and_then(Value::as_array) {
        for (i, a) in branches.iter().enumerate() {
            for b in &branches[i + 1..] {
                let disjoint = match (
                    a.get("type").and_then(Value::as_str),
                    b.get("type").and_then(Value::as_str),
                ) {
                    (Some(a), Some(b)) => {
                        a != b
                            && !(["integer", "number"].contains(&a)
                                && ["integer", "number"].contains(&b))
                    }
                    _ => false,
                };
                if !disjoint {
                    return Err("oneOf branches must have provably disjoint primitive types; use anyOf for overlapping alternatives".into());
                }
            }
        }
    }
    for k in [
        "multipleOf",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "dependentRequired",
    ] {
        if o.contains_key(k) {
            return Err(format!("unsupported JSON schema keyword: {k}"));
        }
    }
    if (o.contains_key("pattern") || o.contains_key("format"))
        && (o.contains_key("minLength") || o.contains_key("maxLength"))
    {
        return Err("pattern/format combined with length bounds is unsupported".into());
    }
    if let Some(f) = o.get("format").and_then(Value::as_str)
        && ![
            "email",
            "date",
            "time",
            "date-time",
            "duration",
            "ipv4",
            "ipv6",
            "hostname",
            "uuid",
            "uri",
            "uri-reference",
            "uri-template",
            "json-pointer",
            "relative-json-pointer",
        ]
        .contains(&f)
    {
        return Err(format!("unsupported string format: {f}"));
    }
    if o.contains_key("propertyNames")
        && (o.contains_key("properties")
            || o.contains_key("patternProperties")
            || o.get("additionalProperties").is_some_and(Value::is_object)
            || o.get("unevaluatedProperties")
                .is_some_and(|v| v != &Value::Bool(true)))
    {
        return Err(
            "propertyNames combined with object property constraints is unsupported".into(),
        );
    }
    if let Some(p) = o.get("patternProperties").and_then(Value::as_object)
        && (p.len() > 1 || o.contains_key("properties"))
    {
        return Err("overlapping patternProperties constraints are unsupported".into());
    }
    for k in ["properties", "patternProperties", "$defs", "definitions"] {
        if let Some(m) = o.get(k).and_then(Value::as_object) {
            for v in m.values() {
                validate_json_schema(v)?;
            }
        }
    }
    for k in [
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
    ] {
        if let Some(v) = o.get(k) {
            if let Some(a) = v.as_array() {
                for v in a {
                    validate_json_schema(v)?;
                }
            } else {
                validate_json_schema(v)?;
            }
        }
    }
    for k in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        if let Some(a) = o.get(k).and_then(Value::as_array) {
            for v in a {
                validate_json_schema(v)?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Step {
    gate: Option<Vec<u8>>,
    matcher_step: bool,
    terminal: bool,
}
type ReasoningMasks = Arc<Mutex<HashMap<Vec<u8>, Vec<u32>>>>;

#[derive(Clone)]
pub struct TokenBitmaskConstraint {
    vocab_size: usize,
    matcher: Option<GrammarMatcher>,
    bytes: Arc<Vec<Vec<u8>>>,
    base_mask: Arc<Vec<u32>>,
    eos_ids: Vec<i32>,
    gate: Option<Vec<u8>>,
    implicit_tool: bool,
    reasoning_masks: ReasoningMasks,
    terminal: bool,
    pending_token: Option<i32>,
    mask: Vec<u32>,
    history: Vec<Step>,
}
impl std::fmt::Debug for TokenBitmaskConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBitmaskConstraint")
            .field("vocab_size", &self.vocab_size)
            .field("grammar", &self.matcher.is_some())
            .field("gate", &self.gate)
            .field("terminal", &self.terminal)
            .field("pending_token", &self.pending_token)
            .field("committed", &self.committed_len())
            .field("mask", &self.mask)
            .finish()
    }
}
impl TokenBitmaskConstraint {
    pub fn unconstrained(vocab_size: usize) -> Self {
        let mut mask = vec![u32::MAX; vocab_size.div_ceil(32)];
        if !vocab_size.is_multiple_of(32) {
            *mask.last_mut().unwrap() = (1 << (vocab_size % 32)) - 1;
        }
        Self {
            vocab_size,
            matcher: None,
            bytes: Arc::new(Vec::new()),
            base_mask: Arc::new(mask.clone()),
            eos_ids: Vec::new(),
            gate: None,
            implicit_tool: false,
            reasoning_masks: Arc::new(Mutex::new(HashMap::new())),
            terminal: false,
            pending_token: None,
            mask,
            history: Vec::new(),
        }
    }
    pub fn from_allowed_tokens(vocab_size: usize, allowed: Vec<i32>) -> Self {
        let mut c = Self::unconstrained(vocab_size);
        c.mask.fill(0);
        for id in allowed {
            if id >= 0 && (id as usize) < vocab_size {
                c.mask[id as usize / 32] |= 1 << (id as usize % 32);
            }
        }
        c.base_mask = Arc::new(c.mask.clone());
        c
    }
    pub fn with_terminal_eos_ids(mut self, ids: Vec<i32>) -> Self {
        self.eos_ids = ids;
        self
    }
    pub fn has_pending(&self) -> bool {
        self.pending_token.is_some()
    }
    pub fn is_terminal_eos(&self) -> bool {
        self.terminal
            || self
                .pending_token
                .is_some_and(|id| self.eos_ids.contains(&id))
    }
    pub fn committed_len(&self) -> usize {
        self.history.len() + usize::from(self.has_pending())
    }
    pub fn current_bitmask(&self) -> Vec<u32> {
        self.mask_words().to_vec()
    }
    pub fn mask_words(&self) -> &[u32] {
        assert!(!self.has_pending(), "materialize pending mask first");
        &self.mask
    }
    pub fn is_token_allowed(&self, id: i32) -> bool {
        id >= 0
            && (id as usize) < self.vocab_size
            && self.mask_words()[id as usize / 32] & (1 << (id as usize % 32)) != 0
    }
    pub fn advance_deferred(&mut self, id: i32) -> Result<(), String> {
        self.materialize_pending()?;
        if !self.is_token_allowed(id) {
            return Err(format!("token {id} violates grammar"));
        }
        self.pending_token = Some(id);
        Ok(())
    }
    pub fn advance(&mut self, id: i32) -> Result<(), String> {
        self.advance_deferred(id)?;
        if let Err(e) = self.materialize_pending() {
            self.pending_token = None;
            return Err(e);
        }
        Ok(())
    }
    pub fn materialize_pending(&mut self) -> Result<(), String> {
        let Some(id) = self.pending_token else {
            return Ok(());
        };
        let step = Step {
            gate: self.gate.clone(),
            matcher_step: false,
            terminal: self.terminal,
        };
        let mut used = false;
        if let Some(prefix) = &self.gate {
            if self.eos_ids.contains(&id) {
                self.terminal = true;
            } else {
                let (next, answer) =
                    gate_token(prefix, &self.bytes[id as usize], self.implicit_tool);
                if let Some(answer) = answer
                    && !answer.is_empty()
                {
                    if !self.matcher.as_mut().unwrap().accept_bytes(&answer) {
                        return Err("reasoning boundary suffix violates grammar".into());
                    }
                    used = true;
                }
                self.gate = next;
            }
        } else if let Some(m) = &mut self.matcher {
            if !m.accept_token(id) {
                return Err(format!("native matcher rejected token {id}"));
            }
            used = true;
            self.terminal = m.is_terminated();
        } else {
            self.terminal = self.eos_ids.contains(&id);
        }
        if let Err(e) = self.refresh() {
            if used {
                self.matcher
                    .as_mut()
                    .unwrap()
                    .rollback(1)
                    .map_err(|e| e.to_string())?;
            }
            self.gate = step.gate;
            self.terminal = step.terminal;
            return Err(e);
        }
        self.history.push(Step {
            matcher_step: used,
            ..step
        });
        self.pending_token = None;
        Ok(())
    }
    fn refresh(&mut self) -> Result<(), String> {
        let mut mask = self.base_mask.as_ref().clone();
        if self.terminal {
            mask.fill(0);
        } else if let Some(prefix) = &self.gate {
            // The grammar has not advanced while this gate is active. Each finite
            // delimiter-prefix state therefore has one immutable mask, reusable
            // across reasoning tokens and speculative forks/rollbacks.
            let mut cached = self
                .reasoning_masks
                .lock()
                .map_err(|_| "reasoning mask cache poisoned")?;
            if let Some(saved) = cached.get(prefix) {
                self.mask.clone_from(saved);
                return Ok(());
            }
            // Only tokens that cross a boundary need a matcher fork. Ordinary
            // reasoning tokens never clone the growing grammar parser.
            for (id, bytes) in self.bytes.iter().enumerate() {
                if id >= self.vocab_size || self.eos_ids.contains(&(id as i32)) {
                    continue;
                }
                let (_, answer) = gate_token(prefix, bytes, self.implicit_tool);
                if let Some(answer) = answer
                    && !answer.is_empty()
                    && !self.matcher.as_ref().unwrap().fork().accept_bytes(&answer)
                {
                    mask[id / 32] &= !(1 << (id % 32));
                }
            }
            cached.insert(prefix.clone(), mask.clone());
        } else if let Some(m) = &mut self.matcher {
            let mut native = vec![0i32; mask.len()];
            m.fill_next_token_bitmask(&mut native, &[1, mask.len() as i64], 0)
                .map_err(|e| e.to_string())?;
            for (word, native) in mask.iter_mut().zip(native) {
                *word &= native as u32;
            }
        }
        self.mask = mask;
        Ok(())
    }
    pub fn rollback(&mut self, count: usize) -> Result<(), String> {
        if count > self.committed_len() {
            return Err("grammar rollback exceeds committed tokens".into());
        }
        let mut remaining = count;
        if remaining > 0 && self.pending_token.take().is_some() {
            remaining -= 1;
        }
        if remaining == 0 {
            return Ok(());
        }
        let start = self.history.len() - remaining;
        let native = self.history[start..]
            .iter()
            .filter(|s| s.matcher_step)
            .count();
        if native > 0 {
            self.matcher
                .as_mut()
                .unwrap()
                .rollback(i32::try_from(native).map_err(|_| "rollback too large")?)
                .map_err(|e| e.to_string())?;
        }
        self.gate = self.history[start].gate.clone();
        self.terminal = self.history[start].terminal;
        self.history.truncate(start);
        self.refresh()
    }
}

/// Return the next partial delimiter, or the exact grammar bytes after activation.
fn gate_token(
    prefix: &[u8],
    bytes: &[u8],
    implicit_tool: bool,
) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut input = prefix.to_vec();
    input.extend_from_slice(bytes);
    let markers: &[&[u8]] = if implicit_tool {
        &[b"</think>", b"<tool_call>"]
    } else {
        &[b"</think>"]
    };
    let hit = markers
        .iter()
        .filter_map(|marker| {
            input
                .windows(marker.len())
                .position(|w| w == *marker)
                .map(|pos| (pos, *marker))
        })
        .min_by_key(|(pos, _)| *pos);
    if let Some((pos, marker)) = hit {
        let start = if marker == b"<tool_call>" {
            pos
        } else {
            pos + marker.len()
        };
        return (None, Some(input[start..].to_vec()));
    }
    let keep = (1..=input.len().min(10))
        .rev()
        .find(|n| {
            markers
                .iter()
                .any(|m| m.starts_with(&input[input.len() - n..]))
        })
        .unwrap_or(0);
    (Some(input[input.len() - keep..].to_vec()), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn factory() -> ConstraintFactory {
        let mut bytes: Vec<Vec<u8>> = (0..=255).map(|b| vec![b]).collect();
        bytes.extend([
            b"</s>".to_vec(),
            b"</think>a".to_vec(),
            b"</think>x".to_vec(),
            b"<tool_call>".to_vec(),
        ]);
        let info = TokenizerInfo::new_from_bytes(
            &bytes,
            xgrammar::VocabType::Raw,
            Some(263),
            Some(vec![256]),
            false,
        )
        .unwrap();
        let mut mask = vec![0; 263usize.div_ceil(32)];
        for id in 0..bytes.len() {
            mask[id / 32] |= 1 << (id % 32);
        }
        ConstraintFactory {
            compiler: Arc::new(GrammarCompiler::new(info, 1, true, 1024 * 1024)),
            bytes: Arc::new(bytes),
            valid_mask: Arc::new(mask),
            vocab_size: 263,
            eos_ids: vec![256],
        }
    }
    #[test]
    fn grammar_operators_in_literals_and_comments() {
        let f = factory();
        for (grammar, output) in [
            (r#"start: "::=""#, "::="),
            (r#"start: /::=/"#, "::="),
            ("// ::= ignored\nstart: \"ok\"", "ok"),
            ("# ::= ignored\nstart: \"ok\"", "ok"),
            (r#"start: "\"::=""#, "\"::="),
            (r#"start: /\/::=/"#, "/::="),
            (r##"root ::= "#""##, "#"),
        ] {
            let mut constraint = f
                .lark(grammar, false)
                .unwrap_or_else(|e| panic!("{grammar}: {e}"));
            assert!(!constraint.is_token_allowed(b'x' as i32));
            for byte in output.bytes() {
                constraint.advance(byte as i32).unwrap();
            }
            assert!(constraint.is_token_allowed(256), "{grammar}");
        }
        assert!(f.lark(r#"root ::= "ok""#, false).is_ok());
        assert!(f.lark("root ::= (", false).is_err());
    }
    #[test]
    fn request_stop_tokens_are_masked_until_grammar_completion() {
        let f = factory();
        let request = f.with_stop_tokens(&[b'x' as i32]).unwrap();
        let mut c = request.regex("ab", false).unwrap();
        assert!(!c.is_token_allowed(b'x' as i32));
        c.advance(b'a' as i32).unwrap();
        assert!(!c.is_token_allowed(b'x' as i32));
        c.advance(b'b' as i32).unwrap();
        assert!(c.is_token_allowed(b'x' as i32));
        c.advance(b'x' as i32).unwrap();
        assert!(c.is_terminal_eos());
        c.rollback(1).unwrap();
        assert!(c.is_token_allowed(256));
        assert!(f.with_stop_tokens(&[262]).is_err());
        assert!(!f.regex("ab", false).unwrap().is_token_allowed(b'x' as i32));
    }

    #[test]
    fn native_eos_deferred_clone_rollback() {
        let f = factory();
        let mut c = f.regex("ab", false).unwrap();
        assert!(c.is_token_allowed(97));
        assert!(!c.is_token_allowed(256));
        c.advance_deferred(97).unwrap();
        let mut branch = c.clone();
        branch.materialize_pending().unwrap();
        branch.advance(98).unwrap();
        assert!(branch.is_token_allowed(256));
        branch.advance(256).unwrap();
        assert!(branch.is_terminal_eos());
        assert!(branch.mask_words().iter().all(|w| *w == 0));
        branch.rollback(1).unwrap();
        assert!(branch.is_token_allowed(256));
        branch.rollback(2).unwrap();
        assert_eq!(branch.committed_len(), 0);
        assert!(branch.is_token_allowed(97));
        c.rollback(1).unwrap();
        assert_eq!(c.current_bitmask(), branch.current_bitmask());
        assert!(c.advance(98).is_err());
        assert_eq!(c.committed_len(), 0);
    }
    #[test]
    fn gate_split_transition_and_suffix_are_reversible() {
        let f = factory();
        let mut c = f.regex("ab", true).unwrap();
        assert!(c.is_token_allowed(257));
        assert!(!c.is_token_allowed(258));
        assert!(!c.is_token_allowed(260));
        c.advance(257).unwrap();
        assert!(c.is_token_allowed(98));
        c.rollback(1).unwrap();
        assert!(c.is_token_allowed(257));
        for b in b"reason</think>" {
            c.advance(*b as i32).unwrap();
        }
        assert!(c.is_token_allowed(97));
        assert!(!c.is_token_allowed(120));
        c.rollback(1).unwrap();
        assert!(c.is_token_allowed(120));
        c.advance(b'>' as i32).unwrap();
        c.advance(97).unwrap();
        c.advance(98).unwrap();
        c.rollback(3).unwrap();
        assert!(c.is_token_allowed(120));
    }
    #[test]
    fn implicit_tool_boundary_preserves_marker() {
        let f = factory();
        let g = f.compiler.compile_regex("<tool_call>a").unwrap();
        let mut c = f.wrap(g, true, true).unwrap();
        c.advance(259).unwrap();
        assert!(c.is_token_allowed(97));
        c.rollback(1).unwrap();
        assert!(c.is_token_allowed(120));
    }
    #[test]
    fn static_masks_and_validation() {
        let mut c = TokenBitmaskConstraint::from_allowed_tokens(35, vec![1, 34, 35, -1])
            .with_terminal_eos_ids(vec![34]);
        c.advance(1).unwrap();
        c.advance(34).unwrap();
        c.rollback(1).unwrap();
        assert!(c.is_token_allowed(34));
        assert_eq!(c.mask_words()[1], 4);
        assert!(
            validate_json_schema(&json!({"type":"string","pattern":"a+","maxLength":3})).is_err()
        );
        assert!(validate_json_schema(&json!({"type":"object","properties":{"contains":{"type":"string"}},"enum":[{"not":true}]})).is_ok());
        for schema in [
            json!({"type":"integer","multipleOf":3}),
            json!({"type":"integer","multipleOf":3,"minimum":0,"maximum":9}),
            json!({"type":"number","multipleOf":0.5}),
            json!({"multipleOf":null}),
            json!({"properties":{"value":{"multipleOf":2}}}),
        ] {
            assert_eq!(
                validate_json_schema(&schema).unwrap_err(),
                "unsupported JSON schema keyword: multipleOf"
            );
        }
        assert!(
            validate_json_schema(
                &json!({"properties":{"multipleOf":{"type":"integer"}},"enum":[{"multipleOf":3}]})
            )
            .is_ok()
        );
    }
}

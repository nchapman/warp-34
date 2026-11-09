//! Tokenizer-aware structured decoding with cached packed-u32 masks.
//!
//! llguidance owns byte/grammar state, including the reasoning-to-answer
//! transition. Speculative clones never reconstruct state from decoded UTF-8.

use llguidance::{
    Matcher, ParserFactory,
    api::{GrammarWithLexer, TopLevelGrammar},
};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct StructuralTag {
    pub begin: String,
    pub end: String,
    pub schema: serde_json::Value,
}

pub struct ConstraintFactory {
    factory: ParserFactory,
    vocab_size: usize,
    eos_ids: Vec<i32>,
}

impl ConstraintFactory {
    pub fn new(
        tokenizer: tokenizers::Tokenizer,
        vocab_size: usize,
        eos_ids: &[u32],
    ) -> Result<Self, String> {
        let mut tokenizer = ByteTokenizer::from_tokenizer(tokenizer).map_err(|e| e.to_string())?;
        if eos_ids.is_empty()
            || eos_ids
                .iter()
                .any(|&id| id >= tokenizer.tokrx_info().vocab_size)
        {
            return Err(
                "grammar EOS IDs must be nonempty and inside the tokenizer vocabulary".into(),
            );
        }
        tokenizer.set_eos_tokens(eos_ids);
        // The pinned adapter treats every <...> added token as special, even
        // when HF marks it ordinary. Restore HF's classification for Qwen's
        // reasoning/tool delimiters so byte grammars can consume their native IDs.
        let ordinary_added = tokenizer
            .hf_tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, token)| !token.special)
            .map(|(id, token)| (id, token.content.into_bytes()))
            .collect::<Vec<_>>();
        let mut env =
            ByteTokenizerEnv::new(tokenizer, Some(vocab_size)).map_err(|e| e.to_string())?;
        let mut bytes = (0..env.tok_trie.info().vocab_size)
            .map(|id| env.tok_trie.token(id).to_vec())
            .collect::<Vec<_>>();
        for (id, content) in ordinary_added {
            if !eos_ids.contains(&id) {
                bytes[id as usize] = content;
            }
        }
        env.tok_trie = llguidance::toktrie::TokTrie::from(env.tok_trie.info(), &bytes)
            .with_eos_tokens(eos_ids);
        let env = env.to_env();
        let mut factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
        factory.quiet();
        Ok(Self {
            factory,
            vocab_size,
            eos_ids: eos_ids.iter().map(|&id| id as i32).collect(),
        })
    }

    pub fn json_object(&self, thinking: bool) -> Result<TokenBitmaskConstraint, String> {
        self.json_schema(serde_json::json!({"type":"object"}), thinking)
    }

    pub fn json_schema(
        &self,
        mut schema: serde_json::Value,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        validate_strict_schema_options(&schema)?;
        normalize_schema_for_llguidance(&mut schema);
        self.compile(TopLevelGrammar::from_json_schema(schema), thinking)
    }

    pub fn regex(&self, regex: &str, thinking: bool) -> Result<TokenBitmaskConstraint, String> {
        self.compile(TopLevelGrammar::from_regex(regex), thinking)
    }

    pub fn choice(
        &self,
        choices: &[String],
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        if choices.is_empty() || choices.iter().any(String::is_empty) {
            return Err("choice requires a non-empty array of non-empty strings".into());
        }
        let mut lits = Vec::with_capacity(choices.len());
        for c in choices {
            let lit = serde_json::to_string(c).map_err(|e| e.to_string())?;
            if !lits.contains(&lit) {
                lits.push(lit);
            }
        }
        let prefix = if thinking { "/[ \\t\\r\\n]*/ " } else { "" };
        self.lark(
            &format!("start: {prefix}({})\n", lits.join(" | ")),
            thinking,
        )
    }

    /// Lark CFG syntax, not GBNF. Token limits and hidden-stop backtracking are
    /// deliberately excluded because this parser must support arbitrary rollback.
    pub fn lark(&self, grammar: &str, thinking: bool) -> Result<TokenBitmaskConstraint, String> {
        self.compile(TopLevelGrammar::from_lark(grammar.into()), thinking)
    }

    /// llguidance StructTag's lazy-trigger construction, with ordinary byte
    /// terminals (Qwen tool delimiters are added tokens, not special tokens).
    pub fn structural_tags(
        &self,
        tags: &[StructuralTag],
        triggers: &[String],
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        if tags.is_empty() || triggers.is_empty() || triggers.iter().any(String::is_empty) {
            return Err("structural tags require nonempty structures and triggers".into());
        }
        if triggers.iter().enumerate().any(|(i, t)| {
            triggers
                .iter()
                .enumerate()
                .any(|(j, u)| i != j && t.starts_with(u))
        }) {
            return Err("structural triggers must not overlap".into());
        }
        let alternatives = (0..tags.len())
            .map(|i| format!("tag_{i}"))
            .collect::<Vec<_>>()
            .join(" | ");
        let mut lark =
            format!("start: ({alternatives})* tag_end\ntag_end: TAG_TEXT\nTAG_TEXT: /(?s:.*)/\n");
        for (index, tag) in tags.iter().enumerate() {
            validate_strict_schema_options(&tag.schema)?;
            let mut normalized_schema = tag.schema.clone();
            normalize_schema_for_llguidance(&mut normalized_schema);
            let trigger = triggers
                .iter()
                .find(|trigger| tag.begin.starts_with(trigger.as_str()))
                .ok_or("each structural begin must start with a declared trigger")?;
            let suffix = &tag.begin[trigger.len()..];
            lark.push_str(&format!(
                "tag_{index}_trig[lazy]: TAG_TEXT {}\ntag_{index}: tag_{index}_trig {} %json {} {}\n",
                serde_json::to_string(trigger).unwrap(),
                serde_json::to_string(suffix).unwrap(),
                normalized_schema,
                serde_json::to_string(&tag.end).unwrap(),
            ));
        }
        if triggers
            .iter()
            .any(|trigger| !tags.iter().any(|tag| tag.begin.starts_with(trigger)))
        {
            return Err("every structural trigger must have at least one structure".into());
        }
        self.lark(&lark, thinking)
    }

    pub fn tools(
        &self,
        tools: &serde_json::Value,
        choice: &crate::tool_constraints::ToolChoice,
        format: crate::tool_constraints::ToolFormat,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        self.tools_with_options(tools, choice, format, true, thinking)
    }

    pub fn tools_with_options(
        &self,
        tools: &serde_json::Value,
        choice: &crate::tool_constraints::ToolChoice,
        format: crate::tool_constraints::ToolFormat,
        parallel_tool_calls: bool,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        if *choice != crate::tool_constraints::ToolChoice::None {
            for tool in tools.as_array().ok_or("tools must be an array")? {
                let func = tool.get("function").unwrap_or(tool);
                if let crate::tool_constraints::ToolChoice::Named(name) = choice
                    && func.get("name").and_then(serde_json::Value::as_str) != Some(name.as_str())
                {
                    continue;
                }
                if let Some(schema) = func.get("parameters").filter(|v| !v.is_null()) {
                    // Validate complete schemas before XML lowering, including
                    // keywords (such as additionalProperties) not emitted there.
                    let mut check_schema = schema.clone();
                    if let Some(root_obj) = check_schema.as_object_mut() {
                        if !root_obj.contains_key("type")
                            && !root_obj.contains_key("$ref")
                            && !root_obj.contains_key("anyOf")
                            && !root_obj.contains_key("oneOf")
                            && !root_obj.contains_key("allOf")
                        {
                            root_obj
                                .insert("type".into(), serde_json::Value::String("object".into()));
                        }
                        if let Some(props) = root_obj
                            .get("properties")
                            .and_then(serde_json::Value::as_object)
                            .cloned()
                        {
                            for def_key in ["$defs", "definitions"] {
                                for prop_val in props.values() {
                                    let Some(prop_defs) = prop_val
                                        .get(def_key)
                                        .and_then(serde_json::Value::as_object)
                                    else {
                                        continue;
                                    };
                                    let root_defs = root_obj
                                        .entry(def_key.to_string())
                                        .or_insert_with(|| serde_json::json!({}));
                                    if let Some(root_defs_obj) = root_defs.as_object_mut() {
                                        for (k, v) in prop_defs {
                                            root_defs_obj
                                                .entry(k.clone())
                                                .or_insert_with(|| v.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    self.json_schema(check_schema, false)?;
                }
            }
        }
        let grammar = crate::tool_constraints::tool_grammar_with_options(
            tools,
            choice,
            format,
            parallel_tool_calls,
        )?;
        self.lark(
            &grammar,
            thinking && *choice != crate::tool_constraints::ToolChoice::None,
        )
    }

    fn compile(
        &self,
        mut grammar: TopLevelGrammar,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        if thinking {
            grammar.grammars[0].name = Some("answer".into());
            grammar.grammars.insert(
                0,
                GrammarWithLexer::from_lark(
                    "start: reasoning @answer\nreasoning[suffix=\"</think>\"]: REASON_TEXT\nREASON_TEXT: /(?s:.*)/ & ~/(?s:.*)<tool_call>(?s:.*)/".into(),
                ),
            );
        }
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(|e| e.to_string())?;
        parser
            .parser
            .grammar()
            .lexer_spec()
            .check_rollback()
            .map_err(|e| e.to_string())?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(error) = matcher.get_error() {
            return Err(error);
        }
        // Never silently weaken an unsupported strict schema.
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(format!("unsupported grammar: {}", warnings.join("; ")));
        }
        let mask = matcher
            .compute_mask_or_eos()
            .map_err(|e| e.to_string())?
            .as_slice()
            .to_vec();
        Ok(TokenBitmaskConstraint {
            vocab_size: self.vocab_size,
            matcher: Some(matcher),
            eos_ids: self.eos_ids.clone(),
            terminal_eos: false,
            pending_token: None,
            pre_eos_mask: None,
            mask,
            committed: 0,
        })
    }
}

fn normalize_type_name(t: &str) -> &str {
    match t {
        "int" | "uint" | "int32" | "int64" | "long" => "integer",
        "str" | "text" | "varchar" | "char" => "string",
        "bool" => "boolean",
        "float" | "double" | "decimal" => "number",
        "dict" | "map" => "object",
        "list" | "set" | "tuple" => "array",
        other => other,
    }
}

pub(crate) fn is_llguidance_supported_format(fmt: &str) -> bool {
    matches!(
        fmt,
        "date-time"
            | "time"
            | "date"
            | "duration"
            | "email"
            | "hostname"
            | "ipv4"
            | "ipv6"
            | "uuid"
            | "uri"
    )
}

fn normalize_type_and_format(obj: &mut serde_json::Map<String, serde_json::Value>) {
    match obj.get_mut("type") {
        Some(serde_json::Value::String(s)) => {
            let norm = normalize_type_name(s);
            if norm != s {
                *s = norm.to_string();
            }
        }
        Some(serde_json::Value::Array(arr)) => {
            for item in arr {
                if let serde_json::Value::String(s) = item {
                    let norm = normalize_type_name(s);
                    if norm != s {
                        *s = norm.to_string();
                    }
                }
            }
        }
        _ => {}
    }

    let type_allows_string = match obj.get("type") {
        None => true,
        Some(serde_json::Value::String(s)) => s == "string",
        Some(serde_json::Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("string")),
        _ => false,
    };
    if let Some(fmt_val) = obj.get("format") {
        let keep =
            type_allows_string && fmt_val.as_str().is_some_and(is_llguidance_supported_format);
        if !keep {
            obj.remove("format");
        }
    }
    obj.remove("uniqueItems");
}

fn lower_property_names_and_unevaluated(obj: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(uneval_props) = obj.remove("unevaluatedProperties")
        && !obj.contains_key("additionalProperties")
        && !obj.contains_key("allOf")
        && !obj.contains_key("anyOf")
        && !obj.contains_key("oneOf")
        && !obj.contains_key("$ref")
    {
        obj.insert("additionalProperties".into(), uneval_props);
    }
    if let Some(uneval_items) = obj.remove("unevaluatedItems")
        && !obj.contains_key("additionalItems")
    {
        obj.insert("additionalItems".into(), uneval_items);
    }

    let Some(prop_names) = obj.remove("propertyNames") else {
        return;
    };
    let Some(pn_obj) = prop_names.as_object() else {
        return;
    };
    if let Some(serde_json::Value::Array(enum_keys)) = pn_obj.get("enum")
        && !obj.contains_key("properties")
        && enum_keys.iter().all(serde_json::Value::is_string)
    {
        let val_schema = obj
            .remove("additionalProperties")
            .filter(serde_json::Value::is_object)
            .unwrap_or_else(|| serde_json::json!({}));
        let mut props = serde_json::Map::new();
        for k in enum_keys {
            if let Some(ks) = k.as_str() {
                props.insert(ks.to_string(), val_schema.clone());
            }
        }
        obj.insert("properties".into(), serde_json::Value::Object(props));
        obj.insert(
            "additionalProperties".into(),
            serde_json::Value::Bool(false),
        );
    } else if let Some(serde_json::Value::String(pat)) = pn_obj.get("pattern")
        && !obj.contains_key("patternProperties")
    {
        let val_schema = obj
            .remove("additionalProperties")
            .filter(serde_json::Value::is_object)
            .unwrap_or_else(|| serde_json::json!({}));
        let mut pat_props = serde_json::Map::new();
        pat_props.insert(pat.clone(), val_schema);
        obj.insert(
            "patternProperties".into(),
            serde_json::Value::Object(pat_props),
        );
        obj.insert(
            "additionalProperties".into(),
            serde_json::Value::Bool(false),
        );
    }
}

fn lower_openapi_nullable(obj: &mut serde_json::Map<String, serde_json::Value>) {
    if obj.remove("nullable") != Some(serde_json::Value::Bool(true)) {
        return;
    }
    if let Some(serde_json::Value::Array(arr)) = obj.get_mut("enum")
        && !arr.contains(&serde_json::Value::Null)
    {
        arr.push(serde_json::Value::Null);
    }
    if let Some(t) = obj.get_mut("type") {
        match t {
            serde_json::Value::String(s) if s != "null" => {
                *t = serde_json::json!([s.clone(), "null"]);
            }
            serde_json::Value::Array(arr) if !arr.iter().any(|v| v.as_str() == Some("null")) => {
                arr.push(serde_json::Value::String("null".into()));
            }
            _ => {}
        }
    } else if let Some(serde_json::Value::Array(arr)) = obj.get_mut("anyOf") {
        if !arr
            .iter()
            .any(|v| v.get("type").and_then(serde_json::Value::as_str) == Some("null"))
        {
            arr.push(serde_json::json!({"type": "null"}));
        }
    } else if obj.contains_key("$ref")
        || obj.contains_key("properties")
        || obj.contains_key("additionalProperties")
        || obj.contains_key("oneOf")
        || obj.contains_key("allOf")
    {
        let mut inner = serde_json::Map::new();
        let keys: Vec<String> = obj
            .keys()
            .filter(|k| !matches!(k.as_str(), "$defs" | "definitions" | "$schema"))
            .cloned()
            .collect();
        for k in keys {
            if let Some(v) = obj.remove(&k) {
                inner.insert(k, v);
            }
        }
        obj.insert(
            "anyOf".into(),
            serde_json::json!([serde_json::Value::Object(inner), {"type": "null"}]),
        );
    }
}

pub(crate) fn normalize_schema_for_llguidance(value: &mut serde_json::Value) {
    let Some(obj) = value.as_object_mut() else {
        if let serde_json::Value::Array(items) = value {
            for item in items {
                normalize_schema_for_llguidance(item);
            }
        }
        return;
    };

    normalize_type_and_format(obj);
    lower_property_names_and_unevaluated(obj);
    lower_openapi_nullable(obj);

    for (k, v) in obj.iter_mut() {
        if matches!(
            k.as_str(),
            "properties" | "$defs" | "definitions" | "patternProperties" | "dependentSchemas"
        ) {
            if let serde_json::Value::Object(map) = v {
                for sub_schema in map.values_mut() {
                    normalize_schema_for_llguidance(sub_schema);
                }
            }
        } else if !matches!(
            k.as_str(),
            "enum" | "const" | "default" | "examples" | "example"
        ) {
            normalize_schema_for_llguidance(v);
        }
    }
}

fn validate_strict_schema_options(value: &serde_json::Value) -> Result<(), String> {
    match value {
        serde_json::Value::Object(object) => {
            if object
                .get("x-guidance")
                .and_then(|v| v.get("lenient"))
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                return Err("lenient schemas are not allowed for strict constrained output".into());
            }
            for value in object.values() {
                validate_strict_schema_options(value)?;
            }
        }
        serde_json::Value::Array(items) => {
            for value in items {
                validate_strict_schema_options(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone)]
pub struct TokenBitmaskConstraint {
    vocab_size: usize,
    matcher: Option<Matcher>,
    eos_ids: Vec<i32>,
    terminal_eos: bool,
    pending_token: Option<i32>,
    pre_eos_mask: Option<Vec<u32>>,
    mask: Vec<u32>,
    committed: usize,
}

impl std::fmt::Debug for TokenBitmaskConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBitmaskConstraint")
            .field("vocab_size", &self.vocab_size)
            .field("grammar", &self.matcher.is_some())
            .field("pending_token", &self.pending_token)
            .field("committed", &self.committed)
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
            eos_ids: Vec::new(),
            terminal_eos: false,
            pending_token: None,
            pre_eos_mask: None,
            mask,
            committed: 0,
        }
    }

    pub fn from_allowed_tokens(vocab_size: usize, allowed: Vec<i32>) -> Self {
        let mut mask = vec![0; vocab_size.div_ceil(32)];
        for id in allowed {
            if id >= 0 && (id as usize) < vocab_size {
                mask[id as usize / 32] |= 1 << (id as usize % 32);
            }
        }
        Self {
            vocab_size,
            matcher: None,
            eos_ids: Vec::new(),
            terminal_eos: false,
            pending_token: None,
            pre_eos_mask: None,
            mask,
            committed: 0,
        }
    }

    pub fn with_terminal_eos_ids(mut self, eos_ids: Vec<i32>) -> Self {
        self.eos_ids = eos_ids;
        self
    }

    #[inline]
    pub fn eos_ids(&self) -> &[i32] {
        &self.eos_ids
    }

    #[inline]
    pub fn has_pending(&self) -> bool {
        self.pending_token.is_some()
    }

    #[inline]
    pub fn is_terminal_eos(&self) -> bool {
        self.terminal_eos
    }

    pub fn committed_len(&self) -> usize {
        self.committed
    }

    /// Returns `Some(token_id)` when the active bitmask allows **exactly one** token across
    /// the entire vocabulary (`popcount == 1`). Runs in $O(1)$ on non-singleton masks by
    /// short-circuiting on the first word with $\ge 2$ bits set.
    pub fn single_allowed_token(&self) -> Option<i32> {
        debug_assert!(
            self.pending_token.is_none(),
            "materialize_pending() must run before checking single_allowed_token()"
        );
        let mut found = None;
        for (word_idx, &word) in self.mask.iter().enumerate() {
            if word == 0 {
                continue;
            }
            if !word.is_power_of_two() || found.is_some() {
                return None;
            }
            let id = word_idx * 32 + word.trailing_zeros() as usize;
            if id >= self.vocab_size {
                return None;
            }
            found = Some(id as i32);
        }
        found
    }

    /// Returns the lowest-numbered grammar-allowed non-EOS token, if any.
    pub fn first_allowed_non_eos_token(&self) -> Option<i32> {
        debug_assert!(
            self.pending_token.is_none(),
            "materialize_pending() must run before checking first_allowed_non_eos_token()"
        );
        for (word_idx, &word) in self.mask.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let id = word_idx * 32 + bit;
                if id >= self.vocab_size {
                    return None;
                }
                let tok = id as i32;
                if !self.eos_ids.contains(&tok) {
                    return Some(tok);
                }
                w &= w - 1;
            }
        }
        None
    }

    /// Computes the deterministic fast-forward ("jump-forward") non-EOS token sequence forced
    /// by the grammar at the current state without mutating committed tokens.
    pub fn compute_ff_tokens(&mut self) -> Result<Vec<i32>, String> {
        self.materialize_pending()?;
        if self.terminal_eos {
            return Ok(Vec::new());
        }
        if let Some(matcher) = &mut self.matcher {
            let raw = matcher.compute_ff_tokens();
            if !raw.is_empty() {
                let mut out = Vec::with_capacity(raw.len());
                for id in raw {
                    let tok = id as i32;
                    if (id as usize) >= self.vocab_size || self.eos_ids.contains(&tok) {
                        break;
                    }
                    out.push(tok);
                }
                if !out.is_empty() {
                    return Ok(out);
                }
            }
        }
        Ok(self
            .single_allowed_token()
            .filter(|tok| !self.eos_ids.contains(tok))
            .into_iter()
            .collect())
    }

    pub fn current_bitmask(&self) -> Vec<u32> {
        debug_assert!(
            self.pending_token.is_none(),
            "materialize_pending() must run before reading current_bitmask()"
        );
        self.mask.clone()
    }

    pub fn mask_words(&self) -> &[u32] {
        debug_assert!(
            self.pending_token.is_none(),
            "materialize_pending() must run before reading mask_words()"
        );
        &self.mask
    }

    pub fn is_token_allowed(&self, id: i32) -> bool {
        debug_assert!(
            self.pending_token.is_none(),
            "materialize_pending() must run before checking is_token_allowed()"
        );
        id >= 0
            && (id as usize) < self.vocab_size
            && self.mask[id as usize / 32] & (1 << (id as usize % 32)) != 0
    }

    /// Materializes a deferred `advance_deferred` token (`consume_token` + `compute_mask_or_eos`)
    /// in-place while the next GPU forward pass is already executing on the device stream.
    pub fn materialize_pending(&mut self) -> Result<(), String> {
        let Some(id) = self.pending_token.take() else {
            return Ok(());
        };
        if let Some(matcher) = &self.matcher {
            let mut next = matcher.clone();
            if let Err(e) = next.consume_token(id as u32) {
                self.pending_token = Some(id);
                return Err(e.to_string());
            }
            match next.compute_mask_or_eos() {
                Ok(mask) => {
                    self.mask.clear();
                    self.mask.extend_from_slice(mask.as_slice());
                    self.matcher = Some(next);
                }
                Err(e) => {
                    self.pending_token = Some(id);
                    return Err(e.to_string());
                }
            }
        }
        Ok(())
    }

    /// Validates `id` against the active bitmask and records it for deferred `llguidance`
    /// materialization during the next GPU forward pass.
    pub fn advance_deferred(&mut self, id: i32) -> Result<(), String> {
        self.materialize_pending()?;
        if !self.is_token_allowed(id) {
            return Err(format!("token {id} violates grammar"));
        }
        // `eos_ids` holds terminal EOS tokens (populated by `ConstraintFactory` or
        // `with_terminal_eos_ids`). Save `pre_eos_mask` when matcher-less so `rollback`
        // restores the pre-terminal bitmask.
        if self.eos_ids.contains(&id) {
            if self.matcher.is_none() {
                self.pre_eos_mask = Some(self.mask.clone());
            }
            self.terminal_eos = true;
            self.mask.fill(0);
            self.committed += 1;
            return Ok(());
        }
        if self.matcher.is_some() {
            self.pending_token = Some(id);
        }
        self.committed += 1;
        Ok(())
    }

    /// Publish only after consuming and computing the successor mask succeeds.
    pub fn advance(&mut self, id: i32) -> Result<(), String> {
        self.materialize_pending()?;
        if !self.is_token_allowed(id) {
            return Err(format!("token {id} violates grammar"));
        }
        // EOS is an output event, not a grammar byte/token. llguidance may
        // already be stopped at NoExtension; consuming its EOS-only mask then
        // would error. Track EOS separately so rollback still counts emitted tokens.
        if self.eos_ids.contains(&id) {
            if self.matcher.is_none() {
                self.pre_eos_mask = Some(self.mask.clone());
            }
            self.terminal_eos = true;
            self.mask.fill(0);
            self.committed += 1;
            return Ok(());
        }
        if let Some(matcher) = &self.matcher {
            let mut next = matcher.clone();
            next.consume_token(id as u32).map_err(|e| e.to_string())?;
            let mask = next.compute_mask_or_eos().map_err(|e| e.to_string())?;
            self.mask.clear();
            self.mask.extend_from_slice(mask.as_slice());
            self.matcher = Some(next);
        }
        self.committed += 1;
        Ok(())
    }

    pub fn rollback(&mut self, count: usize) -> Result<(), String> {
        if count > self.committed {
            return Err("grammar rollback exceeds committed tokens".into());
        }
        if count == 0 {
            return Ok(());
        }
        let mut rem = count;
        if self.pending_token.is_some() {
            self.pending_token = None;
            self.committed -= 1;
            rem -= 1;
            if rem == 0 {
                return Ok(());
            }
        }
        if let Some(matcher) = &self.matcher {
            let mut next = matcher.clone();
            let grammar_count = rem - usize::from(self.terminal_eos);
            if grammar_count > 0 {
                next.rollback(grammar_count).map_err(|e| e.to_string())?;
            }
            let mask = next.compute_mask_or_eos().map_err(|e| e.to_string())?;
            self.mask.clear();
            self.mask.extend_from_slice(mask.as_slice());
            self.matcher = Some(next);
        } else if let Some(saved) = self.pre_eos_mask.take() {
            self.mask = saved;
        }
        self.terminal_eos = false;
        self.committed -= rem;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bitmask_allowed_tokens() {
        let mask = TokenBitmaskConstraint::from_allowed_tokens(128, vec![0, 31, 32, 63, 64]);
        assert_eq!(mask.vocab_size, 128);

        // Word 0 (tokens 0..32): bit 0 and bit 31 set
        assert!(mask.is_token_allowed(0));
        assert!(mask.is_token_allowed(31));
        assert!(!mask.is_token_allowed(1));
        assert!(!mask.is_token_allowed(30));

        // Word 1 (tokens 32..64): bit 0 and bit 31 set
        assert!(mask.is_token_allowed(32));
        assert!(mask.is_token_allowed(63));
        assert!(!mask.is_token_allowed(33));

        // Word 2 (tokens 64..96): bit 0 set
        assert!(mask.is_token_allowed(64));
        assert!(!mask.is_token_allowed(65));

        // Out of bounds
        assert!(!mask.is_token_allowed(-1));
        assert!(!mask.is_token_allowed(128));
        assert!(!mask.is_token_allowed(200));
    }

    #[test]
    fn test_token_bitmask_u32_slice_packing() {
        let mask = TokenBitmaskConstraint::from_allowed_tokens(64, vec![0, 33]);
        let slice = mask.mask_words();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], 1u32); // bit 0
        assert_eq!(slice[1], 1u32 << 1); // bit 1 of word 1 (token 33 = 32 + 1)
    }

    #[test]
    fn test_strict_schema_options_validation() {
        let valid = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        assert!(validate_strict_schema_options(&valid).is_ok());

        let invalid = serde_json::json!({
            "type": "object",
            "x-guidance": {
                "lenient": true
            }
        });
        assert!(validate_strict_schema_options(&invalid).is_err());
    }

    #[test]
    fn test_terminal_eos_advance_and_rollback_without_matcher() {
        let mut c = TokenBitmaskConstraint::unconstrained(64).with_terminal_eos_ids(vec![7]);
        assert!(!c.is_terminal_eos());
        assert!(c.is_token_allowed(3));
        assert!(c.is_token_allowed(7));

        c.advance(3).unwrap();
        assert!(!c.is_terminal_eos());
        assert_eq!(c.committed_len(), 1);

        c.advance(7).unwrap();
        assert!(c.is_terminal_eos());
        assert_eq!(c.committed_len(), 2);
        assert!(!c.is_token_allowed(3));
        assert!(!c.is_token_allowed(7));

        c.rollback(1).unwrap();
        assert!(!c.is_terminal_eos());
        assert_eq!(c.committed_len(), 1);
        assert!(c.is_token_allowed(3));
        assert!(c.is_token_allowed(7));
    }

    fn make_test_factory() -> (tokenizers::Tokenizer, ConstraintFactory) {
        let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .collect();
        alphabet.sort();
        let mut vocab = serde_json::Map::new();
        for (id, ch) in alphabet.into_iter().enumerate() {
            vocab.insert(ch.to_string(), serde_json::json!(id));
        }
        vocab.insert("<|im_end|>".into(), serde_json::json!(256));
        vocab.insert("<think>".into(), serde_json::json!(257));
        vocab.insert("</think>".into(), serde_json::json!(258));
        vocab.insert("<tool_call>".into(), serde_json::json!(259));
        vocab.insert("</tool_call>".into(), serde_json::json!(260));
        let tok = tokenizers::Tokenizer::from_bytes(
            serde_json::to_vec(&serde_json::json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
                    {"id": 256, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                    {"id": 257, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
                    {"id": 258, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
                    {"id": 259, "content": "<tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
                    {"id": 260, "content": "</tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
                ],
                "normalizer": null,
                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
                "post_processor": null,
                "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
                "model": {"type": "BPE", "vocab": vocab, "merges": []}
            }))
            .unwrap(),
        )
        .unwrap();
        let factory = ConstraintFactory::new(tok.clone(), 264, &[256]).unwrap();
        (tok, factory)
    }

    #[test]
    fn test_constraint_factory_grammars_advance_deferred_and_rollback() {
        let (tok, factory) = make_test_factory();
        assert!(ConstraintFactory::new(tok.clone(), 264, &[]).is_err());
        assert!(ConstraintFactory::new(tok.clone(), 264, &[999]).is_err());

        let mut c = factory.regex("ab", false).unwrap();
        assert!(format!("{c:?}").contains("TokenBitmaskConstraint"));
        assert!(!c.current_bitmask().is_empty());

        let ids = tok.encode("ab", false).unwrap().get_ids().to_vec();
        c.advance_deferred(ids[0] as i32).unwrap();
        assert!(c.has_pending());
        c.materialize_pending().unwrap();
        assert!(!c.has_pending());

        c.advance_deferred(ids[1] as i32).unwrap();
        c.rollback(1).unwrap();
        assert!(!c.has_pending());
        c.advance(ids[1] as i32).unwrap();
        c.advance(256).unwrap();
        assert!(c.is_terminal_eos());
        c.rollback(2).unwrap();
        assert!(!c.is_terminal_eos());
        assert!(c.rollback(99).is_err());

        // JSON object, thinking wrapper, structural tags, and tools
        assert!(factory.json_object(true).is_ok());
        let tag = StructuralTag {
            begin: "<tool>".into(),
            end: "</tool>".into(),
            schema: serde_json::json!({"type": "object"}),
        };
        assert!(
            factory
                .structural_tags(std::slice::from_ref(&tag), &["<tool".into()], false)
                .is_ok()
        );
        assert!(
            factory
                .structural_tags(&[], &["<tool".into()], false)
                .is_err()
        );
        assert!(
            factory
                .structural_tags(
                    std::slice::from_ref(&tag),
                    &["<t".into(), "<tool".into()],
                    false
                )
                .is_err()
        );
        assert!(
            factory
                .structural_tags(&[tag], &["<other".into()], false)
                .is_err()
        );

        let tools = serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}
                }
            }
        ]);
        assert!(
            factory
                .tools(
                    &tools,
                    &crate::tool_constraints::ToolChoice::Auto,
                    crate::tool_constraints::ToolFormat::QwenXml,
                    true,
                )
                .is_ok()
        );
    }
}

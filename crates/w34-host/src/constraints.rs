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
        schema: serde_json::Value,
        thinking: bool,
    ) -> Result<TokenBitmaskConstraint, String> {
        validate_strict_schema_options(&schema)?;
        self.compile(TopLevelGrammar::from_json_schema(schema), thinking)
    }

    pub fn regex(&self, regex: &str, thinking: bool) -> Result<TokenBitmaskConstraint, String> {
        self.compile(TopLevelGrammar::from_regex(regex), thinking)
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
            let trigger = triggers
                .iter()
                .find(|trigger| tag.begin.starts_with(trigger.as_str()))
                .ok_or("each structural begin must start with a declared trigger")?;
            let suffix = &tag.begin[trigger.len()..];
            lark.push_str(&format!(
                "tag_{index}_trig[lazy]: TAG_TEXT {}\ntag_{index}: tag_{index}_trig {} %json {} {}\n",
                serde_json::to_string(trigger).unwrap(),
                serde_json::to_string(suffix).unwrap(),
                tag.schema,
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
        if *choice != crate::tool_constraints::ToolChoice::None {
            for tool in tools.as_array().ok_or("tools must be an array")? {
                if let crate::tool_constraints::ToolChoice::Named(name) = choice
                    && tool["function"]["name"].as_str() != Some(name.as_str())
                {
                    continue;
                }
                if let Some(schema) = tool.get("function").and_then(|f| f.get("parameters")) {
                    // Validate complete schemas before XML lowering, including
                    // keywords (such as additionalProperties) not emitted there.
                    self.json_schema(schema.clone(), false)?;
                }
            }
        }
        let grammar = crate::tool_constraints::tool_grammar(tools, choice, format)?;
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
            mask,
            committed: 0,
        })
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
            mask,
            committed: 0,
        }
    }

    #[inline]
    pub fn has_pending(&self) -> bool {
        self.pending_token.is_some()
    }

    pub fn committed_len(&self) -> usize {
        self.committed
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
        if let Some(matcher) = &mut self.matcher {
            if let Err(e) = matcher.consume_token(id as u32) {
                self.pending_token = Some(id);
                return Err(e.to_string());
            }
            match matcher.compute_mask_or_eos() {
                Ok(mask) => {
                    self.mask.clear();
                    self.mask.extend_from_slice(mask.as_slice());
                }
                Err(e) => {
                    let _ = matcher.rollback(1);
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
        if self.matcher.is_some() && self.eos_ids.contains(&id) {
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
        if self.matcher.is_some() && self.eos_ids.contains(&id) {
            self.terminal_eos = true;
            self.mask.fill(0);
            self.committed += 1;
            return Ok(());
        }
        if let Some(matcher) = &mut self.matcher {
            matcher
                .consume_token(id as u32)
                .map_err(|e| e.to_string())?;
            match matcher.compute_mask_or_eos() {
                Ok(mask) => {
                    self.mask.clear();
                    self.mask.extend_from_slice(mask.as_slice());
                }
                Err(e) => {
                    let _ = matcher.rollback(1);
                    return Err(e.to_string());
                }
            }
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
}

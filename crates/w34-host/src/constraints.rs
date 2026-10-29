//! Tokenizer-aware structured decoding with cached packed-u32 masks.
//!
//! llguidance owns byte/grammar state, including the reasoning-to-answer
//! transition. Speculative clones never reconstruct state from decoded UTF-8.

use llguidance::{
    Matcher, ParserFactory,
    api::{GrammarWithLexer, TopLevelGrammar},
};
use toktrie_hf_tokenizers::ByteTokenizer;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct StructuralTag {
    pub begin: String,
    pub end: String,
    pub schema: serde_json::Value,
}

pub struct ConstraintFactory {
    factory: ParserFactory,
    vocab_size: usize,
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
        let env = tokenizer
            .into_tok_env(Some(vocab_size))
            .map_err(|e| e.to_string())?;
        let mut factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
        factory.quiet();
        Ok(Self {
            factory,
            vocab_size,
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
    mask: Vec<u32>,
    committed: usize,
}

impl std::fmt::Debug for TokenBitmaskConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBitmaskConstraint")
            .field("vocab_size", &self.vocab_size)
            .field("grammar", &self.matcher.is_some())
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
            mask,
            committed: 0,
        }
    }

    pub fn committed_len(&self) -> usize {
        self.committed
    }

    pub fn current_bitmask(&self) -> Vec<u32> {
        self.mask.clone()
    }

    pub fn mask_words(&self) -> &[u32] {
        &self.mask
    }

    pub fn is_token_allowed(&self, id: i32) -> bool {
        id >= 0
            && (id as usize) < self.vocab_size
            && self.mask[id as usize / 32] & (1 << (id as usize % 32)) != 0
    }

    /// Publish only after consuming and computing the successor mask succeeds.
    pub fn advance(&mut self, id: i32) -> Result<(), String> {
        if !self.is_token_allowed(id) {
            return Err(format!("token {id} violates grammar"));
        }
        if let Some(matcher) = &self.matcher {
            let mut next = matcher.clone();
            next.consume_token(id as u32).map_err(|e| e.to_string())?;
            let mask = next.compute_mask_or_eos().map_err(|e| e.to_string())?;
            self.mask = mask.as_slice().to_vec();
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
        if let Some(matcher) = &self.matcher {
            let mut next = matcher.clone();
            next.rollback(count).map_err(|e| e.to_string())?;
            let mask = next.compute_mask_or_eos().map_err(|e| e.to_string())?;
            self.mask = mask.as_slice().to_vec();
            self.matcher = Some(next);
        }
        self.committed -= count;
        Ok(())
    }
}

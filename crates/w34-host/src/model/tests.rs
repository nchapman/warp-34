//! Model serving capacity and tokenizer regression tests.

use super::*;

#[cfg(all(test, feature = "cuda"))]
mod serving_capacity_tests {
    use super::*;

    #[test]
    fn normalize_hf_tokenizer_clears_embedded_truncation() {
        let mut tok = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        tok.add_tokens(&[
            tokenizers::AddedToken::from("hello", false),
            tokenizers::AddedToken::from("world", false),
        ]);
        tok.with_truncation(Some(tokenizers::TruncationParams {
            max_length: 2,
            ..Default::default()
        }))
        .unwrap();
        let prompt = "hello world hello world hello";
        assert_eq!(tok.encode(prompt, false).unwrap().get_ids().len(), 2);

        let tok = normalize_hf_tokenizer(tok);
        assert_eq!(tok.encode(prompt, false).unwrap().get_ids().len(), 5);
    }

    #[test]
    fn serving_budget_is_checked() {
        assert_eq!(serving_state_slots(16).unwrap(), 64);
        assert!(serving_state_slots(0).is_err());
        assert!(serving_state_slots(usize::MAX / 4 + 1).is_err());
    }

    #[test]
    fn serving_budget_respects_memory_clamp() {
        assert_eq!(clamp_state_slots(64, 1000, 400, 10), 60);
        assert_eq!(clamp_state_slots(64, 2000, 400, 10), 64);
        assert_eq!(clamp_state_slots(24, 2000, 400, 10), 24);
        assert_eq!(clamp_state_slots(10, 2000, 400, 10), 10);
        assert_eq!(clamp_state_slots(64, 400, 400, 10), 3);
        assert_eq!(clamp_state_slots(64, 410, 400, 10), 3);
        assert_eq!(clamp_state_slots(64, 0, 400, 0), 64);
    }
}

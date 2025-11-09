use std::{
    collections::{BTreeSet, HashMap},
    error::Error,
};

use tiktoken_rs::{r50k_base, CoreBPE};

pub struct Tokenizer {
    encoder_map: HashMap<char, i64>,
    decoder_map: HashMap<i64, char>,
    pub vocab_size: i64,
}

impl Tokenizer {
    pub fn new(dataset: &str) -> Self {
        let unique_chars: BTreeSet<char> = dataset.chars().collect();

        let encoder_map: HashMap<char, i64> = unique_chars
            .iter()
            .enumerate()
            .map(|(index, &character)| return (character, index as i64))
            .collect();

        let decoder_map: HashMap<i64, char> = unique_chars
            .iter()
            .enumerate()
            .map(|(index, &character)| return (index as i64, character))
            .collect();

        let vocab_size = unique_chars.len() as i64;

        return Self {
            encoder_map,
            decoder_map,
            vocab_size,
        };
    }

    pub fn encode(&self, input: &str) -> Vec<i64> {
        return input
            .chars()
            .filter_map(|character| return self.encoder_map.get(&character).copied())
            .collect();
    }

    pub fn decode(&self, input: &[i64]) -> String {
        return input
            .iter()
            .filter_map(|&token_id| return self.decoder_map.get(&token_id))
            .collect();
    }
}

pub struct LLMTokenizer {
    tokenizer: CoreBPE,
    pub vocab_size: i64,
}

impl LLMTokenizer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        return Ok(Self {
            tokenizer: r50k_base()?,
            vocab_size: 50257,
        });
    }

    pub fn encode(&self, input: &str) -> Vec<i64> {
        let encoded_tokens = self.tokenizer.encode_with_special_tokens(input);
        return encoded_tokens.into_iter().map(|token| return token as i64).collect();
    }

    pub fn decode(&self, input: &[i64]) -> Result<String, Box<dyn Error>> {
        let token_ids = input.iter().map(|&token_id| return token_id as u32).collect();
        return Ok(self.tokenizer.decode(token_ids)?);
    }
}

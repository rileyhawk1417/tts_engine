//! A [`Tokenizer`] backed by Hugging Face `tokenizers` — faithful port of the
//! ptts-0.6.0 oracle's `tok.rs` (same crate, same options: `encode(text,
//! false)`, `decode(ids, true)`).

/// Text/token-id codec a checkpoint's LUT conditioner expects.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
    fn decode(&self, tokens: &[u32]) -> Result<String, String>;
}

/// A Hugging Face tokenizer (over the checkpoint's `tokenizer.json`).
pub struct Tok(tokenizers::Tokenizer);

impl Tok {
    /// Opens the checkpoint's explicitly supplied Hugging Face tokenizer JSON.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("failed to load tokenizer {path:?}: {e}"))?;
        Ok(Tok(tok))
    }

    /// Loads the contents of a `tokenizer.json`.
    pub fn from_bytes(json: &[u8]) -> Result<Self, String> {
        let tok = tokenizers::Tokenizer::from_bytes(json).map_err(|e| format!("tokenizer: {e}"))?;
        Ok(Tok(tok))
    }
}

impl Tokenizer for Tok {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoded = self
            .0
            .encode(text, false)
            .map_err(|e| format!("encode: {e}"))?;
        Ok(encoded.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.0.decode(ids, true).map_err(|e| format!("decode: {e}"))
    }
}

//! LEXICON-CORE: Vocabulary & Token Piece Arbitration Subsystem
//! Strict EOG verification and token piece decoding logic.

pub struct LexiconCore;

impl LexiconCore {
    /// Determines whether a sampled token ID is an End-Of-Generation token.
    /// Eliminates hardcoded checks for 0 or 2 unless explicitly flagged in vocab eog list.
    pub fn is_eog(token_id: i32, eog_tokens: &[i32]) -> bool {
        eog_tokens.contains(&token_id)
    }

    /// Cleans and sanitizes decoded token text pieces.
    /// Suppresses special control tokens such as [control_XX] from contaminating the output stream.
    pub fn sanitize_piece(piece: &str) -> String {
        if piece.starts_with("[control_") && piece.ends_with(']') {
            return String::new(); // Strip raw Tekken control tokens
        }
        piece.to_string()
    }
}

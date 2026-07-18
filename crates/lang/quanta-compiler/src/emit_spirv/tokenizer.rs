//! Shader expression tokenizer.
//!
//! Tokenizes Rust-like shader body source (from proc macro `to_token_stream()`)
//! into a flat token stream for the expression parser.

use super::constants::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ShaderCmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ShaderToken {
    Float(f32),
    /// A `u32`-suffixed integer literal (`3u32`). The suffix is the DSL's
    /// explicit unsigned spelling; a BARE integer (`3`) stays [`Float`] for
    /// backward compatibility and is coerced by type context (a comparison
    /// against a u32 value converts it with `OpConvertFToU`).
    UInt(u32),
    Ident(String),
    Op(char),         // + - * /
    Cmp(ShaderCmpOp), // < > <= >= == !=
    Dot,              // .
    ColonColon,       // ::
    Comma,            // ,
    Open,             // (
    Close,            // )
    BraceOpen,        // {
    BraceClose,       // }
    BracketOpen,      // [
    BracketClose,     // ]
    Semi,             // ;
    Eq,               // = (assignment / let binding)
}

pub(crate) fn tokenize_shader_expr(src: &str) -> Vec<ShaderToken> {
    let mut tokens = Vec::new();
    for w in src.split_whitespace() {
        tokenize_word(w, &mut tokens);
    }
    tokens
}

fn tokenize_word(w: &str, tokens: &mut Vec<ShaderToken>) {
    match w {
        "::" => tokens.push(ShaderToken::ColonColon),
        "." => tokens.push(ShaderToken::Dot),
        "," => tokens.push(ShaderToken::Comma),
        "(" => tokens.push(ShaderToken::Open),
        ")" => tokens.push(ShaderToken::Close),
        "{" => tokens.push(ShaderToken::BraceOpen),
        "}" => tokens.push(ShaderToken::BraceClose),
        "[" => tokens.push(ShaderToken::BracketOpen),
        "]" => tokens.push(ShaderToken::BracketClose),
        "+" => tokens.push(ShaderToken::Op('+')),
        "-" => tokens.push(ShaderToken::Op('-')),
        "*" => tokens.push(ShaderToken::Op('*')),
        "/" => tokens.push(ShaderToken::Op('/')),
        "<" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Lt)),
        ">" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Gt)),
        "<=" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Le)),
        ">=" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Ge)),
        "==" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Eq)),
        "!=" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Ne)),
        ";" => tokens.push(ShaderToken::Semi),
        "=" => tokens.push(ShaderToken::Eq),
        _ => {
            if let Some(split_pos) = w.find(['(', ')', ',', '{', '}', '[', ']', ';']) {
                let (before, rest) = w.split_at(split_pos);
                if !before.is_empty() {
                    tokenize_word(before, tokens);
                }
                tokenize_word(&rest[..1], tokens);
                if rest.len() > 1 {
                    tokenize_word(&rest[1..], tokens);
                }
            } else if let Some(idx) = w.find("..") {
                // A range word (`0..8u32`, `0..=4`): rustc's token printer
                // keeps `..`/`..=` attached to its operands, so the split must
                // happen here — a digit-leading word would otherwise fall
                // through the float parse into one garbage `Ident`. The `..=`
                // tail becomes `Dot Dot Eq` so the for-loop parser can reject
                // inclusive ranges by name.
                let (before, after) = (&w[..idx], &w[idx + 2..]);
                if !before.is_empty() {
                    tokenize_word(before, tokens);
                }
                tokens.push(ShaderToken::Dot);
                tokens.push(ShaderToken::Dot);
                if let Some(rest) = after.strip_prefix('=') {
                    tokens.push(ShaderToken::Eq);
                    if !rest.is_empty() {
                        tokenize_word(rest, tokens);
                    }
                } else if !after.is_empty() {
                    tokenize_word(after, tokens);
                }
            } else if w.contains('.') && !w.starts_with(|c: char| c.is_ascii_digit()) {
                if let Some(dot_pos) = w.find('.') {
                    let (before, rest) = w.split_at(dot_pos);
                    if !before.is_empty() {
                        tokenize_word(before, tokens);
                    }
                    tokens.push(ShaderToken::Dot);
                    if rest.len() > 1 {
                        tokenize_word(&rest[1..], tokens);
                    }
                }
            } else if let Some(v) = parse_u32_suffixed(w) {
                tokens.push(ShaderToken::UInt(v));
            } else if let Ok(f) = w.parse::<f32>() {
                tokens.push(ShaderToken::Float(f));
            } else {
                tokens.push(ShaderToken::Ident(w.to_string()));
            }
        }
    }
}

/// Parse a `u32`-suffixed integer literal word (`3u32`, `12u32`). Rust's
/// token printer keeps a suffixed literal contiguous, so the whole word
/// arrives as one token. A bare `u32` (the type ident in `as u32`) has no
/// digits and falls through to `Ident`; digits with underscores (`1_000u32`)
/// are accepted the way Rust spells them.
fn parse_u32_suffixed(w: &str) -> Option<u32> {
    let digits = w.strip_suffix("u32")?;
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit() || c == '_') {
        return None;
    }
    digits.replace('_', "").parse::<u32>().ok()
}

/// Map function name to GLSL.std.450 extended instruction opcode.
pub(crate) fn glsl_func_id(name: &str) -> Option<u32> {
    match name {
        "sin" => Some(GLSL_SIN),
        "cos" => Some(GLSL_COS),
        "tan" => Some(GLSL_TAN),
        "asin" => Some(GLSL_ASIN),
        "acos" => Some(GLSL_ACOS),
        "atan" => Some(GLSL_ATAN),
        "sqrt" => Some(GLSL_SQRT),
        "inverseSqrt" | "inverse_sqrt" => Some(GLSL_INVERSE_SQRT),
        "abs" => Some(GLSL_FABS),
        "floor" => Some(GLSL_FLOOR),
        "ceil" => Some(GLSL_CEIL),
        "round" => Some(GLSL_ROUND),
        "fract" => Some(GLSL_FRACT),
        "min" => Some(GLSL_FMIN),
        "max" => Some(GLSL_FMAX),
        "clamp" => Some(GLSL_FCLAMP),
        "mix" => Some(GLSL_FMIX),
        "step" => Some(GLSL_STEP),
        "smoothstep" | "smooth_step" => Some(GLSL_SMOOTH_STEP),
        "pow" => Some(GLSL_POW),
        "exp" => Some(GLSL_EXP),
        "log" => Some(GLSL_LOG),
        "exp2" => Some(GLSL_EXP2),
        "log2" => Some(GLSL_LOG2),
        "normalize" => Some(GLSL_NORMALIZE),
        "length" => Some(GLSL_LENGTH),
        "distance" => Some(GLSL_DISTANCE),
        "cross" => Some(GLSL_CROSS),
        "fma" => Some(GLSL_FMA),
        "atan2" => Some(GLSL_ATAN2),
        _ => None,
    }
}

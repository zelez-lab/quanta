//! Shader-body tokenizer for the WGSL vertex/fragment emitter.
//!
//! The render-shader body arrives as rustc's `to_token_stream().to_string()`:
//! `::` is spaced, the call paren is attached (`new(`), dots are unspaced
//! (`uv.x`), trailing commas are preserved, and long lines wrap on ~90 cols —
//! a wrap can split `Vec4 ::` and `new(` across a newline. A tokenizer absorbs
//! all of that naturally; the emitter must never regex or string-replace the
//! raw text.
//!
//! This is quanta-ir's own copy of the SPIR-V shader tokenizer's design
//! (`quanta-compiler/src/emit_spirv/tokenizer.rs`) — quanta-ir has an empty
//! `[dependencies]` (no `syn`, no external crates), and the SPIR-V tokenizer
//! lives in a sibling crate that cannot be imported here. Whatever construct
//! the SPIR-V walker accepts, this one accepts; whatever it rejects, the WGSL
//! walker rejects with a clear error.

/// The six comparison operators the shader grammar admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ShaderCmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

impl ShaderCmpOp {
    /// The WGSL spelling — identical to the Rust source spelling.
    pub(super) fn wgsl(self) -> &'static str {
        match self {
            ShaderCmpOp::Lt => "<",
            ShaderCmpOp::Gt => ">",
            ShaderCmpOp::Le => "<=",
            ShaderCmpOp::Ge => ">=",
            ShaderCmpOp::Eq => "==",
            ShaderCmpOp::Ne => "!=",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ShaderToken {
    Float(f32),
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

/// Split the body source into a flat token stream. Whitespace (including the
/// wrap newline) is the primary delimiter; each whitespace-separated word is
/// then peeled apart at punctuation so an attached `new(` or `1.0,)` splits
/// correctly. Mirrors the SPIR-V tokenizer's split discipline exactly.
pub(super) fn tokenize_shader_expr(src: &str) -> Vec<ShaderToken> {
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
            // A word carrying attached punctuation (`new(`, `1.0,)`, `uv.x`)
            // is peeled apart one delimiter at a time. Bracket/brace/paren/
            // comma/semicolon first, since those bind tighter than the dot.
            if let Some(split_pos) = w.find(['(', ')', ',', '{', '}', '[', ']', ';']) {
                let (before, rest) = w.split_at(split_pos);
                if !before.is_empty() {
                    tokenize_word(before, tokens);
                }
                tokenize_word(&rest[..1], tokens);
                if rest.len() > 1 {
                    tokenize_word(&rest[1..], tokens);
                }
            } else if w.contains('.') && !w.starts_with(|c: char| c.is_ascii_digit()) {
                // `uv.x` → `uv` `.` `x`. A leading digit means a float literal
                // (`1.0`), which is NOT split — the parse below handles it.
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
            } else if let Ok(f) = w.parse::<f32>() {
                tokens.push(ShaderToken::Float(f));
            } else {
                tokens.push(ShaderToken::Ident(w.to_string()));
            }
        }
    }
}

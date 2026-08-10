//! The bridge between the regex engine's property seam and the
//! vendored Unicode tables.
//!
//! [`regex::Regex`](crate::regex) resolves `\p{…}`, `\s`, `\d` and
//! `\w` through the [`PropertyLookup`] seam so the engine owns no
//! tables; this module implements the seam over [`crate::unicode`]
//! and exposes the one `'static` instance patterns are compiled
//! against (`'static` keeps `Regex: Send + Sync` for scoped-thread
//! batch encode).
//!
//! `Word` follows the reference (Oniguruma) definition: letters,
//! marks, decimal digits and connector punctuation (`Pc` — the
//! underscore class). The fixture layer re-pins this against real
//! tokenizer outputs.

use crate::regex::{PropClass, PropertyLookup};
use crate::unicode::{self, GeneralCategory};

/// The vendored-table implementation of the regex property seam.
pub struct UnicodeTables;

/// The instance every pattern in this crate compiles against.
pub static TABLES: UnicodeTables = UnicodeTables;

impl PropertyLookup for UnicodeTables {
    fn is_class(&self, c: char, class: PropClass) -> bool {
        match class {
            PropClass::Letter => unicode::is_letter(c),
            PropClass::Number => unicode::is_number(c),
            PropClass::Mark => unicode::is_mark(c),
            PropClass::NonspacingMark => unicode::is_nonspacing_mark(c),
            PropClass::Punctuation => {
                unicode::category_class(c) == unicode::CategoryClass::Punctuation
            }
            PropClass::Symbol => unicode::category_class(c) == unicode::CategoryClass::Symbol,
            PropClass::Control => unicode::category_class(c) == unicode::CategoryClass::Other,
            PropClass::DecimalDigit => unicode::general_category(c) == GeneralCategory::Nd,
            PropClass::Whitespace => unicode::is_whitespace(c),
            PropClass::Word => {
                unicode::is_letter(c)
                    || unicode::is_mark(c)
                    || unicode::general_category(c) == GeneralCategory::Nd
                    || unicode::general_category(c) == GeneralCategory::Pc
            }
        }
    }
}

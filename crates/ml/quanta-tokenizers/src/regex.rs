//! The split-regex engine: a backtracking matcher over the closed construct
//! set the tokenizer-artifact corpus uses.
//!
//! `Split`, `ByteLevel(use_regex)`, and `Replace(Regex)` carry patterns the
//! reference implementation runs on Oniguruma. Reimplementing a general
//! Oniguruma is not honest zero-dep work; hardcoding the known model
//! patterns would be reopened by every new model lineage. This module is
//! the declared middle: an in-crate backtracking engine over exactly the
//! construct inventory of the GPT-2 pattern, the cl100k/Llama-3 pattern,
//! and the whitespace/punctuation pattern family. A pattern using anything
//! outside that set is a loud [`RegexError::UnsupportedConstruct`] naming
//! the construct — the claim boundary: "runs the patterns the model corpus
//! writes"; a new construct is additive engine work, never a misparse.
//!
//! # Supported construct set
//!
//! | Construct | Notes |
//! |---|---|
//! | literals | incl. escaped metachars (`\.`, `\\`, …), `\n` `\r` `\t` `\f` `\v` `\0`, and the literal spellings `\x{HEX}` / `\uHHHH` |
//! | `.` | any char except `\n` (Oniguruma default, no dotall) |
//! | `[...]` classes | ranges, negation `[^...]`, escapes, `\p{…}`/`\P{…}` and `\s`-family items inside |
//! | `\p{X}` / `\P{X}` | property names `L` `N` `M` `Mn` `Nd` `P` `S` `C` (closed set; unknown names error loudly) |
//! | `\s` `\d` `\w` + `\S` `\D` `\W` | via the property seam (Unicode-wide, matching the reference's Oniguruma behavior) |
//! | alternation | ordered, leftmost-first (PCRE/Oniguruma preference, not POSIX-longest) |
//! | `(...)` / `(?:...)` | plain groups; captures are never exposed (pre-tokenizers consume whole-match spans only) |
//! | `(?i:...)` | scoped case-insensitivity, simple one-to-one case folding via `char::to_lowercase` |
//! | `(?!...)` | negative lookahead |
//! | `*` `+` `?` `{m}` `{m,}` `{m,n}` | greedy, plus lazy `?` variants; counts capped at 512 |
//!
//! Rejected loudly (each names itself in the error): anchors `^` `$` `\A`
//! `\z` `\Z` `\G`, word boundaries `\b` `\B`, backreferences, lookbehind,
//! positive lookahead `(?=`, possessive quantifiers, atomic groups `(?>`,
//! named groups, comment groups, bare inline flags like `(?i)`, POSIX
//! bracket classes `[:alpha:]`, unknown escapes and property names, and
//! unbounded quantifiers over a sub-expression that can match empty
//! (`(?:a?)*`-style, which a backtracker can only answer with a budget
//! trip — rejecting it at parse keeps the budget error meaning "blowup").
//!
//! An Oniguruma-compatible reading is kept for one ambiguity: a `{` that
//! does not open a well-formed `{m}` / `{m,}` / `{m,n}` quantifier is a
//! literal `{` (the reference accepts e.g. `a{x}` as literal text).
//!
//! # The Unicode property seam
//!
//! `\p{L}`-class constructs need the Unicode category tables, which are
//! owned by the crate's `unicode` module (vendored generated tables). To
//! keep this engine free of that dependency — and testable against a stub —
//! property membership is resolved through the [`PropertyLookup`] trait:
//! the engine asks `is_class(c, PropClass::…)` at match time and never
//! interprets table data itself. [`Regex::parse`] takes a
//! `&'static dyn PropertyLookup` (the real tables are a `static`, and a
//! `'static` borrow keeps [`Regex`] `Send + Sync` for
//! `std::thread::scope`-style batch encoding). At integration the
//! pre-tokenizer layer passes the unicode module's table singleton; tests
//! pass a stub.
//!
//! ```
//! use quanta_tokenizers::regex::{PropClass, PropertyLookup, Regex};
//!
//! struct Ascii;
//! impl PropertyLookup for Ascii {
//!     fn is_class(&self, c: char, class: PropClass) -> bool {
//!         match class {
//!             PropClass::Letter => c.is_ascii_alphabetic(),
//!             PropClass::Number | PropClass::DecimalDigit => c.is_ascii_digit(),
//!             PropClass::Whitespace => c.is_ascii_whitespace(),
//!             PropClass::Word => c.is_ascii_alphanumeric() || c == '_',
//!             _ => false,
//!         }
//!     }
//! }
//! static ASCII: Ascii = Ascii;
//!
//! let re = Regex::parse(r" ?\p{L}+", &ASCII)?;
//! let spans: Vec<_> = re.find_iter("hi there").collect::<Result<_, _>>()?;
//! assert_eq!(spans, vec![(0, 2), (2, 8)]);
//! # Ok::<(), quanta_tokenizers::regex::RegexError>(())
//! ```
//!
//! # Serving `Split { pattern, behavior }`
//!
//! Pre-tokenizers need the matches *and* the gaps between them.
//! [`Regex::segments`] returns the full partition of the input as an
//! ordered run of non-empty [`Segment`]s tagged [`SegmentKind::Match`] or
//! [`SegmentKind::Gap`]; every `SplitDelimiterBehavior` is a trivial fold
//! over it (`Removed` keeps gaps, `Isolated` keeps everything,
//! `MergedWithPrevious`/`MergedWithNext` fuse each match into its
//! neighbor, `Contiguous` fuses adjacent matches; `invert` swaps the two
//! kinds). [`Regex::find_iter`] yields raw match spans for
//! `Replace`-style consumers. All spans are **byte offsets** into the
//! input `&str`, always on `char` boundaries.
//!
//! Empty matches (possible only if the whole pattern is nullable, e.g.
//! `a*`) are yielded by `find_iter` as zero-width spans with the scan
//! advancing one char, and act as pure split points in `segments` (they
//! contribute no `Match` segment but still separate gaps). No corpus
//! pattern can match empty; the behavior is pinned by tests for the claim
//! boundary's sake.
//!
//! # The backtracking budget (hostile-artifact posture)
//!
//! Patterns come from artifacts, i.e. they are untrusted. Every
//! `find_iter` / `segments` / `is_match` call carries a step budget of
//! `4096 + 1024 × input_chars` VM steps for the whole scan, and a
//! backtrack-stack cap of `4096 + 4 × input_chars` entries. Corpus
//! patterns run at a few dozen steps per char; a pattern that goes
//! super-linear (`(a+)+b`-style) trips the budget and surfaces
//! [`RegexError::Budget`] — a loud error, never a hang. Every VM loop
//! iteration consumes budget, so termination is unconditional.

use std::fmt;
use std::iter::FusedIterator;

/// Nesting depth cap for groups (parser and compiler recursion bound).
const MAX_NEST: usize = 64;
/// Cap on `{m,n}` repetition counts.
const MAX_REPEAT: u32 = 512;
/// Cap on compiled program size (instructions, all lookahead bodies included).
const MAX_PROG: usize = 8192;
/// Flat part of the per-scan step budget.
const BUDGET_BASE: usize = 4096;
/// Per-input-char part of the per-scan step budget.
const BUDGET_PER_CHAR: usize = 1024;
/// Flat part of the backtrack-stack cap.
const STACK_BASE: usize = 4096;
/// Per-input-char part of the backtrack-stack cap.
const STACK_PER_CHAR: usize = 4;

/// The closed set of character properties the engine can ask about.
///
/// This enum is the vocabulary of the [`PropertyLookup`] seam: pattern
/// spellings map onto it at parse time (`\p{L}` → `Letter`, `\d` →
/// `DecimalDigit`, …) and the unicode-table lane answers membership at
/// match time. Growing the corpus's property inventory means adding a
/// variant here and a row in the table lane — additive on both sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PropClass {
    /// `\p{L}` — any letter.
    Letter,
    /// `\p{N}` — any number.
    Number,
    /// `\p{M}` — any mark.
    Mark,
    /// `\p{Mn}` — nonspacing mark.
    NonspacingMark,
    /// `\p{P}` — punctuation (all subcategories).
    Punctuation,
    /// `\p{S}` — symbol (all subcategories).
    Symbol,
    /// `\p{C}` — control/other.
    Control,
    /// `\d` / `\p{Nd}` — decimal digit.
    DecimalDigit,
    /// `\s` — White_Space.
    Whitespace,
    /// `\w` — word character (the table lane owns the exact definition,
    /// pinned against reference fixtures at integration).
    Word,
}

/// The Unicode property seam: answers "is `c` in `class`?".
///
/// Implemented by the crate's vendored generated tables at integration and
/// by a stub in this lane's tests. `Sync` is required so a `&'static`
/// lookup keeps [`Regex`] `Send + Sync`.
pub trait PropertyLookup: Sync {
    /// Whether `c` belongs to `class`.
    fn is_class(&self, c: char, class: PropClass) -> bool;
}

/// Errors from parsing or running a pattern.
///
/// Wrapped by the crate-level error type as the `RegexConstruct` family:
/// both the claim-boundary rejections and the backtracking-budget trip
/// carry the offending pattern in the message, per the house contract.
#[derive(Debug, Clone)]
pub enum RegexError {
    /// Malformed pattern (unterminated class/group, reversed range, …).
    Parse {
        /// The full pattern text.
        pattern: String,
        /// Byte offset of the fault within the pattern.
        at: usize,
        /// What is wrong.
        what: String,
    },
    /// Syntactically recognizable construct outside the supported set.
    UnsupportedConstruct {
        /// The full pattern text.
        pattern: String,
        /// Byte offset of the construct within the pattern.
        at: usize,
        /// The construct's name, e.g. "backreference `\1`".
        construct: String,
    },
    /// The backtracking budget tripped: the pattern went super-linear on
    /// this input. Loud error instead of a hang.
    Budget {
        /// The full pattern text.
        pattern: String,
        /// The step budget that was exhausted.
        steps: usize,
        /// Input length in chars, from which the budget was sized.
        input_chars: usize,
    },
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexError::Parse { pattern, at, what } => {
                write!(
                    f,
                    "regex parse error at byte {at} of pattern `{pattern}`: {what}"
                )
            }
            RegexError::UnsupportedConstruct {
                pattern,
                at,
                construct,
            } => {
                write!(
                    f,
                    "unsupported regex construct at byte {at} of pattern `{pattern}`: \
                     {construct} is outside the closed construct set quanta-tokenizers \
                     implements (the tokenizer-artifact corpus profile)"
                )
            }
            RegexError::Budget {
                pattern,
                steps,
                input_chars,
            } => {
                write!(
                    f,
                    "regex backtracking budget exceeded: pattern `{pattern}` needed more \
                     than {steps} steps on a {input_chars}-char input; artifact-supplied \
                     patterns are bounded to input-linear work and fail loudly instead of \
                     hanging"
                )
            }
        }
    }
}

impl std::error::Error for RegexError {}

/// Which side of the match/gap partition a [`Segment`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// A span the pattern matched.
    Match,
    /// A span between matches (also before the first and after the last).
    Gap,
}

/// One non-empty span of the input, as produced by [`Regex::segments`].
///
/// `start..end` are byte offsets into the input `&str`, always on `char`
/// boundaries; consecutive segments tile the input exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// Byte offset of the span's start.
    pub start: usize,
    /// Byte offset one past the span's end.
    pub end: usize,
    /// Match or gap.
    pub kind: SegmentKind,
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum ClassItem {
    Char(char),
    Range(char, char),
    Prop(PropClass),
    NegProp(PropClass),
}

#[derive(Debug, Clone)]
struct ClassDef {
    negated: bool,
    items: Vec<ClassItem>,
}

#[derive(Debug)]
enum Ast {
    Empty,
    Char(char),
    Any,
    Class(ClassDef),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat {
        node: Box<Ast>,
        min: u32,
        max: Option<u32>,
        lazy: bool,
    },
    Group(Box<Ast>),
    CaseInsensitive(Box<Ast>),
    NegLookahead(Box<Ast>),
}

/// Whether `ast` can match the empty string (drives the parse-time
/// rejection of unbounded quantifiers over nullable bodies).
fn nullable(ast: &Ast) -> bool {
    match ast {
        Ast::Empty => true,
        Ast::Char(_) | Ast::Any | Ast::Class(_) => false,
        Ast::Concat(items) => items.iter().all(nullable),
        Ast::Alt(branches) => branches.iter().any(nullable),
        Ast::Repeat { node, min, .. } => *min == 0 || nullable(node),
        Ast::Group(inner) | Ast::CaseInsensitive(inner) => nullable(inner),
        Ast::NegLookahead(_) => true,
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

enum ClassAtom {
    Char(char),
    Set(ClassItem),
}

struct Parser<'p> {
    pattern: &'p str,
    chars: Vec<(usize, char)>,
    pos: usize,
}

impl<'p> Parser<'p> {
    fn new(pattern: &'p str) -> Self {
        Parser {
            pattern,
            chars: pattern.char_indices().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).map(|&(_, c)| c)
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).map(|&(_, c)| c)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn eat(&mut self, want: char) -> bool {
        if self.peek() == Some(want) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Byte offset of the current position within the pattern.
    fn here(&self) -> usize {
        self.chars
            .get(self.pos)
            .map_or(self.pattern.len(), |&(b, _)| b)
    }

    fn parse_err(&self, at: usize, what: impl Into<String>) -> RegexError {
        RegexError::Parse {
            pattern: self.pattern.to_string(),
            at,
            what: what.into(),
        }
    }

    fn unsupported(&self, at: usize, construct: impl Into<String>) -> RegexError {
        RegexError::UnsupportedConstruct {
            pattern: self.pattern.to_string(),
            at,
            construct: construct.into(),
        }
    }

    fn parse(pattern: &'p str) -> Result<Ast, RegexError> {
        let mut parser = Parser::new(pattern);
        let ast = parser.parse_alt(0)?;
        if let Some(c) = parser.peek() {
            // The only way parse_alt stops early at top level.
            return Err(parser.parse_err(parser.here(), format!("unmatched `{c}`")));
        }
        Ok(ast)
    }

    fn parse_alt(&mut self, depth: usize) -> Result<Ast, RegexError> {
        let mut branches = vec![self.parse_concat(depth)?];
        while self.eat('|') {
            branches.push(self.parse_concat(depth)?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().expect("one branch")
        } else {
            Ast::Alt(branches)
        })
    }

    fn parse_concat(&mut self, depth: usize) -> Result<Ast, RegexError> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            items.push(self.parse_repeat(depth)?);
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().expect("one item"),
            _ => Ast::Concat(items),
        })
    }

    fn parse_repeat(&mut self, depth: usize) -> Result<Ast, RegexError> {
        let atom = self.parse_atom(depth)?;
        let quant_at = self.here();
        let Some((min, max, lazy)) = self.parse_quantifier()? else {
            return Ok(atom);
        };
        // A second quantifier directly after the first: possessive (`a++`)
        // or nested (`a**`, `a{2}{3}`) — both outside the set.
        if self.peek() == Some('+') {
            return Err(self.unsupported(self.here(), "a possessive quantifier (`++`/`*+`/`?+`)"));
        }
        let nested_at = self.here();
        if self.parse_quantifier()?.is_some() {
            return Err(self.parse_err(nested_at, "nested quantifier"));
        }
        if max.is_none() && nullable(&atom) {
            return Err(self.unsupported(
                quant_at,
                "an unbounded quantifier over a sub-expression that can match the \
                 empty string (`(?:a?)*`-style)",
            ));
        }
        Ok(Ast::Repeat {
            node: Box::new(atom),
            min,
            max,
            lazy,
        })
    }

    /// `Some((min, max, lazy))` if a quantifier is present at the cursor.
    /// A `{` that does not form a well-formed counted quantifier is left
    /// unconsumed (it will re-enter as a literal atom).
    fn parse_quantifier(&mut self) -> Result<Option<(u32, Option<u32>, bool)>, RegexError> {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.bump();
                (0, None)
            }
            Some('+') => {
                self.bump();
                (1, None)
            }
            Some('?') => {
                self.bump();
                (0, Some(1))
            }
            Some('{') => match self.try_counted()? {
                Some(bounds) => bounds,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        let lazy = self.eat('?');
        Ok(Some((min, max, lazy)))
    }

    /// Attempts `{m}` / `{m,}` / `{m,n}` at the cursor (which is on `{`).
    /// Restores the cursor and returns `None` when the braces do not form
    /// a counted quantifier (Oniguruma-compatible literal-`{` fallback).
    fn try_counted(&mut self) -> Result<Option<(u32, Option<u32>)>, RegexError> {
        let start = self.pos;
        let brace_at = self.here();
        self.bump(); // '{'
        let mut min_digits = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                min_digits.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if min_digits.is_empty() {
            self.pos = start;
            return Ok(None);
        }
        let max = match self.peek() {
            Some('}') => {
                self.bump();
                Some(min_digits.clone())
            }
            Some(',') => {
                self.bump();
                let mut max_digits = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        max_digits.push(c);
                        self.bump();
                    } else {
                        break;
                    }
                }
                if self.eat('}') {
                    if max_digits.is_empty() {
                        None
                    } else {
                        Some(max_digits)
                    }
                } else {
                    self.pos = start;
                    return Ok(None);
                }
            }
            _ => {
                self.pos = start;
                return Ok(None);
            }
        };
        let parse_count = |digits: &str| -> Result<u32, RegexError> {
            let n: u32 = digits
                .parse()
                .map_err(|_| self.parse_err(brace_at, "repetition count too large"))?;
            if n > MAX_REPEAT {
                return Err(self.parse_err(
                    brace_at,
                    format!("repetition count {n} exceeds the cap of {MAX_REPEAT}"),
                ));
            }
            Ok(n)
        };
        let min = parse_count(&min_digits)?;
        let max = max.as_deref().map(parse_count).transpose()?;
        match max {
            Some(n) if min > n => Err(self.parse_err(
                brace_at,
                format!("reversed repetition range `{{{min},{n}}}`"),
            )),
            _ => Ok(Some((min, max))),
        }
    }

    fn parse_atom(&mut self, depth: usize) -> Result<Ast, RegexError> {
        let at = self.here();
        let c = self.peek().expect("caller checked for an atom");
        match c {
            '(' => self.parse_group(depth),
            '[' => self.parse_class(),
            '.' => {
                self.bump();
                Ok(Ast::Any)
            }
            '\\' => self.parse_escape_atom(),
            '^' => Err(self.unsupported(at, "the anchor `^`")),
            '$' => Err(self.unsupported(at, "the anchor `$`")),
            '*' | '+' | '?' => {
                Err(self.parse_err(at, format!("quantifier `{c}` with nothing to repeat")))
            }
            // `{` here means the literal-`{` fallback already applied (or a
            // bare `{`); `}` and `]` outside their constructs are literal.
            _ => {
                self.bump();
                Ok(Ast::Char(c))
            }
        }
    }

    fn parse_group(&mut self, depth: usize) -> Result<Ast, RegexError> {
        let open_at = self.here();
        self.bump(); // '('
        if depth >= MAX_NEST {
            return Err(self.parse_err(open_at, format!("group nesting deeper than {MAX_NEST}")));
        }
        let node = if self.eat('?') {
            match self.peek() {
                Some(':') => {
                    self.bump();
                    Ast::Group(Box::new(self.parse_alt(depth + 1)?))
                }
                Some('i') => {
                    if self.peek_at(1) == Some(':') {
                        self.bump();
                        self.bump();
                        Ast::CaseInsensitive(Box::new(self.parse_alt(depth + 1)?))
                    } else {
                        return Err(self.unsupported(
                            open_at,
                            "an inline flag group without a `:` scope (only `(?i:…)` is in \
                             the corpus set)",
                        ));
                    }
                }
                Some('!') => {
                    self.bump();
                    Ast::NegLookahead(Box::new(self.parse_alt(depth + 1)?))
                }
                Some('=') => return Err(self.unsupported(open_at, "positive lookahead `(?=`")),
                Some('<') => {
                    return Err(match self.peek_at(1) {
                        Some('=') => self.unsupported(open_at, "lookbehind `(?<=`"),
                        Some('!') => self.unsupported(open_at, "lookbehind `(?<!`"),
                        _ => self.unsupported(open_at, "a named group `(?<`"),
                    });
                }
                Some('P') => return Err(self.unsupported(open_at, "a named group `(?P`")),
                Some('>') => return Err(self.unsupported(open_at, "an atomic group `(?>`")),
                Some('#') => return Err(self.unsupported(open_at, "a comment group `(?#`")),
                Some(other) => {
                    return Err(
                        self.unsupported(open_at, format!("the group modifier `(?{other}`"))
                    );
                }
                None => return Err(self.parse_err(open_at, "unterminated group")),
            }
        } else {
            // Plain `(...)`: parsed as a group; captures are never exposed
            // (the pre-tokenizer surface consumes whole-match spans only).
            Ast::Group(Box::new(self.parse_alt(depth + 1)?))
        };
        if !self.eat(')') {
            return Err(self.parse_err(open_at, "unterminated group"));
        }
        Ok(node)
    }

    fn parse_class(&mut self) -> Result<Ast, RegexError> {
        let open_at = self.here();
        self.bump(); // '['
        let negated = self.eat('^');
        let mut items: Vec<ClassItem> = Vec::new();
        loop {
            let item_at = self.here();
            let Some(c) = self.peek() else {
                return Err(self.parse_err(open_at, "unterminated character class"));
            };
            if c == ']' {
                self.bump();
                if items.is_empty() {
                    return Err(self.parse_err(open_at, "empty character class"));
                }
                return Ok(Ast::Class(ClassDef { negated, items }));
            }
            if c == '[' && self.peek_at(1) == Some(':') {
                return Err(self.unsupported(item_at, "a POSIX bracket class `[:…:]`"));
            }
            if c == '-' {
                // An unescaped `-` at item position is literal ('[-a]',
                // '[a--]'); only char–`-`–char forms a range below.
                self.bump();
                items.push(ClassItem::Char('-'));
                continue;
            }
            match self.parse_class_atom()? {
                ClassAtom::Set(item) => {
                    if self.peek() == Some('-') && self.peek_at(1).is_some_and(|c| c != ']') {
                        return Err(self.parse_err(
                            self.here(),
                            "invalid range: `\\d`-style and `\\p{…}` items cannot be range \
                             endpoints",
                        ));
                    }
                    items.push(item);
                }
                ClassAtom::Char(lo) => {
                    if self.peek() == Some('-') && self.peek_at(1).is_some_and(|c| c != ']') {
                        let dash_at = self.here();
                        self.bump(); // '-'
                        let hi_at = self.here();
                        match self.parse_class_atom()? {
                            ClassAtom::Char(hi) => {
                                if lo > hi {
                                    return Err(self.parse_err(
                                        dash_at,
                                        format!("reversed range `{lo}-{hi}`"),
                                    ));
                                }
                                items.push(ClassItem::Range(lo, hi));
                            }
                            ClassAtom::Set(_) => {
                                return Err(self.parse_err(
                                    hi_at,
                                    "invalid range: `\\d`-style and `\\p{…}` items cannot be \
                                     range endpoints",
                                ));
                            }
                        }
                    } else {
                        items.push(ClassItem::Char(lo));
                    }
                }
            }
        }
    }

    fn parse_class_atom(&mut self) -> Result<ClassAtom, RegexError> {
        let at = self.here();
        let c = self.peek().expect("caller checked for a class atom");
        if c != '\\' {
            self.bump();
            return Ok(ClassAtom::Char(c));
        }
        self.bump(); // '\'
        let Some(e) = self.bump() else {
            return Err(self.parse_err(at, "dangling `\\` in character class"));
        };
        Ok(match e {
            'n' => ClassAtom::Char('\n'),
            'r' => ClassAtom::Char('\r'),
            't' => ClassAtom::Char('\t'),
            'f' => ClassAtom::Char('\u{c}'),
            'v' => ClassAtom::Char('\u{b}'),
            '0' => ClassAtom::Char('\0'),
            'd' => ClassAtom::Set(ClassItem::Prop(PropClass::DecimalDigit)),
            'D' => ClassAtom::Set(ClassItem::NegProp(PropClass::DecimalDigit)),
            's' => ClassAtom::Set(ClassItem::Prop(PropClass::Whitespace)),
            'S' => ClassAtom::Set(ClassItem::NegProp(PropClass::Whitespace)),
            'w' => ClassAtom::Set(ClassItem::Prop(PropClass::Word)),
            'W' => ClassAtom::Set(ClassItem::NegProp(PropClass::Word)),
            'p' => ClassAtom::Set(ClassItem::Prop(self.parse_prop_name(at)?)),
            'P' => ClassAtom::Set(ClassItem::NegProp(self.parse_prop_name(at)?)),
            'x' => ClassAtom::Char(self.parse_hex_brace(at)?),
            'u' => ClassAtom::Char(self.parse_u_hex4(at)?),
            other if other.is_ascii_alphanumeric() => {
                return Err(self.unsupported(
                    at,
                    format!("the escape `\\{other}` inside a character class"),
                ));
            }
            other => ClassAtom::Char(other),
        })
    }

    fn parse_escape_atom(&mut self) -> Result<Ast, RegexError> {
        let at = self.here();
        self.bump(); // '\'
        let Some(e) = self.bump() else {
            return Err(self.parse_err(at, "dangling `\\` at end of pattern"));
        };
        let prop = |class: PropClass| {
            Ast::Class(ClassDef {
                negated: false,
                items: vec![ClassItem::Prop(class)],
            })
        };
        let neg_prop = |class: PropClass| {
            Ast::Class(ClassDef {
                negated: false,
                items: vec![ClassItem::NegProp(class)],
            })
        };
        Ok(match e {
            'n' => Ast::Char('\n'),
            'r' => Ast::Char('\r'),
            't' => Ast::Char('\t'),
            'f' => Ast::Char('\u{c}'),
            'v' => Ast::Char('\u{b}'),
            '0' => Ast::Char('\0'),
            'd' => prop(PropClass::DecimalDigit),
            'D' => neg_prop(PropClass::DecimalDigit),
            's' => prop(PropClass::Whitespace),
            'S' => neg_prop(PropClass::Whitespace),
            'w' => prop(PropClass::Word),
            'W' => neg_prop(PropClass::Word),
            'p' => prop(self.parse_prop_name(at)?),
            'P' => neg_prop(self.parse_prop_name(at)?),
            'x' => Ast::Char(self.parse_hex_brace(at)?),
            'u' => Ast::Char(self.parse_u_hex4(at)?),
            '1'..='9' => return Err(self.unsupported(at, format!("a backreference `\\{e}`"))),
            'b' | 'B' => return Err(self.unsupported(at, format!("the word boundary `\\{e}`"))),
            'A' | 'z' | 'Z' | 'G' => {
                return Err(self.unsupported(at, format!("the anchor `\\{e}`")));
            }
            'k' => return Err(self.unsupported(at, "a named backreference `\\k`")),
            other if other.is_ascii_alphanumeric() => {
                return Err(self.unsupported(at, format!("the escape `\\{other}`")));
            }
            other => Ast::Char(other),
        })
    }

    /// Parses `{Name}` after `\p` / `\P` and maps it onto the closed
    /// [`PropClass`] set.
    fn parse_prop_name(&mut self, at: usize) -> Result<PropClass, RegexError> {
        if !self.eat('{') {
            return Err(self.parse_err(at, "`\\p` requires the `\\p{…}` form"));
        }
        let mut name = String::new();
        loop {
            match self.bump() {
                Some('}') => break,
                Some(c) => name.push(c),
                None => return Err(self.parse_err(at, "unterminated `\\p{…}`")),
            }
        }
        Ok(match name.as_str() {
            "L" => PropClass::Letter,
            "N" => PropClass::Number,
            "M" => PropClass::Mark,
            "Mn" => PropClass::NonspacingMark,
            "Nd" => PropClass::DecimalDigit,
            "P" => PropClass::Punctuation,
            "S" => PropClass::Symbol,
            "C" => PropClass::Control,
            _ => return Err(self.unsupported(at, format!("the unicode property `\\p{{{name}}}`"))),
        })
    }

    /// Parses `{HEX}` after `\x` — a literal-char spelling.
    fn parse_hex_brace(&mut self, at: usize) -> Result<char, RegexError> {
        if !self.eat('{') {
            return Err(self.parse_err(at, "`\\x` requires the `\\x{…}` form"));
        }
        let mut digits = String::new();
        loop {
            match self.bump() {
                Some('}') => break,
                Some(c) if c.is_ascii_hexdigit() && digits.len() < 6 => digits.push(c),
                Some(c) => {
                    return Err(self.parse_err(at, format!("invalid `\\x{{…}}` digit `{c}`")));
                }
                None => return Err(self.parse_err(at, "unterminated `\\x{…}`")),
            }
        }
        if digits.is_empty() {
            return Err(self.parse_err(at, "empty `\\x{…}`"));
        }
        let value = u32::from_str_radix(&digits, 16).expect("bounded hex digits");
        char::from_u32(value)
            .ok_or_else(|| self.parse_err(at, format!("`\\x{{{digits}}}` is not a scalar value")))
    }

    /// Parses exactly four hex digits after `\u` — a literal-char spelling.
    fn parse_u_hex4(&mut self, at: usize) -> Result<char, RegexError> {
        let mut digits = String::new();
        for _ in 0..4 {
            match self.bump() {
                Some(c) if c.is_ascii_hexdigit() => digits.push(c),
                _ => return Err(self.parse_err(at, "`\\u` requires exactly four hex digits")),
            }
        }
        let value = u32::from_str_radix(&digits, 16).expect("four hex digits");
        char::from_u32(value)
            .ok_or_else(|| self.parse_err(at, format!("`\\u{digits}` is not a scalar value")))
    }
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Inst {
    /// Consume one char equal to `c` (case-insensitively if `ci`).
    Char {
        c: char,
        ci: bool,
    },
    /// Consume one char matching `classes[idx]`.
    Class {
        idx: usize,
        ci: bool,
    },
    /// Consume any char except `\n`.
    Any,
    /// Try `prefer` first; on backtrack, resume at `alt` (same position).
    Split {
        prefer: usize,
        alt: usize,
    },
    Jump(usize),
    /// Fail this thread if `lookaheads[prog]` matches at the position.
    NegLook {
        prog: usize,
    },
    Match,
}

/// A compiled program: the main instruction sequence plus the shared class
/// pool and the lookahead sub-programs it references.
struct Compiled {
    prog: Vec<Inst>,
    classes: Vec<ClassDef>,
    lookaheads: Vec<Vec<Inst>>,
}

struct Compiler<'p> {
    pattern: &'p str,
    insts: Vec<Inst>,
    classes: Vec<ClassDef>,
    lookaheads: Vec<Vec<Inst>>,
    total: usize,
}

impl<'p> Compiler<'p> {
    fn compile(pattern: &'p str, ast: &Ast) -> Result<Compiled, RegexError> {
        let mut compiler = Compiler {
            pattern,
            insts: Vec::new(),
            classes: Vec::new(),
            lookaheads: Vec::new(),
            total: 0,
        };
        compiler.compile_node(ast, false)?;
        compiler.emit(Inst::Match)?;
        Ok(Compiled {
            prog: compiler.insts,
            classes: compiler.classes,
            lookaheads: compiler.lookaheads,
        })
    }

    fn emit(&mut self, inst: Inst) -> Result<usize, RegexError> {
        self.total += 1;
        if self.total > MAX_PROG {
            return Err(RegexError::Parse {
                pattern: self.pattern.to_string(),
                at: 0,
                what: format!("compiled pattern exceeds {MAX_PROG} instructions"),
            });
        }
        self.insts.push(inst);
        Ok(self.insts.len() - 1)
    }

    fn patch_split(&mut self, at: usize, prefer: usize, alt: usize) {
        self.insts[at] = Inst::Split { prefer, alt };
    }

    fn compile_node(&mut self, ast: &Ast, ci: bool) -> Result<(), RegexError> {
        match ast {
            Ast::Empty => {}
            Ast::Char(c) => {
                self.emit(Inst::Char { c: *c, ci })?;
            }
            Ast::Any => {
                self.emit(Inst::Any)?;
            }
            Ast::Class(def) => {
                self.classes.push(def.clone());
                let idx = self.classes.len() - 1;
                self.emit(Inst::Class { idx, ci })?;
            }
            Ast::Concat(items) => {
                for item in items {
                    self.compile_node(item, ci)?;
                }
            }
            Ast::Alt(branches) => {
                let last = branches.len() - 1;
                let mut jumps = Vec::new();
                for (i, branch) in branches.iter().enumerate() {
                    if i < last {
                        let split = self.emit(Inst::Split { prefer: 0, alt: 0 })?;
                        self.compile_node(branch, ci)?;
                        jumps.push(self.emit(Inst::Jump(0))?);
                        let next_branch = self.insts.len();
                        self.patch_split(split, split + 1, next_branch);
                    } else {
                        self.compile_node(branch, ci)?;
                    }
                }
                let end = self.insts.len();
                for jump in jumps {
                    self.insts[jump] = Inst::Jump(end);
                }
            }
            Ast::Repeat {
                node,
                min,
                max,
                lazy,
            } => {
                for _ in 0..*min {
                    self.compile_node(node, ci)?;
                }
                match max {
                    None => {
                        // Trailing star: L: Split(body, out); body; Jump L.
                        let split = self.emit(Inst::Split { prefer: 0, alt: 0 })?;
                        self.compile_node(node, ci)?;
                        self.emit(Inst::Jump(split))?;
                        let end = self.insts.len();
                        let body = split + 1;
                        let (prefer, alt) = if *lazy { (end, body) } else { (body, end) };
                        self.patch_split(split, prefer, alt);
                    }
                    Some(n) => {
                        // n - min optional copies, each exiting to the end.
                        let mut splits = Vec::new();
                        for _ in *min..*n {
                            splits.push(self.emit(Inst::Split { prefer: 0, alt: 0 })?);
                            self.compile_node(node, ci)?;
                        }
                        let end = self.insts.len();
                        for split in splits {
                            let body = split + 1;
                            let (prefer, alt) = if *lazy { (end, body) } else { (body, end) };
                            self.patch_split(split, prefer, alt);
                        }
                    }
                }
            }
            Ast::Group(inner) => self.compile_node(inner, ci)?,
            Ast::CaseInsensitive(inner) => self.compile_node(inner, true)?,
            Ast::NegLookahead(inner) => {
                let saved = std::mem::take(&mut self.insts);
                self.compile_node(inner, ci)?;
                self.emit(Inst::Match)?;
                let sub = std::mem::replace(&mut self.insts, saved);
                self.lookaheads.push(sub);
                let prog = self.lookaheads.len() - 1;
                self.emit(Inst::NegLook { prog })?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Matcher
// ---------------------------------------------------------------------------

/// Single-char case folding via `char::to_lowercase`; chars whose lowercase
/// expands to more than one char (e.g. `İ`) are left unfolded — simple-fold
/// semantics, matching the reference's `(?i:…)` use over the ASCII corpus.
fn simple_lower(c: char) -> char {
    let mut lower = c.to_lowercase();
    match (lower.next(), lower.next()) {
        (Some(l), None) => l,
        _ => c,
    }
}

fn simple_upper(c: char) -> char {
    let mut upper = c.to_uppercase();
    match (upper.next(), upper.next()) {
        (Some(u), None) => u,
        _ => c,
    }
}

fn char_eq(input: char, wanted: char, ci: bool) -> bool {
    input == wanted || (ci && simple_lower(input) == simple_lower(wanted))
}

/// Internal marker: the step budget or backtrack-stack cap tripped.
struct BudgetTripped;

/// A compiled pattern over the corpus construct set.
///
/// Built once per artifact load ([`Regex::parse`]); matching borrows the
/// `&'static` [`PropertyLookup`] captured at parse time. `Send + Sync`
/// (the program is immutable data), so batch encoders can share it across
/// `std::thread::scope` threads.
#[derive(Clone)]
pub struct Regex {
    pattern: String,
    prog: Vec<Inst>,
    classes: Vec<ClassDef>,
    lookaheads: Vec<Vec<Inst>>,
    props: &'static dyn PropertyLookup,
}

impl fmt::Debug for Regex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Regex")
            .field("pattern", &self.pattern)
            .finish_non_exhaustive()
    }
}

impl Regex {
    /// Parses and compiles `pattern`, resolving `\p{…}`-class constructs
    /// through `props` at match time.
    ///
    /// Errors are loud and self-describing: [`RegexError::Parse`] for
    /// malformed patterns, [`RegexError::UnsupportedConstruct`] for
    /// syntactically valid constructs outside the corpus set.
    pub fn parse(pattern: &str, props: &'static dyn PropertyLookup) -> Result<Self, RegexError> {
        let ast = Parser::parse(pattern)?;
        let compiled = Compiler::compile(pattern, &ast)?;
        Ok(Regex {
            pattern: pattern.to_string(),
            prog: compiled.prog,
            classes: compiled.classes,
            lookaheads: compiled.lookaheads,
            props,
        })
    }

    /// The pattern text this regex was compiled from.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Iterates non-overlapping match spans as `(start, end)` byte offsets,
    /// leftmost-first. Yields `Err` once (then fuses) if the backtracking
    /// budget trips.
    pub fn find_iter(&self, text: &str) -> Matches<'_> {
        let chars: Vec<char> = text.chars().collect();
        let byte_of: Vec<usize> = text
            .char_indices()
            .map(|(b, _)| b)
            .chain(std::iter::once(text.len()))
            .collect();
        let budget = BUDGET_BASE.saturating_add(chars.len().saturating_mul(BUDGET_PER_CHAR));
        let stack_cap = STACK_BASE.saturating_add(chars.len().saturating_mul(STACK_PER_CHAR));
        Matches {
            re: self,
            chars,
            byte_of,
            scan: 0,
            budget,
            budget_total: budget,
            stack_cap,
            done: false,
        }
    }

    /// Whether the pattern matches anywhere in `text`.
    pub fn is_match(&self, text: &str) -> Result<bool, RegexError> {
        match self.find_iter(text).next() {
            None => Ok(false),
            Some(Ok(_)) => Ok(true),
            Some(Err(e)) => Err(e),
        }
    }

    /// Partitions `text` into the ordered, non-empty match/gap segments the
    /// `Split` pre-tokenizer behaviors fold over. Segments tile the input:
    /// consecutive `start..end` byte ranges are adjacent, and no segment is
    /// empty. Empty matches contribute no `Match` segment (they act as bare
    /// split points).
    pub fn segments(&self, text: &str) -> Result<Vec<Segment>, RegexError> {
        let mut segments = Vec::new();
        let mut last = 0usize;
        for span in self.find_iter(text) {
            let (start, end) = span?;
            if start > last {
                segments.push(Segment {
                    start: last,
                    end: start,
                    kind: SegmentKind::Gap,
                });
            }
            if end > start {
                segments.push(Segment {
                    start,
                    end,
                    kind: SegmentKind::Match,
                });
            }
            last = last.max(end);
        }
        if last < text.len() {
            segments.push(Segment {
                start: last,
                end: text.len(),
                kind: SegmentKind::Gap,
            });
        }
        Ok(segments)
    }

    fn class_matches(&self, def: &ClassDef, c: char, ci: bool) -> bool {
        let raw = |ch: char| {
            def.items.iter().any(|item| match *item {
                ClassItem::Char(x) => x == ch,
                ClassItem::Range(lo, hi) => (lo..=hi).contains(&ch),
                ClassItem::Prop(p) => self.props.is_class(ch, p),
                ClassItem::NegProp(p) => !self.props.is_class(ch, p),
            })
        };
        let mut inside = raw(c);
        if !inside && ci {
            let lower = simple_lower(c);
            let upper = simple_upper(c);
            inside = (lower != c && raw(lower)) || (upper != c && raw(upper));
        }
        inside != def.negated
    }

    /// Runs `insts` anchored at `start`. Returns the end position of the
    /// preference-ordered first match, `None` on mismatch, or the budget
    /// marker. Recursion depth equals lookahead nesting (bounded by
    /// [`MAX_NEST`]); the backtrack stack lives on the heap.
    fn run(
        &self,
        insts: &[Inst],
        chars: &[char],
        start: usize,
        budget: &mut usize,
        stack_cap: usize,
    ) -> Result<Option<usize>, BudgetTripped> {
        let mut stack: Vec<(usize, usize)> = Vec::new();
        let mut pc = 0usize;
        let mut pos = start;
        loop {
            if *budget == 0 {
                return Err(BudgetTripped);
            }
            *budget -= 1;
            let next = match insts[pc] {
                Inst::Match => return Ok(Some(pos)),
                Inst::Char { c, ci } => {
                    (pos < chars.len() && char_eq(chars[pos], c, ci)).then_some((pc + 1, pos + 1))
                }
                Inst::Any => (pos < chars.len() && chars[pos] != '\n').then_some((pc + 1, pos + 1)),
                Inst::Class { idx, ci } => (pos < chars.len()
                    && self.class_matches(&self.classes[idx], chars[pos], ci))
                .then_some((pc + 1, pos + 1)),
                Inst::Jump(target) => Some((target, pos)),
                Inst::Split { prefer, alt } => {
                    if stack.len() >= stack_cap {
                        return Err(BudgetTripped);
                    }
                    stack.push((alt, pos));
                    Some((prefer, pos))
                }
                Inst::NegLook { prog } => {
                    match self.run(&self.lookaheads[prog], chars, pos, budget, stack_cap)? {
                        Some(_) => None, // lookahead matched: this thread fails
                        None => Some((pc + 1, pos)),
                    }
                }
            };
            match next.or_else(|| stack.pop()) {
                Some((p, q)) => {
                    pc = p;
                    pos = q;
                }
                None => return Ok(None),
            }
        }
    }
}

/// Iterator over non-overlapping match spans; see [`Regex::find_iter`].
pub struct Matches<'r> {
    re: &'r Regex,
    chars: Vec<char>,
    byte_of: Vec<usize>,
    scan: usize,
    budget: usize,
    budget_total: usize,
    stack_cap: usize,
    done: bool,
}

impl Iterator for Matches<'_> {
    type Item = Result<(usize, usize), RegexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        while self.scan <= self.chars.len() {
            let start = self.scan;
            match self.re.run(
                &self.re.prog,
                &self.chars,
                start,
                &mut self.budget,
                self.stack_cap,
            ) {
                Err(BudgetTripped) => {
                    self.done = true;
                    return Some(Err(RegexError::Budget {
                        pattern: self.re.pattern.clone(),
                        steps: self.budget_total,
                        input_chars: self.chars.len(),
                    }));
                }
                Ok(Some(end)) => {
                    self.scan = if end == start { start + 1 } else { end };
                    return Some(Ok((self.byte_of[start], self.byte_of[end])));
                }
                Ok(None) => self.scan += 1,
            }
        }
        self.done = true;
        None
    }
}

impl FusedIterator for Matches<'_> {}

//! Construct-by-construct vectors for the split-regex engine
//! (TOKENIZER_SCOPE §6), including the real corpus patterns, run against a
//! stub [`PropertyLookup`].
//!
//! The stub answers property queries for exactly the characters these tests
//! use (full ASCII, plus a handful of enumerated non-ASCII chars). Where
//! stub != real tables matters: the real `\w` also covers marks and
//! non-ASCII connector punctuation, real `White_Space` covers the U+2000
//! block, and real `L`/`N` cover all of Unicode — so the corpus-pattern
//! vectors below deliberately use ASCII-only sample text, where the stub
//! and the real tables agree exactly. The fixture-differential layer (§9)
//! re-pins these patterns against reference split output with the real
//! tables at integration.

use quanta_tokenizers::regex::{
    PropClass, PropertyLookup, Regex, RegexError, Segment, SegmentKind,
};

struct Stub;

impl PropertyLookup for Stub {
    fn is_class(&self, c: char, class: PropClass) -> bool {
        match class {
            PropClass::Letter => {
                c.is_ascii_alphabetic() || matches!(c, 'é' | 'É' | 'ü' | '中' | '文' | 'α')
            }
            PropClass::Number => c.is_ascii_digit() || matches!(c, '٣' | '½'),
            PropClass::DecimalDigit => c.is_ascii_digit() || c == '٣',
            PropClass::Whitespace => {
                matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{a0}')
            }
            PropClass::Word => {
                c.is_ascii_alphanumeric()
                    || c == '_'
                    || self.is_class(c, PropClass::Letter)
                    || self.is_class(c, PropClass::DecimalDigit)
            }
            PropClass::Punctuation => matches!(
                c,
                '.' | ','
                    | '!'
                    | '?'
                    | ';'
                    | ':'
                    | '\''
                    | '"'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '-'
                    | '_'
                    | '/'
                    | '\\'
                    | '@'
                    | '#'
                    | '%'
                    | '&'
                    | '*'
            ),
            PropClass::Symbol => {
                matches!(c, '+' | '<' | '>' | '=' | '$' | '^' | '`' | '|' | '~')
            }
            PropClass::Mark | PropClass::NonspacingMark => matches!(c, '\u{300}' | '\u{301}'),
            PropClass::Control => (c as u32) < 0x20 || c == '\u{7f}',
        }
    }
}

static STUB: Stub = Stub;

fn rx(pattern: &str) -> Regex {
    Regex::parse(pattern, &STUB).expect("pattern should parse")
}

fn spans(pattern: &str, text: &str) -> Vec<(usize, usize)> {
    rx(pattern)
        .find_iter(text)
        .collect::<Result<_, _>>()
        .expect("no budget trip")
}

fn toks<'t>(pattern: &str, text: &'t str) -> Vec<&'t str> {
    spans(pattern, text)
        .iter()
        .map(|&(s, e)| &text[s..e])
        .collect()
}

fn seg(start: usize, end: usize, kind: SegmentKind) -> Segment {
    Segment { start, end, kind }
}

// -- construct rows ---------------------------------------------------------

#[test]
fn literals_and_concatenation() {
    assert_eq!(spans("abc", "xxabcabc"), vec![(2, 5), (5, 8)]);
    assert_eq!(spans(r"a\.b", "a.b axb"), vec![(0, 3)]);
    assert!(spans("abc", "abd").is_empty());
}

#[test]
fn dot_matches_any_char_except_newline() {
    assert_eq!(spans(".", "a\nb"), vec![(0, 1), (2, 3)]);
    assert_eq!(spans("a.c", "a中c"), vec![(0, 5)]);
}

#[test]
fn class_ranges_and_negation() {
    assert_eq!(spans("[a-cx]+", "abxyc"), vec![(0, 3), (4, 5)]);
    assert_eq!(spans("[^a-c]+", "abdea"), vec![(2, 4)]);
}

#[test]
fn class_escapes_and_literal_dash_and_bracket() {
    assert_eq!(spans(r"[\r\n\t]+", "a\r\n\tb"), vec![(1, 4)]);
    assert_eq!(spans(r"[-a]+", "-a-b"), vec![(0, 3)]);
    assert_eq!(spans(r"[a-]+", "a-b"), vec![(0, 2)]);
    assert_eq!(spans(r"[a\]]+", "a]b"), vec![(0, 2)]);
    assert_eq!(spans(r"[\-\]]+", "-]x"), vec![(0, 2)]);
}

#[test]
fn class_property_items() {
    assert_eq!(spans(r"[\p{L}\p{N}]+", "ab12!cd"), vec![(0, 4), (5, 7)]);
    // The GPT-2 pattern's negated class: not whitespace, not letter, not number.
    assert_eq!(spans(r"[^\s\p{L}\p{N}]+", "a! .b"), vec![(1, 2), (3, 4)]);
    assert_eq!(spans(r"[\w]+", "so_x!y"), vec![(0, 4), (5, 6)]);
}

#[test]
fn perl_classes_and_negations() {
    assert_eq!(spans(r"\d+", "ab12cd3"), vec![(2, 4), (6, 7)]);
    assert_eq!(spans(r"\D+", "12ab3"), vec![(2, 4)]);
    assert_eq!(spans(r"\s+", "a b"), vec![(1, 2)]);
    assert_eq!(spans(r"\S+", " ab "), vec![(1, 3)]);
    assert_eq!(spans(r"\w+", "hi_there!x"), vec![(0, 8), (9, 10)]);
    assert_eq!(spans(r"\W+", "ab!?cd"), vec![(2, 4)]);
}

#[test]
fn unicode_property_classes() {
    // Byte offsets stay unicode-correct: é is 2 bytes, 中/文 are 3.
    assert_eq!(spans(r"\p{L}+", "héllo 中文"), vec![(0, 6), (7, 13)]);
    assert_eq!(spans(r"\P{L}+", "ab12"), vec![(2, 4)]);
    assert_eq!(spans(r"\p{L}\p{Mn}", "e\u{301}"), vec![(0, 3)]);
    assert_eq!(spans(r"\p{Nd}+", "x42"), vec![(1, 3)]);
}

#[test]
fn alternation_is_leftmost_first_not_longest() {
    // Ordered preference, the Oniguruma/PCRE semantics the corpus relies on.
    assert_eq!(spans("ab|a", "ab"), vec![(0, 2)]);
    assert_eq!(spans("a|ab", "ab"), vec![(0, 1)]);
}

#[test]
fn groups_plain_and_non_capturing() {
    // Plain groups parse; captures are never exposed (whole-match spans only).
    assert_eq!(spans("(ab)+", "ababx"), vec![(0, 4)]);
    assert_eq!(spans("(?:ab)+c", "ababc"), vec![(0, 5)]);
}

#[test]
fn case_insensitive_group() {
    assert!(rx("(?i:hello)").is_match("xxHeLLoyy").unwrap());
    assert_eq!(spans("(?i:l+)", "LlL"), vec![(0, 3)]);
    assert_eq!(spans("(?i:[sd])", "SD"), vec![(0, 1), (1, 2)]);
    // The flag is scoped to the group, not the whole pattern.
    assert!(rx("a(?i:b)c").is_match("aBc").unwrap());
    assert!(!rx("a(?i:b)c").is_match("Abc").unwrap());
    assert!(!rx("a(?i:b)c").is_match("abC").unwrap());
    // Non-ASCII simple folding via std's case mapping.
    assert!(rx("(?i:é)").is_match("É").unwrap());
}

#[test]
fn case_insensitive_negated_class() {
    // [^a] under (?i:) excludes the whole case-closure {a, A}.
    assert!(!rx("(?i:[^a])").is_match("A").unwrap());
    assert!(rx("(?i:[^a])").is_match("b").unwrap());
}

#[test]
fn greedy_versus_lazy_quantifiers() {
    assert_eq!(spans("a+", "aaa"), vec![(0, 3)]);
    assert_eq!(spans("a+?", "aaa"), vec![(0, 1), (1, 2), (2, 3)]);
    assert_eq!(spans("a*?", "aa"), vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
fn counted_quantifiers() {
    assert_eq!(spans("a{2}", "aaaa"), vec![(0, 2), (2, 4)]);
    assert_eq!(spans("a{2,}", "aaaaa"), vec![(0, 5)]);
    assert_eq!(spans("a{2,3}", "aaaaa"), vec![(0, 3), (3, 5)]);
    assert_eq!(spans("a{2,3}?", "aaaa"), vec![(0, 2), (2, 4)]);
    // Nullable pattern: like a*, it also matches empty at end-of-input.
    assert_eq!(spans("a{0,2}", "aaa"), vec![(0, 2), (2, 3), (3, 3)]);
}

#[test]
fn malformed_braces_are_literal_like_oniguruma() {
    assert_eq!(spans(r"a\{x}", "za{x}z"), vec![(1, 5)]);
    assert_eq!(spans("a{x}", "za{x}z"), vec![(1, 5)]);
    assert_eq!(spans("a{,3}", "a{,3}"), vec![(0, 5)]);
    assert_eq!(spans("{2}", "x{2}"), vec![(1, 4)]);
}

#[test]
fn negative_lookahead() {
    assert_eq!(spans("a(?!b)", "ab ac"), vec![(3, 4)]);
    // Lookahead failure drives backtracking into the quantifier.
    assert_eq!(spans("a+(?!b)", "aab"), vec![(0, 1)]);
    // The GPT-2 idiom: all-but-the-last whitespace of a run.
    assert_eq!(spans(r"\s+(?!\S)", "ab   cd"), vec![(2, 4)]);
}

#[test]
fn empty_matches_advance_the_scan() {
    assert_eq!(spans("a*", "bb"), vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(spans("a*", ""), vec![(0, 0)]);
    assert!(spans("a+", "").is_empty());
}

#[test]
fn hex_and_unicode_literal_escapes() {
    assert_eq!(spans(r"\x{4E2D}", "一中"), vec![(3, 6)]);
    assert_eq!(spans("\\u0041+", "AAB"), vec![(0, 2)]);
    assert_eq!(spans(r"[\x{30}-\x{39}]+", "a123"), vec![(1, 4)]);
}

#[test]
fn multibyte_spans_are_byte_offsets_on_char_boundaries() {
    assert_eq!(spans("中", "中文中"), vec![(0, 3), (6, 9)]);
    assert_eq!(spans("[à-ÿ]+", "être"), vec![(0, 2)]);
}

// -- the real corpus patterns ----------------------------------------------

/// The GPT-2 / RoBERTa split pattern, as shipped in `gpt2`'s
/// `tokenizer.json` (`ByteLevel` pre-tokenizer, `use_regex: true`).
const GPT2_SPLIT: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// The cl100k-family split pattern, as shipped in Llama-3-class
/// `tokenizer.json` artifacts (`Split` pre-tokenizer with a Regex pattern).
const CL100K_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The HF `Whitespace` pre-tokenizer pattern.
const WHITESPACE_SPLIT: &str = r"\w+|[^\w\s]+";

#[test]
fn corpus_gpt2_split_pattern() {
    // ASCII-only sample text: stub == real tables here (see module docs).
    assert_eq!(
        toks(GPT2_SPLIT, "Hello world!  I'm 42."),
        vec!["Hello", " world", "!", " ", " I", "'m", " 42", "."],
    );
    assert_eq!(toks(GPT2_SPLIT, "hey\nyou"), vec!["hey", "\n", "you"]);
    // A space run keeps its last space for the following word.
    assert_eq!(toks(GPT2_SPLIT, "a   b"), vec!["a", "  ", " b"]);
}

#[test]
fn corpus_cl100k_llama3_split_pattern() {
    assert_eq!(
        toks(CL100K_SPLIT, "It's  1234 OK!!\r\n"),
        vec!["It", "'s", " ", " ", "123", "4", " OK", "!!\r\n"],
    );
    // The (?i:) group: uppercase contractions match too.
    assert_eq!(toks(CL100K_SPLIT, "HE'S"), vec!["HE", "'S"]);
    // Newline runs ride the \s*[\r\n]+ alternate.
    assert_eq!(toks(CL100K_SPLIT, "hi\n\nyo"), vec!["hi", "\n\n", "yo"]);
}

#[test]
fn corpus_whitespace_pretokenizer_pattern() {
    assert_eq!(
        toks(WHITESPACE_SPLIT, "Hey, friend!"),
        vec!["Hey", ",", "friend", "!"]
    );
    assert_eq!(
        rx(WHITESPACE_SPLIT).segments("Hey, friend!").unwrap(),
        vec![
            seg(0, 3, SegmentKind::Match),
            seg(3, 4, SegmentKind::Match),
            seg(4, 5, SegmentKind::Gap),
            seg(5, 11, SegmentKind::Match),
            seg(11, 12, SegmentKind::Match),
        ],
    );
}

// -- the Split-serving segment stream ---------------------------------------

#[test]
fn segments_tile_the_input_with_no_empty_segments() {
    let text = "  the quick  brown ";
    assert_eq!(
        rx(r"\s+").segments(text).unwrap(),
        vec![
            seg(0, 2, SegmentKind::Match),
            seg(2, 5, SegmentKind::Gap),
            seg(5, 6, SegmentKind::Match),
            seg(6, 11, SegmentKind::Gap),
            seg(11, 13, SegmentKind::Match),
            seg(13, 18, SegmentKind::Gap),
            seg(18, 19, SegmentKind::Match),
        ],
    );
}

#[test]
fn segments_adjacent_matches_produce_no_empty_gap() {
    assert_eq!(
        rx(r"\d").segments("12a3").unwrap(),
        vec![
            seg(0, 1, SegmentKind::Match),
            seg(1, 2, SegmentKind::Match),
            seg(2, 3, SegmentKind::Gap),
            seg(3, 4, SegmentKind::Match),
        ],
    );
}

#[test]
fn segments_empty_matches_act_as_split_points() {
    assert_eq!(
        rx("a*").segments("bb").unwrap(),
        vec![seg(0, 1, SegmentKind::Gap), seg(1, 2, SegmentKind::Gap)],
    );
}

#[test]
fn segments_serve_the_split_behaviors() {
    // The five SplitDelimiterBehavior modes are folds over the segment
    // stream; pin the two ends of the spectrum here as the seam's contract.
    let text = "  the quick  brown ";
    let segments = rx(r"\s+").segments(text).unwrap();
    let removed: Vec<&str> = segments
        .iter()
        .filter(|s| s.kind == SegmentKind::Gap)
        .map(|s| &text[s.start..s.end])
        .collect();
    assert_eq!(removed, vec!["the", "quick", "brown"]);
    let isolated: Vec<&str> = segments.iter().map(|s| &text[s.start..s.end]).collect();
    assert_eq!(
        isolated,
        vec!["  ", "the", " ", "quick", "  ", "brown", " "]
    );
}

#[test]
fn is_match_and_pattern_accessor() {
    let re = rx(r"\d+");
    assert!(re.is_match("abc123").unwrap());
    assert!(!re.is_match("abcdef").unwrap());
    assert_eq!(re.pattern(), r"\d+");
}

#[test]
fn regex_is_send_and_sync() {
    // The seam promise: a &'static PropertyLookup keeps Regex shareable
    // across std::thread::scope threads for batch encoding.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Regex>();
}

// -- the backtracking budget ------------------------------------------------

#[test]
fn budget_trips_loudly_on_pathological_backtracking() {
    // The classic exponential blowup: must error, not hang.
    let re = rx("(a+)+b");
    let text = "a".repeat(30);
    let err = re
        .find_iter(&text)
        .next()
        .expect("should yield an error")
        .unwrap_err();
    assert!(matches!(err, RegexError::Budget { .. }), "{err:?}");
    let msg = err.to_string();
    assert!(msg.contains("budget"), "{msg}");
    assert!(msg.contains("(a+)+b"), "{msg}");
    assert!(matches!(re.is_match(&text), Err(RegexError::Budget { .. })));
    assert!(matches!(re.segments(&text), Err(RegexError::Budget { .. })));
    // The iterator fuses after the error.
    let mut it = re.find_iter(&text);
    assert!(it.next().is_some());
    assert!(it.next().is_none());
}

#[test]
fn budget_admits_honest_patterns_on_long_input() {
    // The input-linear budget is sized so corpus patterns never graze it.
    let text = "word ".repeat(4000);
    let count = rx(GPT2_SPLIT).find_iter(&text).count();
    assert_eq!(count, 4001); // "word", 3999 x " word", the trailing " "
}

// -- the claim boundary -----------------------------------------------------

#[test]
fn constructs_outside_the_set_are_rejected_loudly() {
    let cases: &[(&str, &str)] = &[
        (r"\1", "backreference"),
        (r"(?<=a)b", "lookbehind"),
        (r"(?<!a)b", "lookbehind"),
        (r"(?=a)b", "positive lookahead"),
        (r"a++", "possessive"),
        (r"a*+", "possessive"),
        (r"a?+", "possessive"),
        (r"(?i)abc", "inline flag"),
        (r"^a", "anchor"),
        (r"a$", "anchor"),
        (r"\A", "anchor"),
        (r"\bx\b", "word boundary"),
        (r"\p{Greek}", "unicode property"),
        (r"[[:alpha:]]", "POSIX"),
        (r"(?P<n>a)", "named group"),
        (r"(?<name>a)", "named group"),
        (r"(?>a)", "atomic group"),
        (r"a(?#c)", "comment group"),
        (r"\e", "escape"),
        (r"[\b]", "character class"),
        (r"(?:a?)*", "empty string"),
        (r"(?:a|)+", "empty string"),
    ];
    for (pattern, needle) in cases {
        let err = Regex::parse(pattern, &STUB).expect_err(pattern);
        assert!(
            matches!(err, RegexError::UnsupportedConstruct { .. }),
            "`{pattern}` should be UnsupportedConstruct, got {err:?}",
        );
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "`{pattern}` message missing `{needle}`: {msg}"
        );
        assert!(
            msg.contains(pattern),
            "`{pattern}` message missing the pattern: {msg}"
        );
    }
}

#[test]
fn malformed_patterns_error_with_context() {
    let cases: &[(&str, &str)] = &[
        (r"a{3,2}", "reversed repetition"),
        (r"[z-a]", "reversed range"),
        (r"[abc", "unterminated character class"),
        (r"(ab", "unterminated group"),
        (r"*a", "nothing to repeat"),
        (r"[]", "empty character class"),
        ("a\\", "dangling"),
        (r"a{600}", "exceeds"),
        (r")", "unmatched"),
        (r"[\d-x]", "invalid range"),
        (r"\p{L", "unterminated"),
        (r"\pL", "requires"),
        (r"\x41", "requires"),
        (r"\u12", "four hex digits"),
        (r"a**", "nested quantifier"),
        (r"a{2}{3}", "nested quantifier"),
    ];
    for (pattern, needle) in cases {
        let err = Regex::parse(pattern, &STUB).expect_err(pattern);
        assert!(
            matches!(err, RegexError::Parse { .. }),
            "`{pattern}` gave {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "`{pattern}` message missing `{needle}`: {msg}"
        );
    }
    let deep = format!("{}a{}", "(".repeat(70), ")".repeat(70));
    let err = Regex::parse(&deep, &STUB).expect_err("deep nesting");
    assert!(err.to_string().contains("nesting"), "{err}");
}

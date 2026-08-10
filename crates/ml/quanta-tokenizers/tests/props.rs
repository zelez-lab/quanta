//! The property bridge, wired: the corpus patterns re-pinned against
//! the REAL vendored tables (the regex lane's own suite runs on a
//! stub and keeps its vectors ASCII; these vectors are exactly the
//! non-ASCII cases where stub and tables can differ).

use quanta_tokenizers::props::TABLES;
use quanta_tokenizers::regex::Regex;

const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn splits(re: &Regex, text: &str) -> Vec<String> {
    re.find_iter(text)
        .map(|r| {
            let (a, b) = r.expect("match");
            text[a..b].to_string()
        })
        .collect()
}

#[test]
fn gpt2_pattern_over_real_tables_handles_non_ascii() {
    let re = Regex::parse(GPT2_PATTERN, &TABLES).unwrap();
    // é is \p{L} only if the real Letter table says so; the combining
    // mark form exercises \p{L}+ NOT matching the Mn codepoint (it
    // falls to the catch-all class, like the reference).
    assert_eq!(splits(&re, "café au lait"), vec!["café", " au", " lait"]);
    assert_eq!(splits(&re, "π≈3.14"), vec!["π", "≈", "3", ".", "14"]);
    // Devanagari digits are \p{N} but not \d-ASCII — the real Number
    // table must claim them.
    assert_eq!(splits(&re, "a १२३"), vec!["a", " १२३"]);
}

#[test]
fn word_class_follows_the_reference_definition() {
    let re = Regex::parse(r"\w+", &TABLES).unwrap();
    // Letters + marks + decimal digits + connector punctuation (Pc):
    // underscore joins, the combining acute (U+0301) joins, the em
    // dash does not.
    assert_eq!(
        splits(&re, "foo_bar1 e\u{0301}x — y"),
        vec!["foo_bar1", "e\u{0301}x", "y"]
    );
}

#[test]
fn nonspacing_mark_class_is_mn_exactly() {
    let re = Regex::parse(r"\p{Mn}+", &TABLES).unwrap();
    // U+0301 is Mn; U+0903 DEVANAGARI SIGN VISARGA is Mc (a mark, but
    // spacing) and must NOT match.
    assert_eq!(splits(&re, "e\u{0301}a\u{0903}"), vec!["\u{0301}"]);
}

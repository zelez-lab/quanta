//! RFC 8259 edge vectors for the in-crate JSON parser: the full value
//! model, string escapes including surrogate pairs, the closed number
//! grammar, the depth cap, duplicate-key rejection, byte-offset error
//! reporting, and truncation fuzz at every byte of a small document
//! (the npy truncation-fuzz pattern).

use quanta_tokenizers::TokenizerError;
use quanta_tokenizers::json::{self, MAX_DEPTH, Number, Value};

fn parse(s: &str) -> Result<Value, TokenizerError> {
    json::parse(s.as_bytes())
}

fn ok(s: &str) -> Value {
    parse(s).unwrap_or_else(|e| panic!("{s:?} must parse, got: {e}"))
}

/// The error and its byte offset, panicking on accidental success.
fn err(s: &str) -> (usize, String) {
    match parse(s) {
        Err(TokenizerError::Json { at, what }) => (at, what),
        Err(other) => panic!("{s:?} must fail with Json, got: {other}"),
        Ok(v) => panic!("{s:?} must fail, parsed to {v:?}"),
    }
}

fn num(s: &str) -> Number {
    match ok(s) {
        Value::Number(n) => n,
        v => panic!("{s:?} must be a number, got {v:?}"),
    }
}

fn string(s: &str) -> String {
    match ok(s) {
        Value::String(t) => t,
        v => panic!("{s:?} must be a string, got {v:?}"),
    }
}

// ── The value model ─────────────────────────────────────────────────────

#[test]
fn literals() {
    assert_eq!(ok("null"), Value::Null);
    assert_eq!(ok("true"), Value::Bool(true));
    assert_eq!(ok("false"), Value::Bool(false));
    assert_eq!(ok("  null \t\r\n"), Value::Null);
}

#[test]
fn misspelled_literals_are_loud() {
    err("nul");
    err("tru");
    err("falsy");
    err("Null");
    err("NaN");
    err("Infinity");
}

#[test]
fn empty_and_whitespace_only_inputs() {
    err("");
    err("   ");
    err("\t\n\r ");
}

#[test]
fn containers() {
    assert_eq!(ok("[]"), Value::Array(vec![]));
    assert_eq!(ok("{}"), Value::Object(vec![]));
    assert_eq!(
        ok(r#"[1, "a", null, true, [], {}]"#),
        Value::Array(vec![
            Value::Number(Number {
                value: 1.0,
                raw: "1".to_string()
            }),
            Value::String("a".to_string()),
            Value::Null,
            Value::Bool(true),
            Value::Array(vec![]),
            Value::Object(vec![]),
        ])
    );
}

#[test]
fn objects_preserve_source_order() {
    let v = ok(r#"{"z": 1, "a": 2, "m": 3}"#);
    let keys: Vec<&str> = v
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(keys, ["z", "a", "m"]);
}

#[test]
fn object_lookup() {
    let v = ok(r#"{"a": {"b": [10]}}"#);
    let inner = v.get("a").unwrap().get("b").unwrap();
    assert_eq!(inner.as_array().unwrap().len(), 1);
    assert!(v.get("missing").is_none());
    assert!(Value::Null.get("a").is_none());
}

#[test]
fn empty_string_key_is_a_key() {
    let v = ok(r#"{"": 1}"#);
    assert!(v.get("").is_some());
}

// ── Structural errors ───────────────────────────────────────────────────

#[test]
fn structural_faults_are_loud() {
    err("[1,]");
    err("[1 2]");
    err("[,1]");
    err("[1");
    err("{\"a\":1");
    err("{\"a\" 1}");
    err("{\"a\":}");
    err("{a: 1}");
    err("{1: 2}");
    err("{\"a\":1,}");
    err("}");
    err("]");
    err(":");
}

#[test]
fn trailing_data_is_loud() {
    err("null null");
    err("1 2");
    err("{} {}");
    err("truex");
    err("[] //comment");
}

#[test]
fn error_offsets_are_byte_accurate() {
    // {"a":x} — the bad value starts at byte 5.
    let (at, _) = err(r#"{"a":x}"#);
    assert_eq!(at, 5);
    // Trailing data after the top value.
    let (at, what) = err("null 1");
    assert_eq!(at, 5);
    assert!(what.contains("trailing"), "{what}");
}

#[test]
fn duplicate_keys_are_rejected_and_named() {
    let (at, what) = err(r#"{"tok": 1, "tok": 2}"#);
    assert!(what.contains("duplicate key"), "{what}");
    assert!(what.contains("\"tok\""), "{what}");
    assert_eq!(at, 11); // the second "tok"
    // Nested objects each get their own key space.
    ok(r#"{"a": {"x": 1}, "b": {"x": 2}}"#);
}

#[test]
fn error_display_carries_the_offset() {
    let e = json::parse(b"[1,]").unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("at byte 3"), "{msg}");
    assert!(msg.starts_with("tokenizer.json:"), "{msg}");
}

// ── Depth cap ───────────────────────────────────────────────────────────

#[test]
fn depth_cap_allows_max_depth_and_rejects_one_more() {
    let deep_ok = format!("{}null{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
    ok(&deep_ok);
    let deep_bad = format!(
        "{}null{}",
        "[".repeat(MAX_DEPTH + 1),
        "]".repeat(MAX_DEPTH + 1)
    );
    let (_, what) = err(&deep_bad);
    assert!(what.contains("nesting"), "{what}");
    // Objects count toward the same cap.
    let mixed = format!(
        "{}{{\"k\": 1}}{}",
        "[".repeat(MAX_DEPTH),
        "]".repeat(MAX_DEPTH)
    );
    err(&mixed);
}

#[test]
fn hostile_deep_nesting_does_not_blow_the_stack() {
    // Far past the cap — must return an error, not crash.
    let hostile = "[".repeat(100_000);
    assert!(json::parse(hostile.as_bytes()).is_err());
}

// ── Strings ─────────────────────────────────────────────────────────────

#[test]
fn simple_escapes() {
    assert_eq!(
        string(r#""\" \\ \/ \b \f \n \r \t""#),
        "\" \\ / \u{8} \u{c} \n \r \t"
    );
}

#[test]
fn unicode_escapes() {
    assert_eq!(string(r#""\u0041""#), "A");
    assert_eq!(string(r#""\u00e9""#), "\u{e9}");
    assert_eq!(string(r#""\u2581""#), "\u{2581}"); // the Metaspace marker
    assert_eq!(string(r#""\u0000""#), "\u{0}");
    assert_eq!(string(r#""\uFFFD""#), "\u{FFFD}");
}

#[test]
fn surrogate_pairs_decode_to_astral_codepoints() {
    assert_eq!(string(r#""\uD83D\uDE00""#), "\u{1F600}"); // emoji
    assert_eq!(string(r#""\uD800\uDC00""#), "\u{10000}"); // lowest astral
    assert_eq!(string(r#""\uDBFF\uDFFF""#), "\u{10FFFF}"); // highest
    assert_eq!(string(r#""x\uD83D\uDE00y""#), "x\u{1F600}y");
}

#[test]
fn lone_surrogates_are_rejected() {
    let (_, what) = err(r#""\uD800""#);
    assert!(what.contains("lone high surrogate"), "{what}");
    let (_, what) = err(r#""\uDC00""#);
    assert!(what.contains("lone low surrogate"), "{what}");
    // High surrogate followed by a non-escape.
    err(r#""\uD800x""#);
    // High surrogate followed by a non-surrogate escape.
    let (_, what) = err(r#""\uD800\u0041""#);
    assert!(what.contains("not a low surrogate"), "{what}");
    // Two high surrogates.
    err(r#""\uD800\uD800""#);
}

#[test]
fn bad_escapes_are_loud() {
    err(r#""\x41""#);
    err(r#""\u00""#); // truncated
    err(r#""\u00gg""#); // non-hex
    err("\"\\"); // dangling at EOF
    err(r#""\"#);
}

#[test]
fn raw_utf8_passes_through() {
    assert_eq!(string("\"héllo wörld\""), "héllo wörld");
    assert_eq!(string("\"日本語\""), "日本語");
    assert_eq!(string("\"🚀\""), "🚀"); // raw astral, no escapes
    assert_eq!(string("\"Ġhello\""), "Ġhello"); // the ByteLevel marker char
}

#[test]
fn invalid_utf8_is_rejected() {
    // Bare continuation byte.
    assert!(json::parse(b"\"\x80\"").is_err());
    // Overlong encoding of '/' (0xC0 0xAF).
    assert!(json::parse(b"\"\xC0\xAF\"").is_err());
    // Truncated 3-byte sequence.
    assert!(json::parse(b"\"\xE2\x82\"").is_err());
    // 0xF5 lead (past U+10FFFF).
    assert!(json::parse(b"\"\xF5\x80\x80\x80\"").is_err());
    // UTF-8-encoded surrogate (0xED 0xA0 0x80 = U+D800).
    assert!(json::parse(b"\"\xED\xA0\x80\"").is_err());
}

#[test]
fn unescaped_control_characters_are_rejected() {
    assert!(json::parse(b"\"a\x01b\"").is_err());
    assert!(json::parse(b"\"a\nb\"").is_err());
    assert!(json::parse(b"\"a\tb\"").is_err());
}

#[test]
fn unterminated_string_is_loud() {
    let (_, what) = err("\"abc");
    assert!(what.contains("unterminated"), "{what}");
}

// ── Numbers ─────────────────────────────────────────────────────────────

#[test]
fn integer_numbers() {
    assert_eq!(num("0").value, 0.0);
    assert_eq!(num("-0").value, 0.0);
    assert!(num("-0").value.is_sign_negative());
    assert_eq!(num("42").value, 42.0);
    assert_eq!(num("-7").value, -7.0);
    assert_eq!(num("4294967295").value, 4294967295.0);
}

#[test]
fn float_numbers() {
    assert_eq!(num("1.5").value, 1.5);
    assert_eq!(num("-13.629").value, -13.629);
    assert_eq!(num("1e3").value, 1000.0);
    assert_eq!(num("1E3").value, 1000.0);
    assert_eq!(num("1e+3").value, 1000.0);
    assert_eq!(num("-1.5e-3").value, -0.0015);
    assert_eq!(num("0.0001").value, 0.0001);
    // Correct rounding is std's (documented delegation).
    assert_eq!(num("3.141592653589793").value, std::f64::consts::PI);
}

#[test]
fn number_raw_text_is_preserved() {
    assert_eq!(num("-1.5e-3").raw, "-1.5e-3");
    assert_eq!(num("50256").raw, "50256");
    assert_eq!(num("-0").raw, "-0");
}

#[test]
fn exact_integer_accessors() {
    assert_eq!(num("50256").as_u32(), Some(50256));
    assert_eq!(num("0").as_u32(), Some(0));
    assert_eq!(num("4294967295").as_u32(), Some(u32::MAX));
    assert_eq!(num("4294967296").as_u32(), None); // 2^32
    assert_eq!(num("4294967296").as_u64(), Some(1 << 32));
    // Non-integer spellings never masquerade as ids.
    assert_eq!(num("1.0").as_u64(), None);
    assert_eq!(num("1e2").as_u64(), None);
    assert_eq!(num("-1").as_u64(), None);
    // Beyond u64 is not an id either.
    assert_eq!(num("18446744073709551616").as_u64(), None);
}

#[test]
fn malformed_numbers_are_loud() {
    err("01"); // leading zero
    err("-01");
    err("+1"); // no leading plus in RFC 8259
    err(".5"); // no bare fraction
    err("5."); // fraction needs digits
    err("1.e5");
    err("1e"); // exponent needs digits
    err("1e+");
    err("-"); // sign alone
    err("--1");
    err("0x10"); // no hex
    err("1_000"); // no separators
}

#[test]
fn number_overflow_is_loud_underflow_rounds_to_zero() {
    let (_, what) = err("1e999");
    assert!(what.contains("overflows"), "{what}");
    err("-1e999");
    assert_eq!(num("1e-999").value, 0.0); // correctly-rounded underflow
}

// ── Truncation fuzz (the npy pattern) ───────────────────────────────────

#[test]
fn truncation_fuzz_every_prefix_errs_without_panic() {
    let doc = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"😀":1},"merges":[["a","a"]]},"logprob":-13.5e-2,"pad":null,"flags":[true,false]}"#.as_bytes();
    // Sanity: whole document parses (schema validity is not the point here).
    assert!(json::parse(doc).is_ok());
    for cut in 0..doc.len() {
        let prefix = &doc[..cut];
        match json::parse(prefix) {
            Err(TokenizerError::Json { at, .. }) => {
                assert!(at <= prefix.len(), "offset {at} past prefix length {cut}");
            }
            Err(other) => panic!("prefix {cut}: non-Json error {other}"),
            Ok(v) => panic!("prefix {cut} must not parse, got {v:?}"),
        }
    }
}

#[test]
fn single_byte_corruption_never_panics() {
    // Flip each byte of a small valid document through a few hostile
    // values; parsing may succeed or fail, but must never panic.
    let doc = br#"{"a":[1,-2.5e3],"b":"Ax","c":null}"#;
    for i in 0..doc.len() {
        for &bad in &[0x00u8, 0x22, 0x5C, 0x7B, 0x5D, 0xFF, 0x80] {
            let mut copy = doc.to_vec();
            copy[i] = bad;
            let _ = json::parse(&copy);
        }
    }
}

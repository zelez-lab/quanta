//! The parity matrix: cases × backends, rendered as markdown.

use crate::CaseVerdict;

/// One rendered matrix cell.
pub fn cell(v: &CaseVerdict) -> String {
    match v {
        CaseVerdict::Pass(c) => {
            if c.divergent_pixels == 0 {
                format!("✅ Δ≤{}", c.max_channel_delta)
            } else {
                format!(
                    "✅ Δ≤{} ({}/{} edge px)",
                    c.max_channel_delta, c.divergent_pixels, c.total_pixels
                )
            }
        }
        CaseVerdict::Fail(c) => format!(
            "❌ Δ{} ({}/{} px out)",
            c.max_channel_delta, c.divergent_pixels, c.total_pixels
        ),
        CaseVerdict::ShapeMismatch => "❌ shape".to_string(),
        CaseVerdict::Unsupported(r) => format!("— NotSupported: {r}"),
        CaseVerdict::Error(e) => format!("💥 {e}"),
    }
}

/// Render the whole matrix. `rows` are `(case, tolerance-terms,
/// per-backend verdicts)` in corpus order; `backends` the column names.
pub fn markdown(backends: &[String], rows: &[(String, String, Vec<CaseVerdict>)]) -> String {
    let mut out = String::new();
    out.push_str("| case | terms |");
    for b in backends {
        out.push_str(&format!(" {b} |"));
    }
    out.push_str("\n|---|---|");
    for _ in backends {
        out.push_str("---|");
    }
    out.push('\n');
    for (name, terms, verdicts) in rows {
        out.push_str(&format!("| `{name}` | {terms} |"));
        for v in verdicts {
            out.push_str(&format!(" {} |", cell(v)));
        }
        out.push('\n');
    }
    out
}

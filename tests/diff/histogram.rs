//! Multi-run litmus histogram harness (race-freedom L2, Phase 1).
//!
//! # What this is
//!
//! A litmus test asks: over many concurrent executions of a tiny
//! program, *which final states appear*? A single dispatch answers that
//! for one execution. To sample the distribution we need thousands of
//! executions — but launching thousands of dispatches is dominated by
//! per-dispatch overhead (pipeline bind, queue submit, fence wait).
//!
//! Real hardware litmus harnesses (e.g. the `litmus7` tool, or Alglave
//! et al.'s GPU studies) amortize that overhead by packing **many
//! independent litmus instances into one dispatch**: each instance gets
//! its own private cells and observer slots, and the instances never
//! touch each other's memory. One dispatch of `2 * INSTANCES` quarks
//! then yields `INSTANCES` independent samples. 10^5 samples come from
//! ONE dispatch, not 10^5.
//!
//! # Instance layout
//!
//! Every instance `i` owns two quarks — a global quark id `g` maps to
//! `instance = g / 2` and `role = g & 1`. Role 0 and role 1 run the two
//! sides of the litmus shape (producer/consumer for MP, the two writers
//! for SB). Each instance's cells live at index `i` of per-cell buffers
//! (`cell_a[i]`, `cell_b[i]`), and its observation lands in per-instance
//! observer slots (`obs0[i]`, `obs1[i]`). No two instances share an
//! address, so there is exactly one race per instance and it is
//! independent of every other.
//!
//! Because the mapping is pure arithmetic on the global quark id, one
//! `gpu.dispatch(&wave, 2 * INSTANCES)` runs the whole batch. The
//! scheduling of role-0 vs role-1 across instances is up to the GPU —
//! that scheduling freedom is precisely what a weak-memory anomaly needs
//! to manifest.
//!
//! # Epistemics
//!
//! These are **empirical falsifiers, not proofs** — same standing as the
//! herd7 tests in `specs/verify/herd7/`. Observing the forbidden MP
//! outcome even once falsifies the claim (and fails the test). *Not*
//! observing it over 10^5 samples is corroboration, never a proof: a
//! particular driver / GPU / scheduling may simply never exercise the
//! offending interleaving. Conversely, a weak outcome that the model
//! *allows* (the SB anomaly under rel/acq) may or may not show on a
//! given device; an in-order software executor will never show it. We
//! therefore assert allowed-anomalies are *permitted*, and only
//! optionally require them to be *observed*.

use std::collections::HashMap;

/// An outcome histogram: a map from an observed outcome vector to the
/// number of instances that produced it.
#[derive(Debug, Clone, Default)]
pub struct Histogram {
    pub counts: HashMap<Vec<u32>, u64>,
    pub instances: u64,
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            instances: 0,
        }
    }

    pub fn record(&mut self, outcome: Vec<u32>) {
        *self.counts.entry(outcome).or_insert(0) += 1;
        self.instances += 1;
    }

    pub fn count_of(&self, outcome: &[u32]) -> u64 {
        self.counts.get(outcome).copied().unwrap_or(0)
    }

    /// Pretty, deterministic (sorted) dump for panic messages.
    pub fn dump(&self) -> String {
        let mut rows: Vec<(&Vec<u32>, &u64)> = self.counts.iter().collect();
        rows.sort_by(|a, b| a.0.cmp(b.0));
        let mut s = format!("histogram ({} instances):\n", self.instances);
        for (outcome, count) in rows {
            s.push_str(&format!("  {:?} => {}\n", outcome, count));
        }
        s
    }
}

/// The outcome of `assert_outcomes`, so callers can report what actually
/// happened even on success (litmus tests are more informative when the
/// observed distribution is printed, not just pass/fail).
#[derive(Debug)]
pub struct OutcomeReport {
    pub ok: bool,
    pub message: String,
}

/// Assert a histogram against a memory-model verdict.
///
/// - `allowed`: every observed outcome MUST be a member of this set.
///   An outcome outside it is a model violation (e.g. a torn read).
/// - `forbidden`: each of these outcomes MUST have count 0. Observing a
///   forbidden outcome even once fails (the MP bad outcome).
/// - `must_observe`: each of these outcomes MUST be seen at least once
///   (anti-vacuity — proves the test can actually reach the interesting
///   states). Leave empty for outcomes a given device may legitimately
///   never exhibit (the SB anomaly on an in-order lane).
///
/// On any violation the returned report is `ok == false` and its message
/// contains the full histogram.
pub fn assert_outcomes(
    hist: &Histogram,
    allowed: &[Vec<u32>],
    forbidden: &[Vec<u32>],
    must_observe: &[Vec<u32>],
) -> OutcomeReport {
    let mut problems: Vec<String> = Vec::new();

    // 1. Membership: no observed outcome may fall outside `allowed`.
    for outcome in hist.counts.keys() {
        if !allowed.iter().any(|a| a == outcome) {
            problems.push(format!(
                "outcome {:?} (count {}) is not in the allowed set",
                outcome,
                hist.count_of(outcome)
            ));
        }
    }

    // 2. Forbidden outcomes must never appear.
    for f in forbidden {
        let c = hist.count_of(f);
        if c != 0 {
            problems.push(format!("forbidden outcome {:?} appeared {} times", f, c));
        }
    }

    // 3. Anti-vacuity: required outcomes must appear.
    for m in must_observe {
        if hist.count_of(m) == 0 {
            problems.push(format!(
                "must-observe outcome {:?} never appeared (test is vacuous)",
                m
            ));
        }
    }

    if problems.is_empty() {
        OutcomeReport {
            ok: true,
            message: format!(
                "all {} outcomes within model\n{}",
                hist.instances,
                hist.dump()
            ),
        }
    } else {
        OutcomeReport {
            ok: false,
            message: format!("{}\n{}", problems.join("\n"), hist.dump()),
        }
    }
}

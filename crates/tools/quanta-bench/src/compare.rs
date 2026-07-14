//! Regression / improvement gate. Returns true if the run passes.

use crate::result::Report;

pub fn report(baseline: &Report, current: &Report, threshold_pct: f64) -> bool {
    println!("Quanta perf regression check");
    println!(
        "  baseline platform: {}  gpu: {}",
        baseline.platform, baseline.gpu_name
    );
    println!(
        "  current  platform: {}  gpu: {}",
        current.platform, current.gpu_name
    );
    println!("  threshold: ±{:.1}%", threshold_pct);
    println!();

    if baseline.platform != current.platform {
        eprintln!(
            "WARN: platform mismatch ({} vs {}). Cross-platform comparison is not meaningful.",
            baseline.platform, current.platform
        );
    }

    let mut ok = true;
    let mut regressions = 0usize;
    let mut wins = 0usize;
    let mut new_entries = 0usize;
    let mut missing = 0usize;

    println!(
        "  {:<24} {:<28} {:>14} {:>14} {:>10}",
        "name", "workload", "baseline_ms", "current_ms", "delta_pct"
    );
    println!("  {}", "─".repeat(94));

    for cur in &current.results {
        match baseline
            .results
            .iter()
            .find(|b| b.name == cur.name && b.workload == cur.workload)
        {
            Some(base) => {
                let pct = if base.gpu_ms > 0.0 {
                    (cur.gpu_ms - base.gpu_ms) / base.gpu_ms * 100.0
                } else {
                    0.0
                };
                let mark = if pct.abs() <= threshold_pct {
                    " "
                } else if pct > 0.0 {
                    regressions += 1;
                    ok = false;
                    "✖"
                } else {
                    wins += 1;
                    ok = false;
                    "✓"
                };
                println!(
                    "{} {:<24} {:<28} {:>14.4} {:>14.4} {:>9.2}%",
                    mark, cur.name, cur.workload, base.gpu_ms, cur.gpu_ms, pct
                );
            }
            None => {
                new_entries += 1;
                println!(
                    "+ {:<24} {:<28} {:>14} {:>14.4} {:>10}",
                    cur.name, cur.workload, "—", cur.gpu_ms, "new"
                );
            }
        }
    }

    for base in &baseline.results {
        if !current
            .results
            .iter()
            .any(|c| c.name == base.name && c.workload == base.workload)
        {
            missing += 1;
            println!(
                "- {:<24} {:<28} {:>14.4} {:>14} {:>10}",
                base.name, base.workload, base.gpu_ms, "—", "missing"
            );
        }
    }

    println!();
    println!(
        "  regressions: {}   improvements: {}   new: {}   missing: {}",
        regressions, wins, new_entries, missing
    );

    if regressions > 0 {
        eprintln!("FAIL: {} regression(s) ≥{}%", regressions, threshold_pct);
    }
    if wins > 0 {
        eprintln!(
            "FAIL: {} unaccounted improvement(s) ≥{}% — update baseline in this PR",
            wins, threshold_pct
        );
    }
    if new_entries > 0 || missing > 0 {
        eprintln!(
            "FAIL: bench set drifted (new: {}, missing: {}) — update baseline",
            new_entries, missing
        );
        ok = false;
    }

    ok
}

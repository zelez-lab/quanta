//! Quanta performance regression harness.
//!
//! Subcommands:
//!   run     — execute all benchmarks, emit JSON results
//!   compare — load baseline + current results, fail on regression ≥5%
//!
//! Designed for CI gating: every PR runs `run` and `compare` against the
//! committed baseline at `bench/baselines/<platform>.json`. Improvements ≥5%
//! also fail (forcing baseline update in the same PR).

mod bench;
mod compare;
mod json;
mod result;

use std::env;
use std::process::ExitCode;

fn print_help() {
    eprintln!("Quanta benchmark regression harness");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    quanta-bench run [--smoke] [--out PATH | --out-dir DIR]");
    eprintln!(
        "    quanta-bench compare (--baseline PATH | --baseline-dir DIR) --current PATH [--threshold PERCENT]"
    );
    eprintln!();
    eprintln!("FLAGS:");
    eprintln!("    --smoke              Run each bench at the smallest size, do not record perf");
    eprintln!("    --out PATH           Write JSON results to PATH (default: stdout)");
    eprintln!("    --out-dir DIR        Write JSON results to DIR/<platform>-<device-slug>.json");
    eprintln!("    --baseline PATH      Path to committed baseline JSON");
    eprintln!(
        "    --baseline-dir DIR   Pick DIR/<platform>-<device-slug>.json for the device that ran;"
    );
    eprintln!("                         a missing file leaves the gate unarmed (pass, loudly)");
    eprintln!("    --current PATH       Path to current run JSON");
    eprintln!("    --threshold PERCENT  Regression/improvement threshold (default: 5.0)");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        return ExitCode::from(2);
    }

    match args[0].as_str() {
        "run" => match run_cmd(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {}", e);
                ExitCode::from(1)
            }
        },
        "compare" => match compare_cmd(&args[1..]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::from(3), // regression / unaccounted improvement
            Err(e) => {
                eprintln!("error: {}", e);
                ExitCode::from(1)
            }
        },
        "-h" | "--help" => {
            print_help();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown subcommand: {}", other);
            print_help();
            ExitCode::from(2)
        }
    }
}

fn run_cmd(args: &[String]) -> Result<(), String> {
    let mut smoke = false;
    let mut out: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--smoke" => smoke = true,
            "--out" => {
                i += 1;
                out = Some(args.get(i).ok_or("--out needs a path")?.clone());
            }
            "--out-dir" => {
                i += 1;
                out_dir = Some(args.get(i).ok_or("--out-dir needs a directory")?.clone());
            }
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    if out.is_some() && out_dir.is_some() {
        return Err("--out and --out-dir are exclusive".into());
    }

    let report = bench::run_all(smoke).map_err(|e| format!("bench failed: {}", e))?;
    let json = json::encode_report(&report);
    // The device-keyed file: one baseline per device per OS/arch.
    let out = out.or_else(|| {
        out_dir.map(|dir| {
            let path = std::path::Path::new(&dir).join(report.baseline_file_name());
            eprintln!(
                "recording baseline for \"{}\" at {}",
                report.gpu_name,
                path.display()
            );
            path.to_string_lossy().into_owned()
        })
    });
    match out {
        Some(path) => std::fs::write(&path, json).map_err(|e| format!("write {}: {}", path, e))?,
        None => println!("{}", json),
    }
    Ok(())
}

fn compare_cmd(args: &[String]) -> Result<bool, String> {
    let mut baseline: Option<String> = None;
    let mut baseline_dir: Option<String> = None;
    let mut current: Option<String> = None;
    let mut threshold = 5.0f64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" => {
                i += 1;
                baseline = Some(args.get(i).ok_or("--baseline needs a path")?.clone());
            }
            "--baseline-dir" => {
                i += 1;
                baseline_dir = Some(
                    args.get(i)
                        .ok_or("--baseline-dir needs a directory")?
                        .clone(),
                );
            }
            "--current" => {
                i += 1;
                current = Some(args.get(i).ok_or("--current needs a path")?.clone());
            }
            "--threshold" => {
                i += 1;
                threshold = args
                    .get(i)
                    .ok_or("--threshold needs a number")?
                    .parse()
                    .map_err(|_| "--threshold must be a number".to_string())?;
            }
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }

    let current_path = current.ok_or("--current is required")?;
    let current_json = std::fs::read_to_string(&current_path)
        .map_err(|e| format!("read {}: {}", current_path, e))?;
    let current_report =
        json::decode_report(&current_json).map_err(|e| format!("parse {}: {}", current_path, e))?;

    // Resolve the baseline: an explicit path, or the device-keyed file in
    // the baselines directory for whatever device produced `current`. No
    // file for this device = the gate is unarmed for it (loud pass, like
    // the device-mismatch case in `compare::report`): recording one arms it.
    let baseline_path = match (baseline, baseline_dir) {
        (Some(p), _) => p,
        (None, Some(dir)) => {
            let path = std::path::Path::new(&dir).join(current_report.baseline_file_name());
            if !path.exists() {
                eprintln!(
                    "SKIP: no baseline for \"{}\" on {} (expected {}) — the gate is unarmed \
                     until `quanta-bench run --out-dir {}` records one on this device",
                    current_report.gpu_name,
                    current_report.platform,
                    path.display(),
                    dir
                );
                return Ok(true);
            }
            path.to_string_lossy().into_owned()
        }
        (None, None) => return Err("--baseline or --baseline-dir is required".into()),
    };

    let baseline_json = std::fs::read_to_string(&baseline_path)
        .map_err(|e| format!("read {}: {}", baseline_path, e))?;
    let baseline_report = json::decode_report(&baseline_json)
        .map_err(|e| format!("parse {}: {}", baseline_path, e))?;

    Ok(compare::report(
        &baseline_report,
        &current_report,
        threshold,
    ))
}

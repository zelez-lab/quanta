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
    eprintln!("    quanta-bench run [--smoke] [--out PATH]");
    eprintln!("    quanta-bench compare --baseline PATH --current PATH [--threshold PERCENT]");
    eprintln!();
    eprintln!("FLAGS:");
    eprintln!("    --smoke              Run each bench at the smallest size, do not record perf");
    eprintln!("    --out PATH           Write JSON results to PATH (default: stdout)");
    eprintln!("    --baseline PATH      Path to committed baseline JSON");
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
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--smoke" => smoke = true,
            "--out" => {
                i += 1;
                out = Some(args.get(i).ok_or("--out needs a path")?.clone());
            }
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }

    let report = bench::run_all(smoke).map_err(|e| format!("bench failed: {}", e))?;
    let json = json::encode_report(&report);
    match out {
        Some(path) => std::fs::write(&path, json).map_err(|e| format!("write {}: {}", path, e))?,
        None => println!("{}", json),
    }
    Ok(())
}

fn compare_cmd(args: &[String]) -> Result<bool, String> {
    let mut baseline: Option<String> = None;
    let mut current: Option<String> = None;
    let mut threshold = 5.0f64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" => {
                i += 1;
                baseline = Some(args.get(i).ok_or("--baseline needs a path")?.clone());
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

    let baseline_path = baseline.ok_or("--baseline is required")?;
    let current_path = current.ok_or("--current is required")?;

    let baseline_json = std::fs::read_to_string(&baseline_path)
        .map_err(|e| format!("read {}: {}", baseline_path, e))?;
    let current_json = std::fs::read_to_string(&current_path)
        .map_err(|e| format!("read {}: {}", current_path, e))?;

    let baseline_report = json::decode_report(&baseline_json)
        .map_err(|e| format!("parse {}: {}", baseline_path, e))?;
    let current_report =
        json::decode_report(&current_json).map_err(|e| format!("parse {}: {}", current_path, e))?;

    Ok(compare::report(
        &baseline_report,
        &current_report,
        threshold,
    ))
}

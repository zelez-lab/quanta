//! The conform runner. Modes:
//!
//! * `--bless <backend> --golden-dir <dir>` — run the corpus on one
//!   backend and write its frames as the goldens.
//! * `--backends a,b --golden-dir <dir>` — run the corpus on each
//!   backend (a child process per backend, `QUANTA_BACKEND` set, so no
//!   driver state crosses over) and compare every frame against the
//!   committed goldens; print the parity matrix; exit non-zero on any
//!   failure.
//! * `--backends a,b --reference a` — no goldens: compare the listed
//!   backends live against one of them (a multi-backend box).
//!
//! There is no CPU rasterizer (the CPU device refuses render passes),
//! so the reference is a committed set of goldens — blessed from the
//! backend named in the goldens' manifest — and CI cross-checks the
//! other backends against it. A divergence is a divergence between two
//! real rasterizers either way.

use quanta_conform::{CaseVerdict, Frame, compare, corpus, matrix};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    if let Some(backend) = get("--run-one") {
        let dir = get("--dump").expect("--run-one needs --dump <dir>");
        run_one(&backend, Path::new(&dir));
        return;
    }
    let exe = std::env::current_exe().expect("own path");
    if let Some(backend) = get("--bless") {
        let dir = PathBuf::from(get("--golden-dir").expect("--bless needs --golden-dir"));
        let status = Command::new(&exe)
            .args(["--run-one", &backend, "--dump"])
            .arg(&dir)
            .env("QUANTA_BACKEND", env_backend(&backend))
            .status()
            .expect("spawn child");
        assert!(status.success(), "bless child failed");
        std::fs::write(dir.join("BLESSED_FROM"), &backend).expect("write manifest");
        eprintln!(
            "[conform] goldens blessed from `{backend}` into {}",
            dir.display()
        );
        return;
    }
    let backends_arg = get("--backends").unwrap_or_default();
    let backends: Vec<String> = backends_arg
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert!(
        !backends.is_empty(),
        "--backends a,b is required (or --bless)"
    );
    let out = get("--out").map(PathBuf::from);
    let golden = get("--golden-dir").map(PathBuf::from);
    let reference = get("--reference");
    orchestrate(
        &exe,
        &backends,
        golden.as_deref(),
        reference.as_deref(),
        out.as_deref(),
    );
}

/// The `QUANTA_BACKEND` spelling for a matrix backend name.
fn env_backend(name: &str) -> &str {
    if name == "software" { "cpu" } else { name }
}

/// The child: one backend, all cases, frames to disk.
fn run_one(backend: &str, dir: &Path) {
    std::fs::create_dir_all(dir).expect("create dump dir");
    // Backend choice is the environment's; the orchestrator set it.
    let gpu = match quanta::init() {
        Ok(g) => g,
        Err(e) => {
            std::fs::write(dir.join("__init.err"), e.to_string()).unwrap();
            return;
        }
    };
    eprintln!("[conform] {backend}: `{}`", gpu.name());
    for case in corpus() {
        match (case.run)(&gpu) {
            Ok(frame) => frame
                .save(&dir.join(format!("{}.frame", case.name)))
                .expect("write frame"),
            Err(e) => {
                let not_supported = matches!(e.kind, quanta::QuantaErrorKind::NotSupported(_));
                let name = if not_supported {
                    format!("{}.unsupported", case.name)
                } else {
                    format!("{}.err", case.name)
                };
                std::fs::write(dir.join(name), e.to_string()).unwrap();
            }
        }
    }
}

/// One backend's per-case outcome, as read back from its dump dir.
fn load_outcome(dir: &Path, case: &str) -> Result<Frame, CaseVerdict> {
    let init_err = dir.join("__init.err");
    if init_err.exists() {
        return Err(CaseVerdict::Error(format!(
            "init failed: {}",
            std::fs::read_to_string(init_err).unwrap_or_default()
        )));
    }
    let unsupported = dir.join(format!("{case}.unsupported"));
    if unsupported.exists() {
        return Err(CaseVerdict::Unsupported(
            std::fs::read_to_string(unsupported).unwrap_or_default(),
        ));
    }
    let err = dir.join(format!("{case}.err"));
    if err.exists() {
        return Err(CaseVerdict::Error(
            std::fs::read_to_string(err).unwrap_or_default(),
        ));
    }
    match Frame::load(&dir.join(format!("{case}.frame"))) {
        Ok(f) => Ok(f),
        Err(e) => Err(CaseVerdict::Error(format!("no frame: {e}"))),
    }
}

fn orchestrate(
    exe: &Path,
    backends: &[String],
    golden: Option<&Path>,
    reference: Option<&str>,
    out: Option<&Path>,
) {
    let root = std::env::temp_dir().join(format!("quanta-conform-{}", std::process::id()));
    for b in backends {
        let dir = root.join(b);
        let status = Command::new(exe)
            .args(["--run-one", b, "--dump"])
            .arg(&dir)
            .env("QUANTA_BACKEND", env_backend(b))
            .status()
            .expect("spawn child");
        if !status.success() {
            eprintln!("[conform] child for `{b}` exited with {status}");
        }
    }
    // The reference: the goldens, or one of the live backends.
    let (ref_dir, ref_name) = match (golden, reference) {
        (Some(g), None) => {
            let from = std::fs::read_to_string(g.join("BLESSED_FROM")).unwrap_or_default();
            (g.to_path_buf(), format!("goldens ({})", from.trim()))
        }
        (None, Some(r)) => {
            assert!(
                backends.iter().any(|b| b == r),
                "--reference must be one of --backends"
            );
            (root.join(r), format!("live {r}"))
        }
        _ => panic!("give exactly one of --golden-dir or --reference"),
    };
    let cols: Vec<&String> = backends
        .iter()
        .filter(|b| format!("live {b}") != ref_name)
        .collect();
    let mut rows = Vec::new();
    let mut failed = false;
    for case in corpus() {
        let reference_frame = load_outcome(&ref_dir, case.name);
        let mut verdicts = Vec::new();
        for b in &cols {
            let v = match (&reference_frame, load_outcome(&root.join(b), case.name)) {
                (Ok(r), Ok(g)) => compare::compare(r, &g, case.tolerance),
                (Err(_), _) => CaseVerdict::Error("no reference frame".to_string()),
                (_, Err(v)) => v,
            };
            match &v {
                CaseVerdict::Fail(_) | CaseVerdict::ShapeMismatch | CaseVerdict::Error(_) => {
                    failed = true
                }
                _ => {}
            }
            verdicts.push(v);
        }
        let terms = format!(
            "Δ≤{}, budget {}‰",
            case.tolerance.channel, case.tolerance.edge_budget_permille
        );
        rows.push((case.name.to_string(), terms, verdicts));
    }
    let col_names: Vec<String> = cols.iter().map(|b| format!("{b} vs {ref_name}")).collect();
    let md = matrix::markdown(&col_names, &rows);
    println!("{md}");
    if let Some(p) = out {
        std::fs::write(p, &md).expect("write matrix");
    }
    let _ = std::fs::remove_dir_all(&root);
    if failed {
        std::process::exit(1);
    }
}

//! Quanta CLI — `quanta` binary.
//!
//! Replaces the ad-hoc `scripts/*.sh` glue with structured Rust that's
//! cross-platform, discoverable (`quanta --help`), and lives inside the
//! cargo workspace so it's checked by `cargo clippy --workspace`.
//!
//! The CLI is the *developer* tool, distinct from the user-facing
//! `quanta` library crate. It freely takes development dependencies
//! (`clap`, …); the no-transitive-deps policy applies to user wasm
//! output, not to dev tooling. Same split as `dija-cli` etc. across
//! the wider workspace.
//!
//! Subcommands:
//!
//! - `quanta build web [<example>]` — compile `web/src/quanta.ts` →
//!   `web/dist/quanta.js`, build the wasm binary for a smoke-test
//!   example, stage outputs into `examples/web_*/`. Default builds
//!   every example.
//! - `quanta serve <example> [--port PORT]` — rebuild then serve the
//!   example dir over HTTP. Embedded `std::net` server.
//! - `quanta check` — `cargo check` + `cargo clippy` + TS `--noEmit` in
//!   one verb.

use clap::{Parser, Subcommand};

mod build;
mod check;
mod codegen;
mod serve;
mod wasm_experiment;
mod workspace;

/// Boxed-error result alias used across subcommand modules.
///
/// We don't pull `anyhow` for a CLI this small. A boxed error covers
/// everything our subcommands return without ceremony.
pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(
    name = "quanta",
    version,
    about = "Quanta dev CLI — build, serve, and check the workspace.",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build a deliverable (currently: browser smoke tests).
    Build(BuildArgs),
    /// Serve a built smoke test over HTTP for manual browser testing.
    Serve(ServeArgs),
    /// Run the full pre-commit-equivalent checks (clippy + TS noEmit).
    Check,
    /// Run code generators that derive Rust + TS source from a spec
    /// (currently: WebGPU IDL → enum tables for both sides).
    Codegen(CodegenArgs),
    /// Research: extract a kernel function as wasm32 and dump the
    /// resulting WASM module (text + binary). Used to scope the
    /// long-term WASM-route translator (roadmap 058 / 059 / 080).
    WasmExperiment(WasmExperimentArgs),
}

#[derive(clap::Args)]
struct WasmExperimentArgs {
    /// Path to a Rust source file containing a `#[quanta::kernel]`
    /// function. The harness wraps the file as a wasm32 lib crate
    /// and emits the WASM rustc produces.
    source: String,
    /// Output directory for `kernel.wasm` and `kernel.wat`. Defaults
    /// to `target/wasm-experiment/`.
    #[arg(long, default_value = "target/wasm-experiment")]
    out_dir: String,
}

#[derive(clap::Args)]
struct CodegenArgs {
    #[command(subcommand)]
    target: CodegenTarget,
}

#[derive(Subcommand)]
enum CodegenTarget {
    /// Read `web/webgpu.idl` and emit
    /// `src/driver/webgpu/generated_codes.rs` +
    /// `web/src/generated/codes.ts`.
    Webgpu,
}

#[derive(clap::Args)]
struct BuildArgs {
    #[command(subcommand)]
    target: BuildTarget,
}

#[derive(Subcommand)]
enum BuildTarget {
    /// Build a browser smoke test (wasm + `quanta.js`) and stage it into
    /// `examples/web_<name>/`.
    Web {
        /// Name of the example (e.g. `web_add_one`). Omit to build all.
        #[arg(default_value = "all")]
        example: String,
        /// Build profile. Defaults to release for smaller wasm.
        #[arg(long, default_value = "release")]
        profile: String,
    },
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Name of the example to serve (e.g. `web_add_one`).
    example: String,
    /// TCP port to bind. Defaults to 8000.
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Skip the rebuild step before serving.
    #[arg(long)]
    no_build: bool,
    /// Build profile if rebuilding.
    #[arg(long, default_value = "release")]
    profile: String,
}

fn run(cli: Cli) -> Result<()> {
    match cli.cmd {
        Cmd::Build(BuildArgs {
            target: BuildTarget::Web { example, profile },
        }) => build::web(&example, &profile),
        Cmd::Serve(ServeArgs {
            example,
            port,
            no_build,
            profile,
        }) => {
            if !no_build {
                build::web(&example, &profile)?;
            }
            serve::run(&example, port)
        }
        Cmd::Check => check::run(),
        Cmd::Codegen(CodegenArgs {
            target: CodegenTarget::Webgpu,
        }) => codegen::webgpu(),
        Cmd::WasmExperiment(WasmExperimentArgs { source, out_dir }) => {
            wasm_experiment::run(&source, &out_dir)
        }
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("quanta: {e}");
        std::process::exit(1);
    }
}

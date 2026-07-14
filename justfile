# Quanta — GPU compute and rendering API

# Build
build:
    cargo build

build-release:
    cargo build --release

build-compiler:
    cargo build -p quanta-compiler --release

build-vulkan:
    cargo build --features vulkan --no-default-features

# Feature-combo matrix: the compute/render boundary must hold in every
# quadrant (core-only / render / compute / both). Now that the split is
# a crate boundary (quanta-core / quanta-render / the facade), this
# gate proves each quadrant still builds — run it after touching
# anything on the Gpu/GpuDevice surface or the driver layer.
check-combos:
    cargo build -p quanta --no-default-features --features metal
    cargo build -p quanta --no-default-features --features "metal render"
    cargo build -p quanta --no-default-features --features "metal compute jit"
    cargo build -p quanta --no-default-features --features "metal render compute jit"

# The crate-split graph guarantee: quanta-render's dependency tree must
# never contain the compute stack (kernel lowering, wasmparser, the
# quanta-ir JIT).
check-render-graph:
    ! cargo tree -p quanta-render -e normal | grep -iE 'wasm|jit'

# Test
test:
    cargo test --all

test-ir:
    cargo test -p quanta-ir

test-conformance:
    cargo test --test conformance_test

test-conformance-validate:
    QUANTA_VALIDATE=1 cargo test --test conformance_test

# Run examples
example-hello:
    cargo run --example hello_quanta

example-bench-compute:
    cargo run --example bench_compute --release

example-bench-mandelbrot:
    cargo run --example bench_mandelbrot --release

example-bench-nbody:
    cargo run --example bench_nbody --release

# Performance regression suite (step 069)
# Records JSON results; gated by `bench-check` against committed baseline.
BENCH_BASELINE := if os() == "macos" { "bench/baselines/macos-aarch64.json" } else { "bench/baselines/linux-x86_64.json" }

bench:
    cargo run --release -p quanta-bench -- run

bench-record:
    cargo run --release -p quanta-bench -- run --out {{BENCH_BASELINE}}

bench-check:
    cargo run --release -p quanta-bench -- run --out /tmp/quanta-bench-current.json
    cargo run --release -p quanta-bench -- compare --baseline {{BENCH_BASELINE}} --current /tmp/quanta-bench-current.json

bench-smoke:
    cargo run --release -p quanta-bench --no-default-features -- run --smoke

# Compiler tests
test-ptx:
    cargo run -p quanta-compiler -- --test-ptx

test-amd:
    cargo run -p quanta-compiler -- --test-amd

test-complex:
    cargo run -p quanta-compiler -- --test-complex

test-ir-gen:
    cargo run -p quanta-compiler -- --test-ir

# RPi 5 testing (Vulkan on real hardware)
# Set PI_HOST in env or pass as: just test-pi PI_HOST=pi@192.168.1.x
PI_HOST := env("PI_HOST", "pi@rpi5.local")

build-pi:
    cross build --target aarch64-unknown-linux-gnu --features vulkan --no-default-features --release

test-pi: build-pi
    scp target/aarch64-unknown-linux-gnu/release/examples/hello_quanta {{PI_HOST}}:/tmp/
    ssh {{PI_HOST}} "cd /tmp && ./hello_quanta"

# Vulkan CTS (dEQP) — run subset on Pi
# Requires dEQP installed on the Pi: sudo apt install vulkan-tools deqp-vk
test-deqp:
    ssh {{PI_HOST}} "deqp-vk --deqp-case=dEQP-VK.api.device_init.create_device_simple"

test-deqp-compute:
    ssh {{PI_HOST}} "deqp-vk --deqp-case=dEQP-VK.compute.*"

test-deqp-memory:
    ssh {{PI_HOST}} "deqp-vk --deqp-case=dEQP-VK.memory.*"

# Verus — emits rlib side-effect outputs into target/verus/ instead of the
# repo root (`verus --crate-type=lib` would otherwise drop libfoo.rlib in cwd).
# Pass anything you'd pass to verus directly.
#   Example: just verus specs/verify/verus/quanta/api_invariants.rs --crate-type=lib
verus +ARGS:
    mkdir -p target/verus
    verus {{ARGS}} -- --out-dir target/verus

# herd7 memory-model litmus tests (steps 055/056).
#
# Runs the LISA litmus tests in specs/verify/herd7/ under the vendored
# release/acquire model (vmm.bell + vmm.cat) and asserts each herd7
# verdict matches the expected outcome. These are empirical cross-checks
# of the A6-A9 / T1600-T1622 memory-model axioms, NOT proofs.
#
# herd7 ships with herdtools7: `opam install herdtools7`. If herd7 is not
# on PATH this recipe skips (exit 0) with an install hint, so it never
# blocks a machine without the toolsuite.
litmus:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v herd7 >/dev/null 2>&1; then
        echo "SKIP: herd7 not found on PATH."
        echo "      Install the diy/herd toolsuite:  opam install herdtools7"
        echo "      (then re-run:  just litmus)"
        exit 0
    fi
    H=specs/verify/herd7
    BELL="$H/vmm.bell"
    CAT="$H/vmm.cat"
    fail=0
    # test-file  expected-verdict  (Never | Sometimes)
    check() {
        local file="$1" expected="$2"
        local out verdict
        out=$(herd7 -bell "$BELL" -model "$CAT" "$H/$file" 2>&1) || {
            echo "FAIL  $file: herd7 errored"; echo "$out" | sed 's/^/    /'; fail=1; return
        }
        verdict=$(printf '%s\n' "$out" | grep -oE 'Observation [^ ]+ (Never|Sometimes|Always)' | awk '{print $3}' | head -1)
        if [ "$verdict" = "$expected" ]; then
            echo "OK    $file: $verdict (expected $expected)"
        else
            echo "FAIL  $file: got '${verdict:-<none>}', expected '$expected'"
            printf '%s\n' "$out" | sed 's/^/    /'
            fail=1
        fi
    }
    echo "herd7 litmus: $(herd7 -version | head -1)"
    check message_passing.litmus      Never
    check store_buffer.litmus         Sometimes
    check store_buffer_sc.litmus      Never
    check atomic_add_visibility.litmus Never
    if [ "$fail" -ne 0 ]; then
        echo "litmus: one or more tests did not match their expected verdict"
        exit 1
    fi
    echo "litmus: all verdicts match"

# Quality
fmt:
    cargo fmt --all

clippy:
    cargo clippy --all -- -D warnings

clippy-vulkan:
    cargo clippy --features vulkan -- -D warnings

quality: fmt clippy test-conformance

verify: quality clippy-vulkan check-combos test

# Wire up the tracked git hooks (.githooks/pre-commit runs fmt + clippy).
# Run once after cloning — git does not pick up .githooks automatically.
hooks:
    git config core.hooksPath .githooks
    @echo "pre-commit hook active (fmt --check + clippy -D warnings)"

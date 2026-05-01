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

# Quality
fmt:
    cargo fmt --all

clippy:
    cargo clippy --all -- -D warnings

clippy-vulkan:
    cargo clippy --features vulkan -- -D warnings

quality: fmt clippy test-conformance

verify: quality clippy-vulkan test

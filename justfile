# Quanta — GPU compute and rendering API

# Build
build:
    cargo build

build-release:
    cargo build --release

build-compiler:
    cargo build -p quanta-compiler --release

# Test
test:
    cargo test --all

test-ir:
    cargo test -p quanta-ir

# Run examples
example-vector-add:
    cargo run --example vector_add

example-kernel-macro:
    cargo run --example kernel_macro

example-bench-heavy:
    cargo run --example bench_heavy --release

example-bench-real:
    cargo run --example bench_real_world --release

example-bench-o2-o3:
    cargo run --example bench_o2_vs_o3 --release

# Compiler tests
test-ptx:
    cargo run -p quanta-compiler -- --test-ptx

test-amd:
    cargo run -p quanta-compiler -- --test-amd

test-complex:
    cargo run -p quanta-compiler -- --test-complex

test-ir-gen:
    cargo run -p quanta-compiler -- --test-ir

# Quality
fmt:
    cargo fmt --all

clippy:
    cargo clippy --all

quality: fmt clippy test

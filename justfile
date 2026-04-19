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

# Quality
fmt:
    cargo fmt --all

clippy:
    cargo clippy --all -- -D warnings

clippy-vulkan:
    cargo clippy --features vulkan -- -D warnings

quality: fmt clippy test-conformance

verify: quality clippy-vulkan test

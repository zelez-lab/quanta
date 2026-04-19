//! Quanta Conformance Test Suite
//!
//! Runs all conformance tests against the default GPU driver.
//! On Mac: tests Metal. On Linux: tests Vulkan.
//!
//! Run:
//!   cargo test --test conformance_test
//!   QUANTA_VALIDATE=1 cargo test --test conformance_test   (with validation layer)

mod conformance;

fn get_gpu() -> quanta::Gpu {
    quanta::init().expect("no GPU found — cannot run conformance tests")
}

// === Memory ===

#[test]
fn memory_field_write_read() {
    conformance::memory::field_write_read(&get_gpu());
}

#[test]
fn memory_field_write_read_u32() {
    conformance::memory::field_write_read_u32(&get_gpu());
}

#[test]
fn memory_field_copy() {
    conformance::memory::field_copy(&get_gpu());
}

#[test]
fn memory_field_resize() {
    conformance::memory::field_resize(&get_gpu());
}

#[test]
fn memory_field_large_alloc() {
    conformance::memory::field_large_alloc(&get_gpu());
}

#[test]
fn memory_uniform_field() {
    conformance::memory::uniform_field(&get_gpu());
}

// === Compute ===

#[test]
fn compute_vector_add() {
    conformance::compute::vector_add(&get_gpu());
}

#[test]
fn compute_push_constant() {
    conformance::compute::push_constant(&get_gpu());
}

#[test]
fn compute_thread_id() {
    conformance::compute::thread_id(&get_gpu());
}

#[test]
fn compute_large_dispatch() {
    conformance::compute::large_dispatch(&get_gpu());
}

#[test]
fn compute_wave_rebind() {
    conformance::compute::wave_rebind(&get_gpu());
}

// === Texture ===

#[test]
fn texture_rgba8_write_read() {
    conformance::texture::rgba8_write_read(&get_gpu());
}

#[test]
fn texture_r8_write_read() {
    conformance::texture::r8_write_read(&get_gpu());
}

#[test]
fn texture_render_target_create() {
    conformance::texture::render_target_create(&get_gpu());
}

#[test]
fn texture_msaa_target_create() {
    conformance::texture::msaa_target_create(&get_gpu());
}

#[test]
fn texture_mipmap_create() {
    conformance::texture::mipmap_create(&get_gpu());
}

#[test]
fn texture_all_formats() {
    conformance::texture::all_formats(&get_gpu());
}

// === Run all (convenience) ===

#[test]
fn conformance_full_suite() {
    let gpu = get_gpu();
    conformance::memory::run_all(&gpu);
    conformance::compute::run_all(&gpu);
    conformance::texture::run_all(&gpu);
}

#![cfg(feature = "render")]
//! Tier 2 -- Advanced features (error handling for unsupported paths).
//!
//! Verifies that features returning "not supported" return proper Err, not panic.
//! Tests: ray tracing, mesh shaders, sparse textures, indirect buffers, bindless.
//! Requires a GPU; skips gracefully if none available.

use quanta::RenderGpu;

use quanta::ray_tracing::RayTracingPipelineDesc;
use quanta::{Format, TextureDesc, TextureUsage};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// === Ray Tracing ===

#[test]
fn build_acceleration_structure_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.acceleration_structure_blas(&[]);
    // Empty geometry should either succeed or return Err -- never panic.
    match result {
        Ok(blas) => {
            // Drop cleans up.
            drop(blas);
        }
        Err(e) => {
            eprintln!("ray tracing not supported (expected): {}", e);
        }
    }
}

#[test]
fn create_ray_tracing_pipeline_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = RayTracingPipelineDesc {
        ray_gen: &[],
        closest_hit: &[],
        miss: &[],
        max_recursion: 1,
    };

    let result = gpu.ray_tracing_pipeline(&desc);
    match result {
        Ok(_) => {}
        Err(e) => {
            eprintln!("ray tracing pipeline not supported (expected): {}", e);
        }
    }
}

#[test]
fn dispatch_rays_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = RayTracingPipelineDesc {
        ray_gen: &[],
        closest_hit: &[],
        miss: &[],
        max_recursion: 1,
    };
    match gpu.ray_tracing_pipeline(&desc) {
        Ok(pipeline) => match pipeline.dispatch_rays(64, 64) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("dispatch_rays not supported (expected): {}", e);
            }
        },
        Err(e) => {
            eprintln!("ray tracing pipeline not supported (expected): {}", e);
        }
    }
}

#[test]
fn acceleration_structure_drop_never_panics() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Dropping a typed acceleration structure releases the handle
    // exactly once and must never panic.
    match gpu.acceleration_structure_blas(&[]) {
        Ok(blas) => drop(blas),
        Err(e) => {
            eprintln!("acceleration structure not supported (expected): {}", e);
        }
    }
}

// === Mesh Shaders ===

#[test]
fn dispatch_mesh_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Try creating a typed mesh pipeline -- backends without mesh
    // shaders surface NotSupported at create time.
    match gpu.mesh_pipeline(quanta::MeshPipelineDesc::default()) {
        Ok(pipeline) => {
            let result = pipeline.dispatch([1, 1, 1]);
            match result {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("mesh dispatch not supported (expected): {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("mesh pipeline not supported (expected): {}", e);
        }
    }
}

// === Sparse Textures ===

#[test]
fn sparse_texture_create_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = TextureDesc::new(256, 256, Format::RGBA8).with_usage(TextureUsage::SHADER_READ);

    let result = gpu.sparse_texture(&desc);
    match result {
        Ok(_tex) => {}
        Err(e) => {
            eprintln!("sparse textures not supported (expected): {}", e);
        }
    }
}

#[test]
fn sparse_map_tile_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = TextureDesc::new(256, 256, Format::RGBA8).with_usage(TextureUsage::SHADER_READ);
    match gpu.sparse_texture(&desc) {
        Ok(tex) => match tex.map_tile(0, 0, 0, 0) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("map_tile not supported (expected): {}", e);
            }
        },
        Err(e) => {
            eprintln!("sparse textures not supported (expected): {}", e);
        }
    }
}

#[test]
fn sparse_unmap_tile_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = TextureDesc::new(256, 256, Format::RGBA8).with_usage(TextureUsage::SHADER_READ);
    match gpu.sparse_texture(&desc) {
        Ok(tex) => match tex.unmap_tile(0, 0, 0) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("unmap_tile not supported (expected): {}", e);
            }
        },
        Err(e) => {
            eprintln!("sparse textures not supported (expected): {}", e);
        }
    }
}

// === Indirect Command Buffers ===

#[test]
fn indirect_buffer_create_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.indirect_command_buffer(16);
    match result {
        Ok(icb) => {
            // Drop cleans up.
            drop(icb);
        }
        Err(e) => {
            eprintln!("indirect buffers not supported (expected): {}", e);
        }
    }
}

#[test]
fn indirect_buffer_execute_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    match gpu.indirect_command_buffer(4) {
        Ok(icb) => match icb.execute_all() {
            Ok(()) => {}
            Err(e) => {
                eprintln!("icb execute not supported (expected): {}", e);
            }
        },
        Err(e) => {
            eprintln!("indirect buffers not supported (expected): {}", e);
        }
    }
}

#[test]
fn indirect_buffer_drop_never_panics() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Dropping the typed wrapper releases the handle exactly once and
    // must never panic.
    match gpu.indirect_command_buffer(4) {
        Ok(icb) => drop(icb),
        Err(e) => {
            eprintln!("indirect buffers not supported (expected): {}", e);
        }
    }
}

// === Bindless Resources ===

#[test]
fn bindless_texture_array_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.bindless_textures(4);
    match result {
        Ok(_arr) => {}
        Err(e) => {
            eprintln!("bindless textures not supported (expected): {}", e);
        }
    }
}

#[test]
fn bindless_buffer_array_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.bindless_buffers(4);
    match result {
        Ok(_arr) => {}
        Err(e) => {
            eprintln!("bindless buffers not supported (expected): {}", e);
        }
    }
}

// === Stencil Read-back ===

#[test]
fn stencil_read_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(
            &TextureDesc::new(16, 16, Format::Depth32Float)
                .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)),
        )
        .unwrap();

    let result = gpu.stencil_read(&tex);
    match result {
        Ok(data) => {
            assert!(!data.is_empty());
        }
        Err(e) => {
            eprintln!("stencil read-back not supported (expected): {}", e);
        }
    }
}

// === Async Compute ===

#[test]
fn supports_async_compute_returns_bool() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Should return a boolean without panicking.
    let _supported = gpu.supports_async_compute();
}

#[test]
fn async_compute_dispatch_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    if !gpu.supports_async_compute() {
        eprintln!("async compute not supported, testing error path");
    }

    // Create a wave for the dispatch.
    let msl = b"#include <metal_stdlib>\nusing namespace metal;\nkernel void noop(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) { data[id] = data[id]; }\n";
    match gpu.wave(msl) {
        Ok(wave) => {
            let result = gpu.async_compute_dispatch(&wave, [1, 1, 1]);
            match result {
                Ok(mut pulse) => {
                    pulse.wait().unwrap();
                }
                Err(e) => {
                    eprintln!("async_compute_dispatch not supported: {}", e);
                }
            }
        }
        Err(_) => {
            eprintln!("skipping: wave creation failed");
        }
    }
}

// === Timestamp query sets (typed) ===

#[test]
fn timestamp_query_create_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.timestamp_query(4);
    match result {
        Ok(query) => {
            assert_eq!(query.count(), 4);
        }
        Err(e) => {
            eprintln!("timestamp queries not supported: {}", e);
        }
    }
}

#[test]
fn timestamp_query_read_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    match gpu.timestamp_query(2) {
        Ok(query) => {
            let result = gpu.read_timestamps(&query);
            match result {
                Ok(values) => {
                    assert_eq!(values.len(), 2);
                }
                Err(e) => {
                    eprintln!("timestamp read failed: {}", e);
                }
            }
        }
        Err(_) => {
            eprintln!("skipping: timestamp queries not supported");
        }
    }
}

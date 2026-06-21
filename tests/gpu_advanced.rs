#![cfg(feature = "render")]
//! Tier 2 -- Advanced features (error handling for unsupported paths).
//!
//! Verifies that features returning "not supported" return proper Err, not panic.
//! Tests: ray tracing, mesh shaders, sparse textures, indirect buffers, bindless.
//! Requires a GPU; skips gracefully if none available.

use quanta::ray_tracing::RayTracingPipelineDesc;
use quanta::{Format, PipelineDesc, TextureDesc, TextureUsage};

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

    let result = gpu.build_acceleration_structure(&[]);
    // Empty geometry should either succeed or return Err -- never panic.
    match result {
        Ok(handle) => {
            // Clean up.
            let _ = gpu.destroy_acceleration_structure(handle);
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

    let result = gpu.create_ray_tracing_pipeline(&desc);
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

    // Dummy pipeline handle.
    let result = gpu.dispatch_rays(0, 64, 64);
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("dispatch_rays not supported (expected): {}", e);
        }
    }
}

#[test]
fn destroy_acceleration_structure_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.destroy_acceleration_structure(0);
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!(
                "destroy_acceleration_structure not supported (expected): {}",
                e
            );
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

    // Create a minimal pipeline for mesh dispatch.
    let desc = PipelineDesc {
        color_formats: vec![Format::RGBA8],
        ..PipelineDesc::default()
    };

    // Try creating a pipeline -- if it fails, test dispatch_mesh with dummy handle.
    match gpu.pipeline(&desc) {
        Ok(pipeline) => {
            let result = gpu.dispatch_mesh(&pipeline, [1, 1, 1]);
            match result {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("dispatch_mesh not supported (expected): {}", e);
                }
            }
        }
        Err(_) => {
            // Pipeline creation failed -- skip.
            eprintln!("skipping dispatch_mesh: pipeline creation failed");
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

    let desc = TextureDesc {
        width: 256,
        height: 256,
        format: Format::RGBA8,
        usage: TextureUsage::SHADER_READ,
        ..TextureDesc::default()
    };

    let result = gpu.sparse_texture_create(&desc);
    match result {
        Ok(_handle) => {}
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

    let result = gpu.sparse_map_tile(0, 0, 0, 0, 0);
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("sparse_map_tile not supported (expected): {}", e);
        }
    }
}

#[test]
fn sparse_unmap_tile_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.sparse_unmap_tile(0, 0, 0, 0);
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("sparse_unmap_tile not supported (expected): {}", e);
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

    let result = gpu.indirect_buffer_create(16);
    match result {
        Ok(handle) => {
            // Clean up.
            let _ = gpu.indirect_buffer_destroy(handle);
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

    let result = gpu.indirect_buffer_execute(0, 0);
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("indirect_buffer_execute not supported (expected): {}", e);
        }
    }
}

#[test]
fn indirect_buffer_destroy_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.indirect_buffer_destroy(0);
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("indirect_buffer_destroy not supported (expected): {}", e);
        }
    }
}

// === Bindless Resources ===

#[test]
fn bind_texture_array_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.bind_texture_array(&[]);
    match result {
        Ok(_handle) => {}
        Err(e) => {
            eprintln!("bindless textures not supported (expected): {}", e);
        }
    }
}

#[test]
fn bind_buffer_array_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.bind_buffer_array(&[]);
    match result {
        Ok(_handle) => {}
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
        .create_texture(&TextureDesc {
            width: 16,
            height: 16,
            format: Format::Depth32Float,
            usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
            ..TextureDesc::default()
        })
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

// === Query Sets (generic) ===

#[test]
fn query_set_create_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.query_set(4);
    match result {
        Ok(handle) => {
            assert!(handle != 0);
        }
        Err(e) => {
            eprintln!("query_set_create not supported: {}", e);
        }
    }
}

#[test]
fn query_set_read_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    match gpu.query_set(2) {
        Ok(handle) => {
            let result = gpu.read_queries(handle, 0, 2);
            match result {
                Ok(values) => {
                    assert_eq!(values.len(), 2);
                }
                Err(e) => {
                    eprintln!("query_set_read failed: {}", e);
                }
            }
        }
        Err(_) => {
            eprintln!("skipping: query sets not supported");
        }
    }
}

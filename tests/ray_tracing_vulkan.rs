//! End-to-end Vulkan acceleration-structure build test —
//! step 063 slice 23.
//!
//! Allocates a tiny vertex buffer (one triangle, three R32G32B32
//! verts), passes it to `gpu.acceleration_structure_blas`, and
//! asserts that the build either succeeds (RT-capable device:
//! lavapipe with VK_KHR_acceleration_structure) or surfaces
//! NotSupported (V3D, software-only Vulkan implementations).
//!
//! Run on Vulkan host:
//!     VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
//!     cargo test --no-default-features --features vulkan \
//!         --test ray_tracing_vulkan -- --nocapture

#![cfg(feature = "vulkan")]
#![cfg(not(feature = "metal"))]

use quanta::QuantaErrorKind;

fn try_vulkan() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn build_blas_either_succeeds_or_surfaces_capability_error() {
    let Some(gpu) = try_vulkan() else {
        eprintln!("skipping: no Vulkan GPU available");
        return;
    };
    eprintln!(
        "vulkan device: name={:?} supports_ray_tracing={}",
        gpu.caps().name,
        gpu.supports_ray_tracing()
    );

    // 3 vertices × 3 floats (R32G32B32_SFLOAT) = 9 floats.
    let verts: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let vbuf = gpu.field::<f32>(verts.len()).expect("vertex field alloc");
    vbuf.write(&verts).expect("vertex field write");

    let r = gpu.acceleration_structure_blas(&[quanta::GeometryDesc {
        vertices: vbuf.handle(),
        indices: None,
        vertex_count: 3,
        index_count: 0,
        vertex_stride: 12,
    }]);

    match r {
        Ok(blas) => {
            assert!(
                gpu.supports_ray_tracing(),
                "build_blas succeeded but supports_ray_tracing() = false — caches out of sync"
            );
            eprintln!(
                "BLAS build succeeded — handle = {:#x}, geom_count = {}",
                blas.handle,
                blas.geom_count()
            );
            // Drop fires destroy_acceleration_structure.
            drop(blas);
        }
        Err(e) => {
            assert!(
                matches!(e.kind, QuantaErrorKind::NotSupported(_)),
                "expected NotSupported, got {:?}",
                e.kind
            );
            eprintln!("BLAS build not supported on this device: {}", e);
        }
    }
}

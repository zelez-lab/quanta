# Expert: Ray Tracing

Hardware-accelerated ray tracing via Quanta's shader macros. Requires GPU
support (NVIDIA RTX, AMD RDNA 2+, Apple M3+ with Metal 3).

## Shader types

Quanta provides three ray tracing shader macros:

```rust
use quanta::*;

/// Entry point -- launched once per pixel (or per ray).
#[quanta::ray_gen]
fn camera_rays(pixel: UVec2, scene: &AccelerationStructure) {
    let ray = compute_ray(pixel);
    trace_ray(scene, ray, 0, 1000.0);
}

/// Invoked when a ray hits the nearest surface.
#[quanta::closest_hit]
fn shade(hit: HitInfo, ray: Ray) -> Vec4 {
    let normal = hit.normal;
    let color = sample_texture(hit.uv);
    color * dot(normal, light_dir).max(0.0)
}

/// Invoked when a ray misses all geometry (sky/environment).
#[quanta::miss]
fn sky(ray: Ray) -> Vec4 {
    let t = 0.5 * (ray.direction.y + 1.0);
    Vec4::lerp(Vec4::splat(1.0), Vec4::new(0.5, 0.7, 1.0, 1.0), t)
}
```

## Pipeline

Ray tracing pipelines bundle all shader stages:

- **Ray generation** -- entry point, fires rays
- **Closest hit** -- shading at intersection points
- **Miss** -- background/sky color
- **Any hit** (optional) -- transparency, alpha testing
- **Intersection** (optional) -- custom geometry intersection

The pipeline is created similarly to render pipelines but uses the ray
tracing shader binaries and an acceleration structure.

## Acceleration structures

Build a bottom-level acceleration structure (BLAS) from triangle geometry,
then a top-level structure (TLAS) that references instances of the BLAS:

```rust
let blas = gpu.build_blas(&mesh_vertices, &mesh_indices)?;
let tlas = gpu.build_tlas(&[
    Instance { blas: &blas, transform: identity_matrix },
    Instance { blas: &blas, transform: translation_matrix },
])?;
```

## Secondary rays

Fire secondary rays from hit shaders for reflections, shadows, and
global illumination:

```rust
#[quanta::closest_hit]
fn reflective(hit: HitInfo, ray: Ray) -> Vec4 {
    let reflected_dir = reflect(ray.direction, hit.normal);
    let reflected_ray = Ray::new(hit.position, reflected_dir);
    trace_ray(scene, reflected_ray, 0, 1000.0)
}
```

## Platform support

| Platform | API | Notes |
|----------|-----|-------|
| NVIDIA RTX | Vulkan RT | Full support via SPIR-V |
| AMD RDNA 2+ | Vulkan RT | Full support via SPIR-V |
| Apple M3+ | Metal 3 | Ray tracing via intersection functions |
| Older GPUs | -- | Not supported (falls back to compute) |

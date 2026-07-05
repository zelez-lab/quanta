# Shadow Mapping

Two-pass shadow mapping with depth rendering and comparison sampling.

## Overview

1. **Shadow pass**: Render the scene from the light's perspective into a depth texture.
2. **Lighting pass**: Render from the camera. Sample the shadow map with a comparison
   sampler to determine if each fragment is in shadow.

## Shaders

### Shadow pass (depth only)

```rust
#[repr(C)]
#[derive(Copy, Clone)]
struct LightUniforms {
    light_view_proj: [f32; 16], // 4x4 matrix
}

#[quanta::vertex]
fn shadow_vertex(position: Vec3, light: &LightUniforms) -> Vec4 {
    mat4_mul_vec4(light.light_view_proj, vec4(position.x, position.y, position.z, 1.0))
}

// No fragment shader needed -- depth-only pass writes to depth buffer automatically.
```

### Lighting pass

```rust
#[repr(C)]
#[derive(Copy, Clone)]
struct CameraUniforms {
    view_proj: [f32; 16],
    light_view_proj: [f32; 16],
    light_dir: [f32; 4],
}

#[quanta::vertex]
fn scene_vertex(position: Vec3, normal: Vec3, camera: &CameraUniforms) -> SceneVaryings {
    let clip_pos = mat4_mul_vec4(camera.view_proj, vec4(position.x, position.y, position.z, 1.0));
    let light_pos = mat4_mul_vec4(camera.light_view_proj, vec4(position.x, position.y, position.z, 1.0));
    SceneVaryings { clip_pos, light_pos, normal }
}

#[quanta::fragment]
fn scene_fragment(
    varying: SceneVaryings,
    shadow_map: &Texture2D<f32>,
    camera: &CameraUniforms,
) -> Vec4 {
    // Project into shadow map UV space
    let shadow_uv = vec2(
        varying.light_pos.x / varying.light_pos.w * 0.5 + 0.5,
        varying.light_pos.y / varying.light_pos.w * -0.5 + 0.5,
    );
    let frag_depth = varying.light_pos.z / varying.light_pos.w;

    // Comparison sample: returns 0.0 (in shadow) or 1.0 (lit)
    let shadow = texture_compare(shadow_map, shadow_uv.x, shadow_uv.y, frag_depth);

    // Simple directional lighting
    let n_dot_l = dot(varying.normal, vec3(camera.light_dir[0], camera.light_dir[1], camera.light_dir[2]));
    let diffuse = max(n_dot_l, 0.0);
    let ambient = 0.15;
    let brightness = ambient + diffuse * shadow;

    vec4(brightness, brightness, brightness, 1.0)
}
```

## Host code

The render methods (`gpu.render_target`, `gpu.pipeline`, `gpu.render`) live
on the `RenderGpu` extension trait — bring it into scope (or `use quanta::*;`).

```rust
use quanta::{
    Color, CompareOp, DepthStencilState, Format, PipelineDesc, RenderGpu,
    SamplerDesc, ShaderSource, TextureDesc, TextureUsage,
};

fn main() {
    let gpu = quanta::init().unwrap();

    let shadow_size = 2048;
    let screen_width = 1920;
    let screen_height = 1080;

    // --- Create shadow map (depth-only texture) ---
    let shadow_map = gpu.create_texture(
        &TextureDesc::new(shadow_size, shadow_size, Format::Depth32Float)
            .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)),
    ).unwrap();

    // --- Shadow pass pipeline (depth only, no color) ---
    // #[quanta::vertex] fn shadow_vertex generates the
    // SHADOW_VERTEX_SHADER static (a multi-vendor ShaderBinary).
    let shadow_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Stages {
            vertex: SHADOW_VERTEX_SHADER.for_vendor(gpu.caps().vendor).unwrap(),
            fragment: &[], // No fragment shader for depth-only
        })
        .with_entries("shadow_vertex", "")
        .with_color_formats(vec![])
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
    ).unwrap();

    // --- Create render targets for the scene ---
    let color_target = gpu.render_target(screen_width, screen_height, Format::BGRA8).unwrap();
    let depth_target = gpu.create_texture(
        &TextureDesc::new(screen_width, screen_height, Format::Depth32Float)
            .with_usage(TextureUsage::RENDER_TARGET),
    ).unwrap();

    // --- Scene pipeline with shadow sampling ---
    let scene_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &SCENE_VERTEX_SHADER,
            fragment: &SCENE_FRAGMENT_SHADER,
        })
        .with_entries("scene_vertex", "scene_fragment")
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
    ).unwrap();

    // --- Comparison sampler for shadow lookups ---
    let shadow_sampler_desc = SamplerDesc::default()
        .with_compare(CompareOp::LessEqual);

    // --- Pass 1: Render shadow map ---
    let mut pulse = gpu.render(&shadow_map).unwrap()
        .clear_depth(1.0)
        .pipeline(&shadow_pipeline)
        // ... bind vertex buffers, uniforms ...
        .draw(scene_vertex_count)
        .pulse().unwrap();
    pulse.wait().unwrap();

    // --- Pass 2: Render scene with shadows ---
    let mut pulse = gpu.render(&color_target).unwrap()
        .clear(Color::rgb(0.1, 0.1, 0.15))
        .clear_depth(1.0)
        .pipeline(&scene_pipeline)
        .texture(0, &shadow_map)
        .sampler(0, shadow_sampler_desc)
        // ... bind vertex buffers, camera uniforms ...
        .draw(scene_vertex_count)
        .pulse().unwrap();
    pulse.wait().unwrap();
}
```

## Comparison sampler

A comparison sampler does not return the texture value. Instead, it compares the
sampled depth against a reference value and returns 0.0 or 1.0.

```rust
SamplerDesc::default().with_compare(CompareOp::LessEqual)
```

With hardware PCF (percentage-closer filtering), the `Linear` filter mode
produces soft shadow edges by averaging comparison results from neighboring
texels.

## Shadow acne prevention

Add a small depth bias to prevent self-shadowing artifacts:

```rust
let frag_depth = varying.light_pos.z / varying.light_pos.w - 0.005;
```

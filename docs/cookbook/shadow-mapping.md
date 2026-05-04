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

```rust
use quanta::{
    Color, CompareOp, DepthStencilState, Format, LoadOp, PipelineDesc, SamplerDesc, StoreOp,
    TextureDesc, TextureUsage,
};

fn main() {
    let gpu = quanta::init().unwrap();

    let shadow_size = 2048;
    let screen_width = 1920;
    let screen_height = 1080;

    // --- Create shadow map (depth-only texture) ---
    let shadow_map = gpu.create_texture(&TextureDesc {
        width: shadow_size,
        height: shadow_size,
        format: Format::Depth32Float,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        ..TextureDesc::default()
    }).unwrap();

    // --- Shadow pass pipeline (depth only, no color) ---
    let shadow_pipeline = gpu.pipeline(&PipelineDesc {
        vertex: shadow_vertex().for_vendor(gpu.caps().vendor).unwrap(),
        fragment: &[], // No fragment shader for depth-only
        vertex_entry: "shadow_vertex",
        fragment_entry: "",
        color_formats: vec![],
        depth_format: Some(Format::Depth32Float),
        depth_stencil: DepthStencilState::DEPTH_LESS,
        ..PipelineDesc::default()
    }).unwrap();

    // --- Create render targets for the scene ---
    let color_target = gpu.render_target(screen_width, screen_height, Format::BGRA8).unwrap();
    let depth_target = gpu.create_texture(&TextureDesc {
        width: screen_width,
        height: screen_height,
        format: Format::Depth32Float,
        usage: TextureUsage::RENDER_TARGET,
        ..TextureDesc::default()
    }).unwrap();

    // --- Scene pipeline with shadow sampling ---
    let scene_pipeline = gpu.pipeline(&PipelineDesc {
        vertex: scene_vertex().for_vendor(gpu.caps().vendor).unwrap(),
        fragment: scene_fragment().for_vendor(gpu.caps().vendor).unwrap(),
        vertex_entry: "scene_vertex",
        fragment_entry: "scene_fragment",
        depth_format: Some(Format::Depth32Float),
        depth_stencil: DepthStencilState::DEPTH_LESS,
        ..PipelineDesc::default()
    }).unwrap();

    // --- Comparison sampler for shadow lookups ---
    let shadow_sampler = gpu.sampler(&SamplerDesc {
        compare: Some(CompareOp::LessEqual),
        ..SamplerDesc::default()
    }).unwrap();

    // --- Pass 1: Render shadow map ---
    let mut pass = gpu.render_begin(&shadow_map).unwrap();
    pass.clear_depth(1.0);
    pass.set_pipeline(&shadow_pipeline);
    // ... bind vertex buffers, uniforms, draw scene geometry ...
    pass.draw(scene_vertex_count);
    let mut pulse = gpu.render_end(pass).unwrap();
    pulse.wait().unwrap();

    // --- Pass 2: Render scene with shadows ---
    let mut pass = gpu.render_begin(&color_target).unwrap();
    pass.clear(Color::rgb(0.1, 0.1, 0.15));
    pass.clear_depth(1.0);
    pass.set_pipeline(&scene_pipeline);
    pass.set_texture(0, &shadow_map);
    pass.set_sampler(0, SamplerDesc {
        compare: Some(CompareOp::LessEqual),
        ..SamplerDesc::default()
    });
    // ... bind vertex buffers, camera uniforms, draw scene ...
    pass.draw(scene_vertex_count);
    let mut pulse = gpu.render_end(pass).unwrap();
    pulse.wait().unwrap();
}
```

## Comparison sampler

A comparison sampler does not return the texture value. Instead, it compares the
sampled depth against a reference value and returns 0.0 or 1.0.

```
SamplerDesc {
    compare: Some(CompareOp::LessEqual),
    ..SamplerDesc::default()
}
```

With hardware PCF (percentage-closer filtering), the `Linear` filter mode
produces soft shadow edges by averaging comparison results from neighboring
texels.

## Shadow acne prevention

Add a small depth bias to prevent self-shadowing artifacts:

```rust
let frag_depth = varying.light_pos.z / varying.light_pos.w - 0.005;
```

# Deferred Rendering

G-buffer pass with multiple render targets (MRT), followed by a fullscreen
lighting pass. Demonstrates MRT setup and multi-texture sampling.

## Overview

1. **G-buffer pass**: Render scene geometry, writing albedo, normals, and depth
   into separate render targets simultaneously.
2. **Lighting pass**: Fullscreen quad reads from all G-buffer textures and
   computes final shading.

## Shaders

### G-buffer pass

```rust
#[repr(C)]
#[derive(Copy, Clone)]
struct GBufferOut {
    albedo: Vec4,   // color target 0
    normal: Vec4,   // color target 1
    position: Vec4, // color target 2
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Uniforms {
    model_view_proj: [f32; 16],
    model: [f32; 16],
}

#[quanta::vertex]
fn gbuffer_vertex(pos: Vec3, normal: Vec3, uv: Vec2, uniforms: &Uniforms) -> GBufferVarying {
    let clip_pos = mat4_mul_vec4(uniforms.model_view_proj, vec4(pos.x, pos.y, pos.z, 1.0));
    let world_pos = mat4_mul_vec4(uniforms.model, vec4(pos.x, pos.y, pos.z, 1.0));
    let world_normal = mat4_mul_vec3(uniforms.model, normal);
    GBufferVarying { clip_pos, world_pos, world_normal, uv }
}

#[quanta::fragment]
fn gbuffer_fragment(varying: GBufferVarying) -> GBufferOut {
    GBufferOut {
        albedo: vec4(0.8, 0.2, 0.1, 1.0),     // material color
        normal: vec4(                            // world-space normal
            varying.world_normal.x * 0.5 + 0.5,
            varying.world_normal.y * 0.5 + 0.5,
            varying.world_normal.z * 0.5 + 0.5,
            1.0,
        ),
        position: varying.world_pos,            // world-space position
    }
}
```

### Lighting pass

```rust
#[repr(C)]
#[derive(Copy, Clone)]
struct LightParams {
    light_pos: [f32; 4],
    light_color: [f32; 4],
    camera_pos: [f32; 4],
}

#[quanta::vertex]
fn fullscreen_vertex(vertex_id: u32) -> Vec4 {
    // Fullscreen triangle: 3 vertices, no vertex buffer needed
    let x = (vertex_id & 1u32) as f32 * 4.0 - 1.0;
    let y = (vertex_id >> 1u32) as f32 * 4.0 - 1.0;
    vec4(x, y, 0.0, 1.0)
}

#[quanta::fragment]
fn lighting_fragment(
    uv: Vec2,
    albedo_tex: &Texture2D<f32>,
    normal_tex: &Texture2D<f32>,
    position_tex: &Texture2D<f32>,
    lights: &LightParams,
) -> Vec4 {
    let albedo = texture_sample(albedo_tex, uv.x, uv.y);
    let normal_raw = texture_sample(normal_tex, uv.x, uv.y);
    let position = texture_sample(position_tex, uv.x, uv.y);

    // Decode normal from [0,1] to [-1,1]
    let normal = vec3(
        normal_raw.x * 2.0 - 1.0,
        normal_raw.y * 2.0 - 1.0,
        normal_raw.z * 2.0 - 1.0,
    );

    // Point light contribution
    let light_dir = normalize(vec3(
        lights.light_pos[0] - position.x,
        lights.light_pos[1] - position.y,
        lights.light_pos[2] - position.z,
    ));
    let n_dot_l = max(dot(normal, light_dir), 0.0);

    let ambient = vec3(0.05, 0.05, 0.05);
    let diffuse = vec3(albedo.x, albedo.y, albedo.z) * n_dot_l;
    let light_col = vec3(lights.light_color[0], lights.light_color[1], lights.light_color[2]);

    vec4(
        ambient.x + diffuse.x * light_col.x,
        ambient.y + diffuse.y * light_col.y,
        ambient.z + diffuse.z * light_col.z,
        1.0,
    )
}
```

## Host code

```rust
use quanta::{
    Color, ColorTarget, DepthStencilState, Format, LoadOp, PipelineDesc, StoreOp,
    TextureDesc, TextureUsage,
};

fn main() {
    let gpu = quanta::init().unwrap();

    let w = 1920;
    let h = 1080;

    // --- G-buffer textures (3 render targets + depth) ---
    let gbuf_albedo = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let gbuf_normal = gpu.render_target(w, h, Format::RGBA16Float).unwrap();
    let gbuf_position = gpu.render_target(w, h, Format::RGBA32Float).unwrap();
    let gbuf_depth = gpu.create_texture(&TextureDesc {
        width: w,
        height: h,
        format: Format::Depth32Float,
        usage: TextureUsage::RENDER_TARGET,
        ..TextureDesc::default()
    }).unwrap();

    // --- G-buffer pipeline: 3 color attachments + depth ---
    let gbuf_pipeline = gpu.pipeline(&PipelineDesc {
        vertex: gbuffer_vertex().for_vendor(gpu.caps().vendor).unwrap(),
        fragment: gbuffer_fragment().for_vendor(gpu.caps().vendor).unwrap(),
        vertex_entry: "gbuffer_vertex",
        fragment_entry: "gbuffer_fragment",
        color_formats: vec![Format::RGBA8, Format::RGBA16Float, Format::RGBA32Float],
        depth_format: Some(Format::Depth32Float),
        depth_stencil: DepthStencilState::DEPTH_LESS,
        ..PipelineDesc::default()
    }).unwrap();

    // --- G-buffer pass: render geometry into 3 targets ---
    let mut pass = gpu.render_begin(&gbuf_albedo).unwrap();
    pass.set_color_targets(vec![
        ColorTarget { texture: gbuf_albedo.handle(), load_op: LoadOp::Clear(Color::BLACK), store_op: StoreOp::Store },
        ColorTarget { texture: gbuf_normal.handle(), load_op: LoadOp::Clear(Color::BLACK), store_op: StoreOp::Store },
        ColorTarget { texture: gbuf_position.handle(), load_op: LoadOp::Clear(Color::BLACK), store_op: StoreOp::Store },
    ]);
    pass.clear_depth(1.0);
    pass.set_pipeline(&gbuf_pipeline);
    // ... bind vertex buffers, uniforms, draw geometry ...
    pass.draw(vertex_count);
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // --- Lighting pipeline: fullscreen, reads G-buffer textures ---
    let light_pipeline = gpu.pipeline(&PipelineDesc {
        vertex: fullscreen_vertex().for_vendor(gpu.caps().vendor).unwrap(),
        fragment: lighting_fragment().for_vendor(gpu.caps().vendor).unwrap(),
        vertex_entry: "fullscreen_vertex",
        fragment_entry: "lighting_fragment",
        ..PipelineDesc::default()
    }).unwrap();

    let final_target = gpu.render_target(w, h, Format::BGRA8).unwrap();
    let mut pass = gpu.render_begin(&final_target).unwrap();
    pass.set_pipeline(&light_pipeline);
    pass.set_texture(0, &gbuf_albedo);
    pass.set_texture(1, &gbuf_normal);
    pass.set_texture(2, &gbuf_position);
    // ... bind light uniforms ...
    pass.draw(3); // Fullscreen triangle
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();
}
```

## Why deferred rendering

**Forward rendering** evaluates all lights per fragment during geometry rendering.
Cost = O(fragments x lights).

**Deferred rendering** decouples geometry from lighting:
- G-buffer pass: O(fragments), writes material data
- Lighting pass: O(screen_pixels x lights), reads material data

For scenes with many lights (100+), deferred wins because the lighting pass
only runs on visible pixels, not on every triangle fragment.

## G-buffer layout

| Target | Format | Contents |
|--------|--------|----------|
| 0 | RGBA8 | Albedo (diffuse color) |
| 1 | RGBA16Float | World-space normal (encoded) |
| 2 | RGBA32Float | World-space position |
| Depth | Depth32Float | Depth buffer |

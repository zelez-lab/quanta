# Deferred Rendering

Decouple geometry from lighting: a **G-buffer** pass records per-pixel material
data into off-screen textures, then a **fullscreen lighting** pass reads them
back and shades. This page shows the parts of that pipeline that are
expressible in Quanta's shader DSL today, and marks the one part that is not.

> **Not yet in the DSL — multiple render targets from one fragment.**
> A classic G-buffer fill writes albedo, normal, and position in a **single**
> geometry pass, from **one** fragment that returns a struct of outputs
> (`-> GBufferOut`). Quanta's shader DSL fragment returns exactly **one**
> `Vec4` (see [Fragment shaders](../tutorials/vertex-fragment.md#fragment-shaders)),
> so a single fragment cannot fill several attachments at once. Until
> struct-valued fragment outputs land, fill each G-buffer target with its
> **own** geometry pass (shown below). The multi-attachment `ColorTarget` list
> is real at the API level — it is the fragment-side MRT write that is missing.

## Overview

1. **G-buffer passes**: render the scene geometry once per G-buffer channel,
   each pass writing one target (albedo, encoded normal, world position). A
   shared depth buffer keeps the passes consistent.
2. **Lighting pass**: a fullscreen triangle reads all three G-buffer textures
   and computes final shading — this runs once per screen pixel, not once per
   triangle fragment.

## Shaders

### G-buffer passes

The geometry vertex is shared by all three fills; it forwards the world-space
normal and position as varyings. Each fill then has its own single-output
fragment.

```rust
use quanta::*;

// The vertex→fragment interface for the geometry passes.
#[derive(quanta::Varyings)]
struct GbufSurface {
    #[position] clip: Vec4, // gl_Position
    world_normal: Vec3,     // Location 0
    world_pos: Vec3,        // Location 1
}

#[quanta::vertex]
fn gbuffer_vertex(pos: Vec3, normal: Vec3, mvp: &Mat4, model: &Mat4) -> GbufSurface {
    let wp = model * Vec4::new(pos.x, pos.y, pos.z, 1.0);
    let wn = model * Vec4::new(normal.x, normal.y, normal.z, 0.0);
    GbufSurface {
        clip: mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0),
        world_normal: Vec3::new(wn.x, wn.y, wn.z),
        world_pos: Vec3::new(wp.x, wp.y, wp.z),
    }
}

// Target 0 — albedo (material colour). One Vec4 out.
#[quanta::fragment]
fn gbuffer_albedo(s: GbufSurface) -> Vec4 {
    Vec4::new(0.8, 0.2, 0.1, 1.0)
}

// Target 1 — world-space normal, encoded [-1,1] → [0,1]. One Vec4 out.
#[quanta::fragment]
fn gbuffer_normal(s: GbufSurface) -> Vec4 {
    Vec4::new(
        s.world_normal.x * 0.5 + 0.5,
        s.world_normal.y * 0.5 + 0.5,
        s.world_normal.z * 0.5 + 0.5,
        1.0,
    )
}

// Target 2 — world-space position. One Vec4 out.
#[quanta::fragment]
fn gbuffer_position(s: GbufSurface) -> Vec4 {
    Vec4::new(s.world_pos.x, s.world_pos.y, s.world_pos.z, 1.0)
}
```

### Lighting pass

The fullscreen triangle is synthesised from `vertex_id()` — no vertex buffer.
The fragment samples the three G-buffer textures through `&Sampled2D`
parameters and does the point-light shading.

```rust
use quanta::*;

// Fullscreen-pass interface: a uv varying for the G-buffer lookups.
#[derive(quanta::Varyings)]
struct FsQuad {
    #[position] clip: Vec4, // gl_Position
    uv: Vec2,               // Location 0
}

#[quanta::vertex]
fn fullscreen_vertex() -> FsQuad {
    // Three vertices covering the screen: (-1,-1), (3,-1), (-1,3).
    let id = vertex_id();
    let x = ((id & 1u32) as f32) * 4.0 - 1.0;
    let y = ((id >> 1u32) as f32) * 4.0 - 1.0;
    FsQuad {
        clip: Vec4::new(x, y, 0.0, 1.0),
        // Map clip xy to [0,1] texcoords (flip .y if your G-buffer is stored
        // top-left origin — see the coordinate conventions in the tutorial).
        uv: Vec2::new((x + 1.0) * 0.5, (y + 1.0) * 0.5),
    }
}

#[quanta::fragment]
fn lighting_fragment(
    s: FsQuad,
    albedo_tex: &Sampled2D,
    normal_tex: &Sampled2D,
    position_tex: &Sampled2D,
    light_pos: &Vec4,
    light_color: &Vec4,
) -> Vec4 {
    let albedo = sample(albedo_tex, s.uv);
    let normal_raw = sample(normal_tex, s.uv);
    let position = sample(position_tex, s.uv);

    // Decode the normal from [0,1] back to [-1,1].
    let normal = Vec3::new(
        normal_raw.x * 2.0 - 1.0,
        normal_raw.y * 2.0 - 1.0,
        normal_raw.z * 2.0 - 1.0,
    );

    // Point-light contribution.
    let light_dir = normalize(Vec3::new(
        light_pos.x - position.x,
        light_pos.y - position.y,
        light_pos.z - position.z,
    ));
    let n_dot_l = max(dot(normal, light_dir), 0.0);

    let ambient = 0.05;
    Vec4::new(
        ambient + albedo.x * n_dot_l * light_color.x,
        ambient + albedo.y * n_dot_l * light_color.y,
        ambient + albedo.z * n_dot_l * light_color.z,
        1.0,
    )
}
```

The light position and colour arrive as separate `&Vec4` uniform parameters
(each bound to its own uniform buffer), read directly as `light_pos.x` and so
on — the DSL reads uniform vectors by component, not by indexing a struct
field.

## Host code

The render methods (`gpu.render_target`, `gpu.pipeline`, `gpu.render`) live
on the `RenderGpu` extension trait — bring it into scope (or `use quanta::*;`).

```rust
use quanta::{
    Color, ColorTarget, DepthStencilState, DepthTarget, Format, PipelineDesc,
    RenderGpu, ShaderSource, TextureDesc, TextureUsage,
};

fn main() {
    let gpu = quanta::init().unwrap();

    let w = 1920;
    let h = 1080;

    // --- G-buffer textures (3 colour targets + a shared depth buffer) ---
    let gbuf_albedo = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let gbuf_normal = gpu.render_target(w, h, Format::RGBA16Float).unwrap();
    let gbuf_position = gpu.render_target(w, h, Format::RGBA32Float).unwrap();
    let gbuf_depth = gpu.create_texture(
        &TextureDesc::new(w, h, Format::Depth32Float)
            .with_usage(TextureUsage::RENDER_TARGET),
    ).unwrap();

    // --- One pipeline per G-buffer channel: the shared vertex, a distinct
    // single-output fragment, and the ONE colour format that fragment writes.
    // (One fragment fills one target — see the box at the top.) ---
    let albedo_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &GBUFFER_VERTEX_SHADER,
            fragment: &GBUFFER_ALBEDO_SHADER,
        })
        .with_entries("gbuffer_vertex", "gbuffer_albedo")
        .with_color_formats(vec![Format::RGBA8])
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
    ).unwrap();

    let normal_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &GBUFFER_VERTEX_SHADER,
            fragment: &GBUFFER_NORMAL_SHADER,
        })
        .with_entries("gbuffer_vertex", "gbuffer_normal")
        .with_color_formats(vec![Format::RGBA16Float])
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
    ).unwrap();

    let position_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &GBUFFER_VERTEX_SHADER,
            fragment: &GBUFFER_POSITION_SHADER,
        })
        .with_entries("gbuffer_vertex", "gbuffer_position")
        .with_color_formats(vec![Format::RGBA32Float])
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
    ).unwrap();

    // --- Fill each G-buffer target with its own geometry pass ---
    // `ColorTarget::new` defaults to LoadOp::Clear(Color::BLACK) + StoreOp::Store.
    // The three passes render the same geometry under the same depth test, so
    // the visible surface — hence the material data — agrees across targets.
    for (pipeline, target) in [
        (&albedo_pipeline, &gbuf_albedo),
        (&normal_pipeline, &gbuf_normal),
        (&position_pipeline, &gbuf_position),
    ] {
        let mut pulse = gpu.render(target).unwrap()
            .color_targets(vec![ColorTarget::new(target)])
            .depth_target(DepthTarget::new(&gbuf_depth))
            .clear_depth(1.0)
            .pipeline(pipeline)
            // ... bind vertex buffers, mvp + model uniforms ...
            .draw(vertex_count)
            .pulse().unwrap();
        pulse.wait().unwrap();
    }

    // --- Lighting pipeline: fullscreen, reads the three G-buffer textures ---
    let light_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &FULLSCREEN_VERTEX_SHADER,
            fragment: &LIGHTING_FRAGMENT_SHADER,
        })
        .with_entries("fullscreen_vertex", "lighting_fragment")
        .with_color_formats(vec![Format::BGRA8]),
    ).unwrap();

    let final_target = gpu.render_target(w, h, Format::BGRA8).unwrap();
    let mut pulse = gpu.render(&final_target).unwrap()
        .pipeline(&light_pipeline)
        .texture(0, &gbuf_albedo)
        .texture(1, &gbuf_normal)
        .texture(2, &gbuf_position)
        // ... bind the light_pos + light_color uniforms ...
        .draw(3) // Fullscreen triangle — no vertex buffer bound
        .pulse().unwrap();
    pulse.wait().unwrap();
}
```

Each color pipeline's `color_formats` is per-attachment and must line up with
the targets the pass binds — a wrong count or a swapped format fails `pulse()`
with a named error rather than misrendering silently. The lighting pipeline's
texture slots (0/1/2) follow the declaration order of the `&Sampled2D`
parameters in `lighting_fragment`.

## Why deferred rendering

**Forward rendering** evaluates all lights per fragment during geometry
rendering. Cost = O(fragments × lights).

**Deferred rendering** decouples geometry from lighting:
- G-buffer passes: O(fragments), write material data
- Lighting pass: O(screen_pixels × lights), read material data

For scenes with many lights (100+), deferred wins because the lighting pass
only runs on visible pixels, not on every triangle fragment. The per-channel
fill above costs one geometry pass per target today; when struct-valued
fragment outputs land, those collapse into a single MRT pass and the geometry
is rasterised once.

## G-buffer layout

| Target | Format | Contents |
|--------|--------|----------|
| 0 | RGBA8 | Albedo (diffuse color) |
| 1 | RGBA16Float | World-space normal (encoded) |
| 2 | RGBA32Float | World-space position |
| Depth | Depth32Float | Shared depth buffer |

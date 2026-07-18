# Shadow Mapping

Two-pass shadow mapping: render the scene depth from the light's point of view,
then compare each camera-visible fragment's light-space depth against the stored
value. Quanta's shader DSL has no hardware comparison sampler, so the depth test
is done **in the shader** — this page shows that form and marks what is missing.

> **Not yet in the DSL — comparison (depth-compare) samplers.**
> Hardware shadow mapping samples the depth texture through a **comparison
> sampler** that returns `0.0`/`1.0` (in-shadow / lit) directly, and with a
> linear filter averages neighbouring comparisons for free (hardware PCF).
> Quanta's render DSL has no comparison-sample intrinsic and no
> `SamplerDesc::with_compare` today: `sample(tex, uv)` returns the stored
> `Vec4`, nothing more. The workaround here **stores light-space depth in a
> colour target** and compares it against the fragment's depth **with an
> ordinary `if`** in the shader. Missing pieces: comparison samplers
> (`CompareOp`) and hardware-PCF soft edges.

## Overview

1. **Shadow pass**: render the scene from the light, writing light-space depth
   into a colour texture (the "shadow map"), with its own depth buffer so only
   the nearest surface's depth is kept.
2. **Lighting pass**: render from the camera. Sample the shadow map, reconstruct
   the fragment's light-space depth, and compare the two in the shader to decide
   lit vs. shadowed.

## Shaders

### Shadow pass (write light-space depth)

The vertex forwards the light-space clip position as a varying; the fragment
divides it per-fragment (perspective-correct) and stores the normalized depth
in every channel of the colour target.

```rust
use quanta::*;

#[derive(quanta::Varyings)]
struct ShadowSurface {
    #[position] clip: Vec4, // gl_Position (light view-projection)
    ndc: Vec4,              // Location 0 — same clip position, divided per-fragment
}

#[quanta::vertex]
fn shadow_vertex(position: Vec3, light_view_proj: &Mat4) -> ShadowSurface {
    let clip = light_view_proj * Vec4::new(position.x, position.y, position.z, 1.0);
    ShadowSurface { clip, ndc: clip }
}

#[quanta::fragment]
fn shadow_depth(s: ShadowSurface) -> Vec4 {
    // Normalized light-space depth — the exact quantity the lighting pass
    // reconstructs and compares against.
    let d = s.ndc.z / s.ndc.w;
    Vec4::new(d, d, d, 1.0)
}
```

### Lighting pass

```rust
use quanta::*;

#[derive(quanta::Varyings)]
struct SceneVaryings {
    #[position] clip: Vec4, // gl_Position (camera view-projection)
    light_pos: Vec4,        // Location 0 — light-space clip position
    normal: Vec3,           // Location 1
}

#[quanta::vertex]
fn scene_vertex(
    position: Vec3,
    normal: Vec3,
    view_proj: &Mat4,
    light_view_proj: &Mat4,
) -> SceneVaryings {
    SceneVaryings {
        clip: view_proj * Vec4::new(position.x, position.y, position.z, 1.0),
        light_pos: light_view_proj * Vec4::new(position.x, position.y, position.z, 1.0),
        normal,
    }
}

#[quanta::fragment]
fn scene_fragment(
    s: SceneVaryings,
    shadow_map: &Texture2D,
    light_dir: &Vec4,
) -> Vec4 {
    // Project into shadow-map UV space.
    let shadow_uv = Vec2::new(
        s.light_pos.x / s.light_pos.w * 0.5 + 0.5,
        s.light_pos.y / s.light_pos.w * -0.5 + 0.5,
    );
    let frag_depth = s.light_pos.z / s.light_pos.w;

    // Read the stored nearest depth and compare it in the shader. A small bias
    // fights self-shadowing "acne". (No hardware comparison sampler.)
    let closest = sample(shadow_map, shadow_uv).x;
    let lit = if frag_depth - 0.005 <= closest { 1.0 } else { 0.0 };

    // Simple directional lighting.
    let n_dot_l = dot(s.normal, Vec3::new(light_dir.x, light_dir.y, light_dir.z));
    let diffuse = max(n_dot_l, 0.0);
    let ambient = 0.15;
    let brightness = ambient + diffuse * lit;

    Vec4::new(brightness, brightness, brightness, 1.0)
}
```

The light direction arrives as a `&Vec4` uniform read by component
(`light_dir.x`), and `shadow_map` is a `&Texture2D` sampled with the ordinary
`sample` intrinsic — the same colour sample as any other texture; the
comparison is the plain `if` above.

## Host code

The render methods (`gpu.render_target`, `gpu.pipeline`, `gpu.render`) live
on the `RenderGpu` extension trait — bring it into scope (or `use quanta::*;`).

```rust
use quanta::{
    Color, DepthStencilState, DepthTarget, Format, PipelineDesc, RenderGpu,
    SamplerDesc, ShaderSource, TextureDesc, TextureUsage,
};

fn main() {
    let gpu = quanta::init().unwrap();

    let shadow_size = 2048;
    let screen_width = 1920;
    let screen_height = 1080;

    // --- Shadow map: a COLOUR texture holding light-space depth, readable in
    // the lighting pass. (A colour target, not a depth texture, because we
    // sample it as an ordinary texture and compare in the shader.) ---
    let shadow_map = gpu.create_texture(
        &TextureDesc::new(shadow_size, shadow_size, Format::R32Float)
            .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)),
    ).unwrap();
    // The shadow pass still needs its own depth buffer so only the nearest
    // surface's depth is written into the map.
    let shadow_depth = gpu.create_texture(
        &TextureDesc::new(shadow_size, shadow_size, Format::Depth32Float)
            .with_usage(TextureUsage::RENDER_TARGET),
    ).unwrap();

    // --- Shadow pass pipeline (writes light-space depth to R32Float) ---
    let shadow_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &SHADOW_VERTEX_SHADER,
            fragment: &SHADOW_DEPTH_SHADER,
        })
        .with_entries("shadow_vertex", "shadow_depth")
        .with_color_formats(vec![Format::R32Float])
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
    ).unwrap();

    // --- Scene render targets ---
    let color_target = gpu.render_target(screen_width, screen_height, Format::BGRA8).unwrap();
    let depth_target = gpu.create_texture(
        &TextureDesc::new(screen_width, screen_height, Format::Depth32Float)
            .with_usage(TextureUsage::RENDER_TARGET),
    ).unwrap();

    // --- Scene pipeline: samples the shadow map, compares in-shader ---
    let scene_pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &SCENE_VERTEX_SHADER,
            fragment: &SCENE_FRAGMENT_SHADER,
        })
        .with_entries("scene_vertex", "scene_fragment")
        .with_color_formats(vec![Format::BGRA8])
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
    ).unwrap();

    // --- Pass 1: render the shadow map from the light ---
    let mut pulse = gpu.render(&shadow_map).unwrap()
        .depth_target(DepthTarget::new(&shadow_depth))
        .clear_depth(1.0)
        .pipeline(&shadow_pipeline)
        // ... bind vertex buffers, light_view_proj uniform ...
        .draw(scene_vertex_count)
        .pulse().unwrap();
    pulse.wait().unwrap();

    // --- Pass 2: render the scene with shadows ---
    // An ordinary (non-comparison) sampler — the compare is in the shader.
    let mut pulse = gpu.render(&color_target).unwrap()
        .clear(Color::rgb(0.1, 0.1, 0.15))
        .clear_depth(1.0)
        .depth_target(DepthTarget::new(&depth_target))
        .pipeline(&scene_pipeline)
        .texture(0, &shadow_map)
        .sampler(0, SamplerDesc::default())
        // ... bind vertex buffers, view_proj + light_view_proj + light_dir uniforms ...
        .draw(scene_vertex_count)
        .pulse().unwrap();
    pulse.wait().unwrap();
}
```

## The in-shader depth test

Because there is no comparison sampler, the shadow decision is an ordinary
value-`if` on two floats:

```rust
let closest = sample(shadow_map, shadow_uv).x; // nearest depth from the light
let lit = if frag_depth - 0.005 <= closest { 1.0 } else { 0.0 };
```

`sample()` returns the stored `Vec4`; `.x` is the light-space depth the shadow
pass wrote. The `if` yields `1.0` (lit) or `0.0` (shadowed).

## Shadow acne prevention

The `- 0.005` bias in the comparison offsets the fragment's depth toward the
light so a surface does not shadow itself:

```rust
let lit = if frag_depth - 0.005 <= closest { 1.0 } else { 0.0 };
```

Too small a bias leaves acne; too large detaches the shadow from the caster
("peter-panning"). Tune it per scene.

> **Softer edges — not yet.** Hardware PCF (a comparison sampler with a linear
> filter) averages several neighbouring comparisons in one fetch for soft edges.
> Without it you would emulate PCF by sampling the map at several offsets in a
> bounded `for` loop and averaging the per-tap `if` results by hand — expressible
> in principle (bounded loops and `sample` are in the DSL), but there is no
> single-fetch hardware path today.

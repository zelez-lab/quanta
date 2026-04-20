# Shader Stages

The GPU render pipeline processes geometry through fixed stages. Some stages
are programmable (you write the code), others are fixed-function hardware.

## Pipeline diagram

```
Vertex data (positions, normals, UVs)
         |
         v
+-------------------+
| Vertex stage      |  #[quanta::vertex]    -- runs once per vertex
| (programmable)    |  transforms positions to clip space
+-------------------+
         |
         v
+-------------------+
| Rasterizer        |  fixed-function hardware
| (not programmable)|  converts triangles to fragments (pixels)
+-------------------+
         |
         v
+-------------------+
| Fragment stage    |  #[quanta::fragment]   -- runs once per fragment
| (programmable)    |  computes pixel color, reads textures
+-------------------+
         |
         v
+-------------------+
| Blending          |  configurable (not programmable)
|                   |  combines fragment with existing framebuffer
+-------------------+
         |
         v
    Framebuffer (final image)
```

## Vertex stage

Runs once per vertex. Transforms model-space positions to clip-space.

```rust
#[quanta::vertex]
fn transform(pos: Vec3, normal: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}
```

Inputs: vertex attributes (position, normal, UV, color, etc.)
Output: clip-space position (Vec4). The `w` component is used for perspective division.

## Rasterization

Not programmable. Hardware does this:
1. Clips triangles to the view frustum.
2. Projects to screen coordinates (perspective division).
3. Determines which pixels are covered by each triangle.
4. Interpolates vertex outputs (normals, UVs) across the triangle surface.
5. Generates one fragment per covered pixel.

## Fragment stage

Runs once per fragment (potential pixel). Computes the final color.

```rust
#[quanta::fragment]
fn shade(uv: Vec2, normal: Vec3, light_dir: &Vec3) -> Vec4 {
    let diffuse = dot(normal, *light_dir).max(0.0);
    Vec4::new(diffuse, diffuse, diffuse, 1.0)
}
```

Inputs: interpolated values from the vertex stage + uniforms/textures.
Output: RGBA color for this pixel.

## Blending

Configurable at pipeline creation. Combines the fragment output with
whatever is already in the framebuffer.

Common modes:
- Opaque: replace (alpha = 1.0).
- Alpha blend: `final = src * src_alpha + dst * (1 - src_alpha)`.
- Additive: `final = src + dst`.

## Compute stage

Not part of the render pipeline. General-purpose GPU work.

```rust
#[quanta::kernel]
fn blur(input: &[f32], output: &mut [f32], width: u32) {
    let i = quark_id();
    output[i] = (input[i-1] + input[i] + input[i+1]) / 3.0;
}
```

No fixed inputs/outputs. No rasterization. You control everything.
Use compute for: physics, particle systems, image processing, ML inference,
sorting, reduction — anything that isn't rasterizing triangles.

## Advanced stages

### Tessellation (`#[quanta::tess_control]` + `#[quanta::tess_eval]`)

Subdivides patches into finer geometry on the GPU.

```
Input patches -> [Tess Control] -> Tessellator (HW) -> [Tess Eval] -> Rasterizer
```

### Mesh shaders (`#[quanta::task]` + `#[quanta::mesh]`)

Replaces vertex + input assembly. Generates geometry directly on the GPU.

```
[Task shader] -> [Mesh shader] -> Rasterizer -> [Fragment] -> Framebuffer
```

### Ray tracing (`#[quanta::ray_gen]` + `#[quanta::closest_hit]` + `#[quanta::miss]`)

Fires rays into an acceleration structure instead of rasterizing triangles.

```
[Ray Gen] -> trace_ray() -> BVH traversal (HW) -> [Closest Hit] or [Miss]
```

## Putting it together

```rust
// Create shaders
let vtx = transform();    // returns &ShaderBinary
let frag = shade();       // returns &ShaderBinary

// Create pipeline (vertex + fragment bound together)
let pipeline = gpu.pipeline(&PipelineDesc {
    vertex: vtx,
    fragment: frag,
    ..Default::default()
})?;

// Render
let pass = gpu.render_begin(&target)?;
// draw commands...
gpu.render_end(pass)?;
```

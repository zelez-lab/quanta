# The Render Pipeline: What Vertex and Fragment Shaders Do

## What is rendering?

Rendering turns 3D geometry into a 2D image. You give the GPU a list of
triangles in 3D space, and it produces a grid of colored pixels. This is what
games, CAD tools, and map applications do every frame.

## The pipeline

The GPU processes geometry through a fixed sequence of stages. Some stages are
programmable (you write the code), others are hardwired into the chip.

```
Your 3D data
  triangles = [(v0, v1, v2), (v3, v4, v5), ...]
  each vertex has: position, color, texture coordinate, normal
         |
         v
+---------------------+
| 1. VERTEX STAGE     |  Programmable: #[quanta::vertex]
|                     |  Runs once per vertex
|  Transforms each    |  Input:  vertex attributes (position, normal, ...)
|  vertex from 3D     |  Output: clip-space position (Vec4)
|  world space to     |
|  screen space       |  Think: "where does this point appear on screen?"
+---------------------+
         |
         v
+---------------------+
| 2. RASTERIZER       |  Fixed-function (not programmable)
|                     |
|  For each triangle: |  - Clips to the visible area
|  figure out which   |  - Projects 3D to 2D screen coordinates
|  pixels it covers   |  - Interpolates vertex outputs across the surface
|                     |  - Generates one "fragment" per covered pixel
+---------------------+
         |
         v
+---------------------+
| 3. FRAGMENT STAGE   |  Programmable: #[quanta::fragment]
|                     |  Runs once per fragment (covered pixel)
|  Computes the final |  Input:  interpolated values from vertex stage
|  color for each     |  Output: RGBA color
|  pixel              |
|                     |  Think: "what color is this pixel?"
+---------------------+
         |
         v
+---------------------+
| 4. BLENDING         |  Configurable (not programmable)
|                     |  Combines fragment color with existing pixel
|                     |  Example: alpha blending for transparency
+---------------------+
         |
         v
    Framebuffer (the final image on screen)
```

## Why two programmable stages?

The vertex stage answers: "where does each corner of my triangle end up on
screen?" It runs once per vertex -- a cube has 8 vertices, so it runs 8 times
regardless of how many pixels the cube covers.

The fragment stage answers: "what color is each pixel inside this triangle?"
It runs once per covered pixel -- a triangle covering 10,000 pixels runs the
fragment shader 10,000 times.

Splitting the work means the vertex stage runs on a small number of points,
while the expensive per-pixel work runs only where needed.

## How varyings work

The vertex stage outputs values (position, normal, texture coordinate). The
rasterizer interpolates these across the triangle surface. The fragment stage
receives the interpolated values.

```
Vertex A: color = red       Vertex B: color = blue
     \                         /
      \    fragment at 50%    /
       \   color = purple    /
        \       |           /
         \      |          /
          \     |         /
    Vertex C: color = green
```

The GPU does this interpolation automatically. You just declare what the vertex
stage outputs, and the fragment stage receives smoothly blended values.

## How textures work

A texture is a 2D image stored on the GPU. The fragment stage samples it using
texture coordinates (UVs) -- two numbers between 0.0 and 1.0 that map a pixel
to a point on the image.

```rust
#[quanta::fragment]
fn textured(uv: Vec2, texture: &Texture) -> Vec4 {
    texture.sample(uv)    // look up the color at this UV coordinate
}
```

The hardware handles filtering (bilinear, trilinear) so the image looks smooth
even when stretched or compressed.

See [Guide: Textures](../guide/06-textures.md) for details on texture formats
and sampling.

## Quanta render code

```rust
// 1. Define vertex and fragment shaders
#[quanta::vertex]
fn transform(pos: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn shade(normal: Vec3, light_dir: &Vec3) -> Vec4 {
    let brightness = dot(normal, *light_dir).max(0.0);
    Vec4::new(brightness, brightness, brightness, 1.0)
}

// 2. Create a pipeline (vertex + fragment bound together)
let pipeline = gpu.pipeline(&PipelineDesc {
    vertex: transform(),
    fragment: shade(),
    ..Default::default()
})?;

// 3. Render
let pass = gpu.render_begin(&target)?;
// draw commands...
gpu.render_end(pass)?;
```

See [Guide: Vertex and Fragment Shaders](../guide/08-vertex-fragment.md) for a
full walkthrough of writing render pipelines.

## Compute vs render: when to use which

The render pipeline is specialized for turning triangles into pixels. For
everything else, use compute kernels (`#[quanta::kernel]`).

| Use render when...              | Use compute when...              |
|---------------------------------|----------------------------------|
| Drawing 3D geometry to screen   | Processing arrays of data        |
| You need rasterization          | Physics simulation               |
| Texture sampling with filtering | Image processing (blur, FFT)     |
| Real-time graphics              | Machine learning inference        |
| UI rendering                    | Sorting, reduction, histogram    |

Compute kernels are simpler -- no pipeline stages, no rasterization, no
vertex/fragment split. You control the inputs, outputs, and dispatch directly.

See [Guide: Compute Basics](../guide/01-compute-basics.md) for compute,
and [Guide: Rendering](../guide/07-rendering.md) for the render pipeline.

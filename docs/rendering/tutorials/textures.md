# Textures

Textures are GPU images -- 2D, 3D, cube maps, or arrays. They support filtered
sampling (bilinear, trilinear, anisotropic), which buffers do not.

## TextureDesc

Every texture is created from a descriptor. `TextureDesc` is
`#[non_exhaustive]` — construct it with `TextureDesc::new(width, height,
format)` (2D, single-sample, one mip, `SHADER_READ`) and adjust through the
`with_*` builder methods:

```rust
use quanta::*;

let desc = TextureDesc::new(512, 512, Format::RGBA8)
    .with_kind(TextureKind::D2)      // default; also D3 / Cube / Array2D
    .with_sample_count(1)
    .with_mip_levels(1)
    .with_usage(TextureUsage::SHADER_READ);

let tex = gpu.create_texture(&desc)?;
```

Dropping a `Texture` releases the underlying driver resource — no manual
destroy call, no leak if it just falls out of scope.

### Formats

| Format            | Components | Bits/pixel | Use case                  |
|-------------------|-----------|------------|---------------------------|
| `Format::RGBA8`   | 4         | 32         | Standard color            |
| `Format::BGRA8`   | 4         | 32         | Swapchain (Windows/Metal) |
| `Format::R8`      | 1         | 8          | Masks, grayscale          |
| `Format::R16Float`| 1         | 16         | HDR single channel        |
| `Format::R32Float`| 1         | 32         | Depth, compute results    |
| `Format::RG32Float`| 2        | 64         | 2D vectors, motion        |
| `Format::RGBA16Float`| 4      | 64         | HDR color                 |
| `Format::RGBA32Float`| 4      | 128        | Full precision compute    |
| `Format::Depth32Float`| 1     | 32         | Depth buffer              |

Compressed formats (BC1-7, ASTC, ETC2) are available for read-only texture data
that ships with an application. They reduce GPU memory and bandwidth.

### Texture kinds

| Kind                  | Description               |
|-----------------------|---------------------------|
| `TextureKind::D2`     | Standard 2D image         |
| `TextureKind::D3`     | 3D volume                 |
| `TextureKind::Cube`   | Cube map (6 faces)        |
| `TextureKind::Array2D`| Array of 2D textures      |
| `TextureKind::ArrayCube`| Array of cube maps      |

### Usage flags

| Flag                           | Meaning                          |
|--------------------------------|----------------------------------|
| `TextureUsage::SHADER_READ`   | Sampled in shaders               |
| `TextureUsage::SHADER_WRITE`  | Written from compute shaders     |
| `TextureUsage::RENDER_TARGET` | Used as a color attachment       |

Combine with `.union()`:

```rust
let usage = TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ);
```

## Creating textures

### Full control

```rust
let tex = gpu.create_texture(
    &TextureDesc::new(1024, 1024, Format::RGBA16Float)
        .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE)),
)?;
```

### Convenience (RGBA8)

```rust
let tex = gpu.texture(256, 256)?;
// Equivalent to: create_texture with RGBA8, SHADER_READ, 1 mip
```

### Render target

```rust
let target = gpu.render_target(1920, 1080, Format::RGBA8)?;
// Usage: RENDER_TARGET | SHADER_READ
```

`render_target`, `msaa_target`, and `resolve_texture` come from the
`RenderGpu` extension trait (`use quanta::*;` brings it into scope).

### MSAA target

```rust
let msaa = gpu.msaa_target(1920, 1080, Format::RGBA8, 4)?;
// 4x MSAA, usage: RENDER_TARGET
```

## Writing pixels

Upload pixel data as raw bytes (row-major, tightly packed):

```rust
let pixels: Vec<u8> = vec![255; 256 * 256 * 4]; // white RGBA8
tex.write(&pixels)?;
```

Write, read, and mipmap generation are methods on `Texture` itself —
resources own their operations.

The byte layout must match the format. For `RGBA8`, each pixel is 4 bytes:
`[R, G, B, A]`.

For incremental updates — a glyph atlas gaining a few glyphs per frame, a
minimap tile refresh — upload only the changed rectangle instead of the whole
texture:

```rust
// 3 new 8x8 glyphs land at (x, y) in a 1024x1024 R8 atlas
tex.write_region((x, y), (8, 8), &glyph_bytes)?;
```

`origin` and `size` are in texels; `data` holds exactly `size.0 * size.1`
texels, tightly packed row-major. Out-of-bounds regions and mis-sized data
are rejected with `InvalidParam`. Backends that can't do sub-region uploads
return `NotSupported` — check `gpu.supports_texture_write_region()` and fall
back to a whole-texture `write`.

## Reading pixels

```rust
let pixels = tex.read()?;
// pixels: Vec<u8>, length = width * height * format.bytes_per_pixel()
```

Reading back is a synchronous transfer. Ensure any render pass or compute dispatch
writing to the texture has completed (wait on the pulse) before reading.

## Mipmaps

Mipmaps are pre-filtered downscaled versions of a texture. They improve quality
and performance when textures are viewed at reduced size.

```rust
let desc = TextureDesc::new(1024, 1024, Format::RGBA8)
    .with_mip_levels(0); // 0 = auto-calculate (log2(max(w,h)) + 1)
let tex = gpu.create_texture(&desc)?;
tex.write(&base_pixels)?;
tex.generate_mipmaps()?;
```

## Texture views

Create a view into a subset of a texture's mip chain or array layers:

```rust
use quanta::TextureViewDesc;

let view = gpu.texture_view_create(&tex, &TextureViewDesc {
    format: None,         // inherit parent's format
    mip_range: 2..4,     // only mip levels 2 and 3
    layer_range: 0..1,
})?;
```

Views share the parent texture's memory. Useful for binding specific mip levels
or array slices to different shader slots.

## Sampling in fragment shaders

Fragment shaders sample textures using the `sample(slot, uv)` function. The
slot number corresponds to `pass.set_texture(slot, ...)`:

```rust
#[quanta::vertex]
fn uv_vertex(pos: Vec3, uv: Vec2) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn textured(uv: Vec2) -> Vec4 {
    sample(0, uv)  // returns Vec4 (RGBA)
}
```

Bind the texture and sampler in the render pass:

```rust
pass.set_texture(0, &albedo);
pass.set_sampler(0, SamplerDesc::default()); // linear filtering
pass.draw(6);
```

`sample()` returns `Vec4` regardless of texture format. For single-channel
textures (e.g., glyph atlas), use `.x` to extract the value.

## Sampling in kernels

Inside compute kernels, textures are accessed via texture parameters:

```rust
#[quanta::kernel]
fn blur(input: &Texture2D<f32>, output: &mut [f32], width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;

    let c = texture_sample_2d(input, x, y);
    let l = texture_sample_2d(input, x - 1, y);
    let r = texture_sample_2d(input, x + 1, y);
    let u = texture_sample_2d(input, x, y - 1);
    let d = texture_sample_2d(input, x, y + 1);

    output[i] = (c + l + r + u + d) / 5.0;
}
```

Texture sampling in compute kernels uses hardware texture units, giving you
free bilinear filtering and edge clamping.

## Samplers

Control how texture reads interpolate and handle edges:

`SamplerDesc` is `#[non_exhaustive]` — start from `SamplerDesc::default()`
(linear min/mag, nearest mip, clamp-to-edge) and adjust with the `with_*`
methods:

```rust
use quanta::{SamplerDesc, Filter, AddressMode};

let sampler = gpu.sampler(
    &SamplerDesc::default()
        .with_filters(Filter::Linear, Filter::Linear)
        .with_mip_filter(Filter::Linear)
        .with_address_modes(AddressMode::Repeat, AddressMode::Repeat)
        .with_max_anisotropy(8),
)?;
```

For depth/shadow comparison samplers, add `.with_compare(CompareOp::Less)`
(or any other `CompareOp` variant). Dropping a `Sampler` releases the
driver resource.

| Filter           | Behavior                                |
|------------------|-----------------------------------------|
| `Filter::Nearest`| Snap to nearest texel (pixelated)       |
| `Filter::Linear` | Interpolate between neighbors (smooth)  |

| AddressMode          | Behavior                           |
|----------------------|------------------------------------|
| `ClampToEdge`        | Repeat edge pixel beyond bounds    |
| `Repeat`             | Tile the texture                   |
| `MirrorRepeat`       | Tile with alternating mirror       |

## Sparse textures

A sparse texture is a texture whose virtual extent (e.g. 16384 × 16384) is
declared up front, but whose backing memory is allocated tile-by-tile on demand.
Use it when the working set is far smaller than the address space — virtual
texturing, terrain megatextures, very large 3D volumes.

```rust
if !gpu.supports_sparse_residency() {
    // fall back to a regular texture
}

let tex = gpu.sparse_texture(&TextureDesc::new(16384, 16384, Format::RGBA8))?;

// Allocate one 256x256 backing tile and map it at (mip=0, x=0, y=0).
let backing = gpu.field::<u8>(256 * 256 * 4)?;
tex.map_tile(0, 0, 0, backing.handle())?;

// Later, when the tile leaves the working set:
tex.unmap_tile(0, 0, 0)?;
```

`map_tile`/`unmap_tile` are blocking: the queue serializes them with following
work. The tile coordinate system is driver-defined — typically 256 × 256 texels
on Vulkan and Metal. Resident pages persist until you unmap them or drop the
`SparseTexture`.

### Capability matrix

| Backend | Status                                               |
|---------|------------------------------------------------------|
| Vulkan  | `vkQueueBindSparse` + `VK_EXT_sparse_binding`        |
| Metal   | `MTLHeap` (Apple family 7+: M1, M2, ...)            |
| WebGPU  | `NotSupported` (sparse residency is not in the spec) |
| CPU     | Software (`HashMap<(mip, x, y), backing>`)            |

See [Expert: Sparse textures](../../expert/sparse-textures.md) for the lowering
details and per-driver caveats.

## MSAA resolve

After rendering to an MSAA target, resolve to a single-sample texture:

```rust
let msaa = gpu.msaa_target(1920, 1080, Format::RGBA8, 4)?;
let resolved = gpu.render_target(1920, 1080, Format::RGBA8)?;

// ... render pass writes to msaa ...

gpu.resolve_texture(&msaa, &resolved)?;
```

## Zero-copy export

A rendered texture can be handed to an external compositor without a
copy via `tex.native_handle()` — see
[Presenting to the screen](presentation.md#compositor-owns-present).

## Next

- [Presenting to the screen](presentation.md) -- surfaces and native-handle interop
- [Rendering](rendering.md) -- the graphics pipeline
- [Vertex and fragment shaders](vertex-fragment.md)

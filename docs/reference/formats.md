# Format Reference

All `Format` variants, their sizes, and capabilities.

## Uncompressed Formats

| Format | BPP | Channels | Renderable | Filterable | Storage | Blendable |
|--------|-----|----------|------------|------------|---------|-----------|
| `R8` | 1 | R (8-bit unorm) | Yes | Yes | Yes | Yes |
| `RGBA8` | 4 | RGBA (8-bit unorm each) | Yes | Yes | Yes | Yes |
| `BGRA8` | 4 | BGRA (8-bit unorm each) | Yes | Yes | No | Yes |
| `R16Float` | 2 | R (16-bit float) | Yes | Yes | Yes | Yes |
| `R32Float` | 4 | R (32-bit float) | Yes | Vendor-dependent | Yes | Vendor-dependent |
| `RG32Float` | 8 | RG (32-bit float each) | Yes | Vendor-dependent | Yes | Vendor-dependent |
| `RGBA16Float` | 8 | RGBA (16-bit float each) | Yes | Yes | Yes | Yes |
| `RGBA32Float` | 16 | RGBA (32-bit float each) | Yes | Vendor-dependent | Yes | Vendor-dependent |
| `Depth32Float` | 4 | Depth (32-bit float) | Yes (depth) | Yes | No | No |

## Compressed Formats

Block-compressed textures reduce memory and bandwidth by encoding fixed-size
pixel blocks (typically 4x4) into a constant number of bytes.

| Format | Block | Bytes/Block | BPP | Channels | Quality |
|--------|-------|-------------|-----|----------|---------|
| `Bc1Rgba` | 4x4 | 8 | 0.5 | RGBA (1-bit alpha) | Low |
| `Bc3Rgba` | 4x4 | 16 | 1.0 | RGBA (full alpha) | Medium |
| `Bc5Rg` | 4x4 | 16 | 1.0 | RG (normal maps) | High (2-channel) |
| `Bc7Rgba` | 4x4 | 16 | 1.0 | RGBA | High |
| `Astc4x4` | 4x4 | 16 | 1.0 | RGBA | High |
| `Astc6x6` | 6x6 | 16 | 0.44 | RGBA | Medium |
| `Astc8x8` | 8x8 | 16 | 0.25 | RGBA | Low |
| `Etc2Rgb8` | 4x4 | 8 | 0.5 | RGB | Medium |
| `Etc2Rgba8` | 4x4 | 16 | 1.0 | RGBA | Medium |

### Compressed format capabilities

| Format | Renderable | Filterable | Storage | Blendable |
|--------|------------|------------|---------|-----------|
| `Bc1Rgba` | No | Yes | No | No |
| `Bc3Rgba` | No | Yes | No | No |
| `Bc5Rg` | No | Yes | No | No |
| `Bc7Rgba` | No | Yes | No | No |
| `Astc4x4` | No | Yes | No | No |
| `Astc6x6` | No | Yes | No | No |
| `Astc8x8` | No | Yes | No | No |
| `Etc2Rgb8` | No | Yes | No | No |
| `Etc2Rgba8` | No | Yes | No | No |

Compressed formats are read-only from shaders. They cannot be used as render
targets or storage textures.

## Vendor support

| Format family | Apple | AMD/NVIDIA | Intel | Mobile (ARM) |
|---------------|-------|------------|-------|---------------|
| BC (1,3,5,7) | macOS only | Yes | Yes | No |
| ASTC | Yes | No | No | Yes |
| ETC2 | Yes | No | No | Yes |

Use `gpu.format_caps(format)` to query at runtime:

```rust
let caps = gpu.format_caps(Format::Bc7Rgba);
if caps.filterable {
    // Safe to use as a sampled texture
}
```

## FormatCaps

Returned by `gpu.format_caps(format)`.

```rust
pub struct FormatCaps {
    pub filterable: bool,  // Linear/mip sampling
    pub renderable: bool,  // Color render target
    pub storage: bool,     // Read-write from shaders
    pub blendable: bool,   // Blending when used as render target
    pub msaa: bool,        // Multi-sample anti-aliasing
    pub depth: bool,       // Depth/stencil attachment
}
```

## Common usage patterns

| Use case | Recommended format |
|----------|--------------------|
| Color render target | `BGRA8` or `RGBA8` |
| HDR render target | `RGBA16Float` |
| G-buffer normals | `RGBA16Float` |
| G-buffer positions | `RGBA32Float` |
| Depth buffer | `Depth32Float` |
| Compute buffer | `R32Float` or `RGBA32Float` |
| Diffuse texture (desktop) | `Bc7Rgba` |
| Diffuse texture (mobile) | `Astc4x4` or `Etc2Rgba8` |
| Normal map (desktop) | `Bc5Rg` |
| Shadow map | `Depth32Float` |

## TextureDesc defaults

`TextureDesc` is `#[non_exhaustive]` — construct it with
`TextureDesc::new(width, height, format)` (or `Default::default()`)
and adjust settings through the `with_*` builder methods:

```rust
let desc = TextureDesc::new(1024, 1024, Format::RGBA8)
    .with_mip_levels(0) // 0 = auto-calculate
    .with_usage(TextureUsage::SHADER_READ);
```

Defaults (what `new` leaves untouched):

| Field | Default | Builder |
|-------|---------|---------|
| `depth` | `1` (2D) | `.with_depth(n)` |
| `kind` | `TextureKind::D2` | `.with_kind(k)` |
| `sample_count` | `1` (no MSAA) | `.with_sample_count(n)` |
| `mip_levels` | `1` (no mipmaps; `0` = auto) | `.with_mip_levels(n)` |
| `array_length` | `1` | `.with_array_length(n)` |
| `usage` | `TextureUsage::SHADER_READ` | `.with_usage(u)` |

## TextureUsage flags

| Flag | Value | Description |
|------|-------|-------------|
| `SHADER_READ` | `1<<0` | Readable from shaders (sampling) |
| `SHADER_WRITE` | `1<<1` | Writable from shaders (compute output) |
| `RENDER_TARGET` | `1<<2` | Usable as color attachment |

Combine with `.union()`:

```rust
TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)
```

## FieldUsage flags

| Flag | Value | Description |
|------|-------|-------------|
| `READ` | `1<<0` | GPU read access |
| `WRITE` | `1<<1` | GPU write access |
| `COMPUTE` | `1<<2` | Used in compute dispatches |
| `RENDER` | `1<<3` | Used as vertex/index data |
| `TRANSFER` | `1<<4` | CPU upload/download |
| `UNIFORM` | `1<<5` | Uniform buffer |

Convenience constructors:

| Method | Flags |
|--------|-------|
| `FieldUsage::default_compute()` | READ + WRITE + COMPUTE + TRANSFER |
| `FieldUsage::default_render()` | READ + RENDER + TRANSFER |
| `FieldUsage::default_uniform()` | READ + UNIFORM + TRANSFER |

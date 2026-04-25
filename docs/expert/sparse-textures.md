# Expert: Sparse Textures

Sparse (virtual) textures allow you to allocate textures that are much
larger than GPU memory by only committing physical pages for regions that
are actually accessed. Used in terrain rendering, megatextures, and
streaming open-world environments.

## Concept

A sparse texture has a large virtual address space (e.g., 16384x16384) but
only tiles that are actively sampled have physical memory backing. The GPU
reports which tiles were accessed, and your application commits/evicts
pages dynamically.

## Creating a sparse texture

```rust
use quanta::*;

let sparse_tex = gpu.create_texture(&TextureDesc {
    width: 16384,
    height: 16384,
    format: Format::RGBA8,
    usage: TextureUsage::SHADER_READ | TextureUsage::SPARSE,
    ..TextureDesc::default()
})?;
```

## Committing and evicting tiles

```rust
// Commit a tile (make it resident in GPU memory)
gpu.sparse_commit(&sparse_tex, tile_x, tile_y, mip_level)?;

// Upload data to the committed tile
gpu.sparse_write(&sparse_tex, tile_x, tile_y, mip_level, &pixel_data)?;

// Evict a tile (release its physical memory)
gpu.sparse_evict(&sparse_tex, tile_x, tile_y, mip_level)?;
```

## Feedback buffer

Query which tiles the GPU actually sampled during a render pass:

```rust
let feedback = gpu.sparse_feedback(&sparse_tex)?;
for tile in &feedback.accessed_tiles {
    if !tile.is_committed {
        // Load and commit this tile from disk/network
        gpu.sparse_commit(&sparse_tex, tile.x, tile.y, tile.mip)?;
    }
}
```

## Platform support

| Platform | API | Sparse support |
|----------|-----|----------------|
| NVIDIA | Vulkan | Full (sparse residency) |
| AMD | Vulkan | Full (sparse residency) |
| Apple | Metal | Sparse textures (macOS 11+) |
| WebGPU | WGSL | Not yet standardized |

## Use cases

- **Terrain rendering** -- 100K+ texture tiles, commit only visible ones
- **Megatextures** -- single giant texture for an entire level
- **Streaming** -- load texture data asynchronously as the camera moves
- **Virtual texturing** -- indirection table + physical tile cache

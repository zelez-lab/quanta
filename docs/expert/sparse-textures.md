# Expert: Sparse Textures

Sparse (virtual) textures allocate textures whose virtual extent is much
larger than committed GPU memory. Pages are reserved virtually but
physical memory is only committed for tiles you explicitly map. Used in
terrain rendering, megatextures, and streaming open-world content.

## Concept

A sparse texture has two distinct sizes:

- **Virtual extent** — the dimensions you query as `width()` / `height()`
  on the typed wrapper. The shader samples this address space.
- **Resident pages** — the physical memory actually backing the texture.
  Initially empty; you commit pages on demand by calling `map_tile`.

The GPU returns sampled-but-uncommitted reads as zero (Vulkan
`sparseResidencyImage2D`) or implementation-defined values (Metal). This
lets a 16384×16384 texture cost only the bytes for tiles that are
actually visible.

## Creating a sparse texture

```rust
use quanta::{Gpu, TextureDesc, TextureKind, Format};

let gpu: Gpu = quanta::init()?;
let sparse_tex = gpu.sparse_texture(&TextureDesc {
    width: 16_384,
    height: 16_384,
    format: Format::RGBA8,
    kind: TextureKind::D2,
    mip_levels: 1,
    ..TextureDesc::default()
})?;
```

The returned value is a typed `SparseTexture`. It owns the backend handle
and frees it on `Drop`.

> **Limits today.** Sparse residency native ships **2D color textures
> with a single mip level** on every backend that supports it.
> 3D / Cube / Array sparse textures and partial-residency mips return
> `NotSupported` at create-time with an explicit message — not silently
> wrong. The full kind / mip-tail handling is post-v0.1 work.

## Mapping and unmapping tiles

```rust
// Allocate scratch for the binding-record contract — typed wrapper
// asks for a backing handle as part of T7602; the underlying driver
// supplies pages itself, but the wrapper records the pairing.
let backing = gpu.field::<u32>(64)?;

// Map tile (mip=0, x=0, y=0). Allocates a fresh page chunk and
// binds it via vkQueueBindSparse / MTLResourceStateCommandEncoder.
sparse_tex.map_tile(0, 0, 0, backing.handle())?;

// Re-binding the same tile is allowed: the old page is freed
// after a queue wait, the new one is bound in its place.
sparse_tex.map_tile(0, 0, 0, backing.handle())?;

// Unmap releases the page back to the driver. Idempotent —
// unmapping an already-unmapped tile is Ok per T7604.
sparse_tex.unmap_tile(0, 0, 0)?;
```

`Drop` on `SparseTexture` calls `sparse_texture_destroy`, which walks
any remaining tile bindings, waits for the queue, frees per-tile memory,
and tears down the image / heap. No leak even if you forgot to unmap.

## Feature detection

```rust
if !gpu.supports_sparse_residency() {
    // Fall back to a non-sparse texture path.
}
```

## Platform support

| Backend | Status                                                         |
|---------|----------------------------------------------------------------|
| Vulkan (real GPU) | ✅ Native — `vkQueueBindSparse` with per-tile `VkDeviceMemory`. Requires `VkPhysicalDeviceFeatures.sparseBinding` + a queue family with `VK_QUEUE_SPARSE_BINDING_BIT`. |
| Vulkan (lavapipe) | ✅ Validated end-to-end via the per-PR `diff-vulkan` lane (RPi 5 + lavapipe both pass). |
| Metal (Apple Silicon family 7+) | ✅ Native — `MTLHeap` of type `placement` + `MTLResourceStateCommandEncoder.updateTextureMapping`. M1 and later. |
| Metal (older Apple GPUs) | `NotSupported("sparse textures require Apple GPU family 7+ (A15/M2)")`. |
| WebGPU            | `NotSupported("sparse residency is not in the WebGPU spec")`. |
| CPU (software)    | Software lifecycle via `HashMap<(mip, x, y), backing>` — exercises the typed-wrapper proof contract but doesn't simulate residency. |

## How it lowers

### Vulkan

1. `sparse_texture` builds a `VkImage` with
   `VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT`.
   No `vkBindImageMemory` — sparse images can't bind memory at create
   time.
2. `map_tile` queries `vkGetImageMemoryRequirements` (alignment = per-
   tile size) and `vkGetImageSparseMemoryRequirements` (granularity in
   pixels). Allocates `mem_reqs.alignment` bytes of `DEVICE_LOCAL`
   memory and submits a `VkBindSparseInfo` whose
   `VkSparseImageMemoryBind` covers the (mip, x, y) tile.
3. `unmap_tile` submits the same shape with `memory = VK_NULL_HANDLE`
   to release the binding, then `vkFreeMemory`.

### Metal

1. `sparse_texture` queries `[device sparseTileSizeWithTextureType:
   pixelFormat:sampleCount:]` for tile granularity, then
   `[device heapTextureSizeAndAlignWithDescriptor:]` for the heap
   footprint, builds a `placement`-type `MTLHeap` of that size, and
   allocates the texture via `[heap newTextureWithDescriptor:offset:0]`
   with `hazardTrackingMode = .untracked` (matches placement-heap
   requirement).
2. `map_tile` opens a one-shot resource-state command encoder and
   issues `updateTextureMapping:mode:.map region:mipLevel:slice:`,
   committing the tile.
3. `unmap_tile` submits the same call with `mode:.unmap`.

## Use cases

- **Terrain rendering** — millions of texture tiles total, commit only
  the visible ones based on camera frustum.
- **Megatextures** — single giant texture for an entire level/world.
- **Texture streaming** — load tiles asynchronously as the camera
  approaches them; unmap when they're no longer in the working set.
- **Virtual texturing** — indirection table + physical tile cache, with
  the indirection table pointing into a sparse atlas.

## Caveats

- The `_backing` handle the wrapper passes through is unused on Vulkan
  and Metal — both backends source physical pages from the driver
  (`vkQueueBindSparse` allocates fresh `VkDeviceMemory`; `MTLHeap`
  supplies pages). It's accepted for parity with the typed-wrapper
  refinement contract.
- `map_tile` blocks until the bind submission completes — there's a
  `vkQueueWaitIdle` (Vulkan) and a `waitUntilCompleted` (Metal) inside.
  For high-frequency tile streaming, batch your binds and call
  the lower-level driver paths instead.
- WebGPU has no sparse residency in the spec; there's no realistic
  shim that gives you on-demand page commit in a browser.

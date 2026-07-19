# Effects Pipeline

A layer effects pipeline the way a UI toolkit drives it: take an RGBA8 layer
texture, run a compute kernel over it, and get the filtered layer back. Two
kinds of effect show up, and they want opposite memory shapes:

- A **point filter** — tint, brightness, invert, per-channel gamma — touches
  one texel at a time. It runs **in place**, mutating the layer texture through
  `&mut Texture2D<u32>`.
- A **neighborhood filter** — blur, sharpen, any kernel that reads its
  neighbors — must **not** run in place, because a neighbor it still needs may
  already have been overwritten. It reads one texture and writes another, and a
  two-texture **ping-pong** carries multi-pass filters.

Everything below is built on the packed-RGBA8 storage contract from
[`#[quanta::kernel]`](../../reference/macros.md): a texel crosses the kernel
boundary as one `0xAABBGGRR` u32 (little-endian byte order R, G, B, A), and you
pack/unpack the channels with bit math.

## The layer format

Effect layers are **RGBA8**. There is no SPIR-V storage format for BGRA8, so a
layer texture bound to a `&mut Texture2D<u32>` slot must be created as
`Format::RGBA8` with both `SHADER_READ` and `STORAGE` usage:

```rust
use quanta::{Format, TextureDesc, TextureUsage};

let layer = gpu
    .create_texture(
        &TextureDesc::new(width, height, Format::RGBA8)
            .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
    )
    .unwrap();
```

Uploaded and read-back bytes are `[R, G, B, A]` per texel. Two host helpers
mirror the kernel's byte order — pack for the seed, unpack for the read-back:

```rust
/// Pack four channel bytes into the kernel's `0xAABBGGRR` u32.
fn rgba8_pack(r: u8, g: u8, b: u8, a: u8) -> u32 {
    u32::from_le_bytes([r, g, b, a])
}

/// Read an RGBA8 texture's raw bytes back as [R, G, B, A] tuples per texel.
fn rgba8_unpack(bytes: &[u8]) -> Vec<(u8, u8, u8, u8)> {
    bytes.chunks_exact(4).map(|c| (c[0], c[1], c[2], c[3])).collect()
}
```

## Capability and tier gates

Compute over textures is not universal. Query it before dispatching:

```rust
if !gpu.supports_compute_textures() {
    // WebGPU returns false today; fall back to a CPU path or skip the effect.
    return;
}
```

RGBA8 **read-write** storage (`&mut Texture2D<u32>`) carries one more
requirement that `supports_compute_textures()` does **not** cover: on Metal it
needs `MTLReadWriteTextureTier2`. A device below tier 2 does not fail the query
— it surfaces `QuantaErrorKind::NotSupported` at **dispatch**. The
**read-only** form (`&Texture2D<u32>`) has no tier gate: RGBA8 `access::read`
works on every Metal device, which is exactly why the read half of a ping-pong
should be spelled `&Texture2D`. (Both are mandatory formats on Vulkan, so
there is no equivalent gate there.) Handle the tier miss where you dispatch,
the same way you would a sampled-read that a backend doesn't wire:

```rust
fn dispatch_or_skip(gpu: &Gpu, wave: &mut Wave, n: u32) -> Option<Pulse> {
    match gpu.dispatch(wave, n) {
        Ok(p) => Some(p),
        Err(e) if matches!(e.kind, QuantaErrorKind::NotSupported(_)) => {
            eprintln!("RGBA8 storage textures not supported here: {e}");
            None
        }
        Err(e) => panic!("dispatch failed: {e}"),
    }
}
```

The format contract is enforced per texel-slot kind: a `Texture2D<u32>` slot
(`&` or `&mut`) must be bound to an RGBA8 texture and a `Texture2D<f32>` slot
to an `R32Float` texture. A mismatch in either direction returns
`QuantaErrorKind::InvalidParam`.

## Point filter: in place through `&mut Texture2D<u32>`

The kernel takes the layer as a read-write storage image, loads the packed
texel, splits it into channels with the documented bit pattern, edits the
channels, repacks, and writes back to the same coordinate. Here we double the
red channel (saturating at 255) and leave G, B, A untouched:

```rust
#[quanta::kernel]
fn boost_red(layer: &mut Texture2D<u32>, width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;

    // Read the packed texel and split it into channels.
    let v = texture_load_2d(layer, x, y);
    let r = v & 0xFF;
    let g = (v >> 8) & 0xFF;
    let b = (v >> 16) & 0xFF;
    let a = (v >> 24) & 0xFF;

    // Edit one channel; saturate so it stays a byte.
    let mut r2 = r * 2;
    if r2 > 255 {
        r2 = 255;
    }

    // Repack in the same 0xAABBGGRR order and store in place.
    let out = r2 | (g << 8) | (b << 16) | (a << 24);
    texture_write_2d(layer, x, y, out);
}
```

Because every quark reads and writes only its own texel, running in place is
safe — no quark depends on another's result. The host binds the single layer at
slot 0, passes the width as a push constant, and dispatches one quark per texel:

```rust
let mut wave = boost_red(&gpu).unwrap();
wave.bind_texture(0, &layer);
wave.set_value(1, width);

let Some(mut pulse) = dispatch_or_skip(&gpu, &mut wave, width * height) else {
    return; // tier-2 gate: skip the effect on this device
};
pulse.wait().unwrap();

// Read the filtered layer back; bytes are [R, G, B, A] per texel.
let filtered = rgba8_unpack(&layer.read().unwrap());
```

To seed a layer from the host before filtering, pack each texel with
`rgba8_pack` and upload the bytes with `layer.write(&bytes)`.

## Neighborhood filter: why in place breaks

Now try a blur. Each output texel is the average of a window of input texels.
If the kernel wrote its result back into the layer it is reading, a
neighbouring quark that has not run yet would read the **already-blurred**
value instead of the original — the blur would bleed across the image and the
result would depend on quark scheduling order. Neighborhood filters need the
input to stay immutable for the whole pass. So they read one texture and write a
**different** one:

```rust
#[quanta::kernel]
fn blur_h(src: &Texture2D<f32>, dst: &mut Texture2D<f32>, width: u32, radius: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;

    // Read the neighborhood from `src` — never touched during this pass.
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for dx in 0..(2u32 * radius + 1u32) {
        let sx = x + dx - radius;
        if sx < width {
            sum += texture_load_2d(src, sx, y);
            count += 1u32;
        }
    }

    // Write the average to the *other* texture.
    texture_write_2d(dst, x, y, sum / (count as f32));
}
```

The ownership spells the roles: the read slot is a read-only texel image
(`&Texture2D<f32>`) and the write slot is read-write
(`&mut Texture2D<f32>`) — `R32Float`, single channel here. A separable
Gaussian blur is two of these passes — horizontal then vertical — which is
why a single buffer will not do: the second pass needs the first pass's full
output as clean input.

> Sampled reads in compute are a separate access mode with its own type:
> `texture_sample_2d` on a `&Sampled2D` slot fetches with a fixed nearest,
> clamp-to-edge, unnormalized-coordinate sampler, so the texel a GPU sample
> returns matches the CPU executor exactly — portable across Metal, native
> Vulkan, and the CPU executor. Reach for `&Sampled2D` when the source
> texture wasn't created with storage usage (e.g. an atlas you also render
> with); a texel `&Texture2D` slot needs `STORAGE` usage on the bound
> texture even though the kernel only reads it. WebGPU still reports
> `QuantaErrorKind::NotSupported` when the wave is built — handle it the same
> way as the tier-2 skip. (Filtered or normalized-UV sampling is a future
> sampler-API addition; today's compute sampler is fixed to the texel-fetch
> contract.)

## Ping-pong: two slots, swap between passes

A ping-pong drives multi-pass neighborhood filters with exactly two textures.
Each pass reads the **read slot** and writes the **write slot**; between passes
you swap which texture plays which role, so the previous pass's output becomes
the next pass's input:

```rust
// Two R32Float textures, both SHADER_READ | STORAGE.
let make = |gpu: &Gpu| {
    gpu.create_texture(
        &TextureDesc::new(width, height, Format::R32Float)
            .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE)),
    )
    .unwrap()
};
let tex_a = make(&gpu);
let tex_b = make(&gpu);
// ... upload the source layer into tex_a ...

let n = width * height;

// Pass 1: read A, write B.
let mut h = blur_h(&gpu).unwrap();
h.bind_texture(0, &tex_a); // read slot
h.bind_texture(1, &tex_b); // write slot
h.set_value(2, width);
h.set_value(3, radius);
gpu.dispatch(&h, n).unwrap().wait().unwrap();

// Pass 2: read B, write A — the roles swap.
let mut v = blur_v(&gpu).unwrap();
v.bind_texture(0, &tex_b); // read slot (pass 1's output)
v.bind_texture(1, &tex_a); // write slot
v.set_value(2, width);
v.set_value(3, radius);
gpu.dispatch(&v, n).unwrap().wait().unwrap();

// The final result lives in tex_a.
```

`blur_v` is `blur_h` with the window walking `y` instead of `x`. After an even
number of passes the result is back in the texture you started from; after an
odd number it is in the other one — track which slot is "current" and read that
one at the end.

## Choosing the shape

| Effect kind | Example | Slots | In place? |
|-------------|---------|-------|-----------|
| Point filter | tint, brightness, invert, gamma | one `&mut Texture2D<u32>` | yes |
| Neighborhood filter | blur, sharpen, edge detect | read `&Texture2D` + write `&mut Texture2D` | no — ping-pong |

The rule is mechanical: if a quark reads only its own texel, run in place and
save a texture; if it reads its neighbors, keep the input immutable and
ping-pong.

# Composite with render groups

A render GROUP is the structural form of offscreen compositing: draw
into a pooled layer texture, then bind that layer in a later pass of
the same frame — UI layer trees, post-processing chains, cached
backdrops. You never size, create, or free the intermediates; the
device pools them by shape and reuses them across frames.

```rust,ignore
// Draw the layer: the closure gets a full RenderBuilder targeting a
// pooled 256x256 texture, and must end the pass with .pulse().
let layer = gpu.render_group((256, 256), Format::RGBA8, |b| {
    b.clear(Color::TRANSPARENT)
        .pipeline(&badge_pipeline)
        .vertices(0, &badge_vertices)
        .draw(badge_count)
        .pulse()
})?;

// Use it: `layer` derefs to Texture and binds anywhere one does.
gpu.render_into(&target, |b| {
    b.clear(Color::BLACK)
        .pipeline(&compose_pipeline)
        .vertices(0, &quad)
        .texture(0, &layer)
        .sampler(0, quanta::SamplerDesc::default())
        .draw(6)
        .pulse()
})?;
```

The contract:

- **No host wait.** The group's pass is submitted when its closure
  pulses; any LATER pass on the same `Gpu` that samples the layer sees
  the finished contents (submission order plus the render-then-sample
  transition the drivers guarantee). Wait inside the closure only if
  the host itself reads the layer back.
- **Nesting is free.** A group drawn inside another group's closure is
  simply an earlier pass — compose layer trees to any depth.
- **`.msaa(n)` composes.** Call `.msaa(n).msaa_resolve()` on the
  group's builder and the multisampled pass ends by resolving into the
  pooled single-sample layer (create the pipeline with the matching
  `PipelineDesc::with_sample_count`; without `.msaa_resolve()` the
  samples stay in the MSAA intermediate, as on any pass).
- **Dropping the handle returns the texture to the pool.** In-flight
  passes keep the driver resource alive through the deferred-destroy
  machinery, so dropping is always safe; idle shapes a consumer stops
  using are trimmed automatically.

Manual control — your own render-target textures, explicit
`ColorTarget` ops, resolve into a texture you keep — remains available
through `render_target()` / `render()`; groups are the pooled,
structural fast path, not a replacement.

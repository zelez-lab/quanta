//! Builder API for render passes.
//!
//! Wraps `RenderPass` in a chainable builder that terminates with `.pulse()`.
//!
//! ```ignore
//! let mut pulse = gpu.render(&target)?
//!     .clear(Color::BLACK)
//!     .pipeline(&pipeline)
//!     .vertices(0, &verts)
//!     .draw(3)
//!     .pulse()?;
//! gpu.wait(&mut pulse)?;
//! ```

use alloc::sync::Arc;
use alloc::vec::Vec;

use quanta_core::render_pass::{ColorTarget, DepthTarget};
use quanta_core::texture::SamplerDesc;
use quanta_core::{
    Color, Field, GpuDevice, OcclusionQuery, Pipeline, Pulse, QuantaError, RenderPass, ShadingRate,
    Texture,
};
#[cfg(feature = "std")]
use quanta_core::{Format, LoadOp, MsaaPool, StoreOp};

/// Builder-managed MSAA state: the snapshot of the pass's final target
/// captured at [`render()`](crate::RenderGpu::render) time (everything
/// `.msaa(n)` needs to size/key the pooled intermediate and aim the
/// end-of-pass resolve, without re-borrowing the `Texture`), plus what
/// the chain requested. Assembled into the pass's color target by
/// `pulse()` — chain methods return `Self`, so their validation errors
/// are deferred there too, matching the encode-time validation style.
#[cfg(feature = "std")]
struct MsaaState {
    pool: Arc<MsaaPool>,
    target_handle: u64,
    target_width: u32,
    target_height: u32,
    target_format: Format,
    target_samples: u32,
    /// `StoreOp::resolve(&target)`, precomputed while the `&Texture`
    /// was in hand.
    resolve_store: StoreOp,
    /// Set by `.msaa(n)`.
    samples: Option<u32>,
    /// Set by `.msaa_resolve()`.
    resolve: bool,
    /// Set by `.load()`.
    explicit_load: bool,
    /// Mirrored from `.clear(color)`.
    clear_color: Option<Color>,
    /// Set by `.color_targets(..)` — conflicts with `.msaa(n)`.
    user_color_targets: bool,
    /// First builder-time validation error; surfaced by `pulse()`.
    deferred: Option<QuantaError>,
}

#[cfg(feature = "std")]
impl MsaaState {
    /// Record a builder-time validation error; the first one wins and
    /// `pulse()` returns it.
    fn defer(&mut self, err: QuantaError) {
        if self.deferred.is_none() {
            self.deferred = Some(err);
        }
    }
}

/// A chainable render pass builder.
///
/// Created by [`RenderGpu::render`](crate::RenderGpu::render). Every method consumes and returns `self`,
/// so the entire pass can be expressed as a single expression ending in
/// `.pulse()`.
///
/// ## Resource lifetimes
///
/// Binding methods record the resource's **handle**, not ownership:
/// every `Field`, `Texture`, and `Pipeline` bound to the pass must stay
/// alive until `pulse()` has submitted it (and, for CPU-side readback,
/// until the GPU finished — see `Pulse`). Dropping a bound resource
/// early makes `pulse()` fail with `NotFound` instead of silently
/// skipping the bind.
pub struct RenderBuilder {
    device: Arc<dyn GpuDevice>,
    pass: RenderPass,
    #[cfg(feature = "std")]
    msaa: MsaaState,
}

impl RenderBuilder {
    #[cfg(feature = "std")]
    pub(crate) fn new(
        device: Arc<dyn GpuDevice>,
        pass: RenderPass,
        pool: Arc<MsaaPool>,
        target: &Texture,
    ) -> Self {
        Self {
            device,
            pass,
            msaa: MsaaState {
                pool,
                target_handle: target.handle(),
                target_width: target.width(),
                target_height: target.height(),
                target_format: target.format(),
                target_samples: target.sample_count(),
                resolve_store: StoreOp::resolve(target),
                samples: None,
                resolve: false,
                explicit_load: false,
                clear_color: None,
                user_color_targets: false,
                deferred: None,
            },
        }
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn new(device: Arc<dyn GpuDevice>, pass: RenderPass) -> Self {
        Self { device, pass }
    }

    // === Pipeline ===

    /// Bind a render pipeline.
    pub fn pipeline(mut self, p: &Pipeline) -> Self {
        self.pass.set_pipeline(p);
        self
    }

    // === Vertex / Index data ===

    /// Bind a vertex buffer at a slot.
    pub fn vertices<T: Copy>(mut self, slot: u32, field: &Field<T>) -> Self {
        self.pass.bind_vertices(slot, field);
        self
    }

    /// Bind a vertex buffer at a slot with a byte offset.
    pub fn vertices_offset<T: Copy>(mut self, slot: u32, field: &Field<T>, offset: u64) -> Self {
        self.pass.bind_vertices_offset(slot, field, offset);
        self
    }

    /// Bind an index buffer (u32 indices).
    pub fn indices(mut self, field: &Field<u32>) -> Self {
        self.pass.bind_indices(field);
        self
    }

    // === Shader resources ===

    /// Bind a storage buffer at a shader slot.
    pub fn field<T: Copy>(mut self, slot: u32, field: &Field<T>) -> Self {
        self.pass.set_field(slot, field);
        self
    }

    /// Bind a uniform buffer at a shader slot.
    pub fn uniform<T: Copy>(mut self, slot: u32, field: &Field<T>) -> Self {
        self.pass.set_uniform(slot, field);
        self
    }

    /// Bind a texture at a shader texture slot.
    pub fn texture(mut self, slot: u32, tex: &Texture) -> Self {
        self.pass.set_texture(slot, tex);
        self
    }

    /// Set sampler state for a texture slot.
    pub fn sampler(mut self, slot: u32, desc: SamplerDesc) -> Self {
        self.pass.set_sampler(slot, desc);
        self
    }

    /// Set push constant / uniform data at a slot.
    pub fn value<V: Copy>(mut self, slot: u32, val: &V) -> Self {
        self.pass.set_value(slot, val);
        self
    }

    // === Draw commands ===

    /// Draw vertices (non-indexed, non-instanced).
    pub fn draw(mut self, vertex_count: u32) -> Self {
        self.pass.draw(vertex_count);
        self
    }

    /// Draw instanced geometry.
    pub fn draw_instanced(mut self, vertex_count: u32, instance_count: u32) -> Self {
        self.pass.draw_instanced(vertex_count, instance_count);
        self
    }

    /// Draw indexed geometry.
    pub fn draw_indexed(mut self, index_count: u32) -> Self {
        self.pass.draw_indexed(index_count);
        self
    }

    /// Draw indexed + instanced.
    pub fn draw_indexed_instanced(mut self, index_count: u32, instance_count: u32) -> Self {
        self.pass
            .draw_indexed_instanced(index_count, instance_count);
        self
    }

    /// Draw with arguments from a GPU buffer (GPU-driven rendering).
    pub fn draw_indirect<T: Copy>(mut self, buffer: &Field<T>, offset: u64) -> Self {
        self.pass.draw_indirect(buffer, offset);
        self
    }

    /// Draw indexed with arguments from a GPU buffer.
    pub fn draw_indexed_indirect<T: Copy>(
        mut self,
        buffer: &Field<T>,
        offset: u64,
        indices: &Field<u32>,
    ) -> Self {
        self.pass.draw_indexed_indirect(buffer, offset, indices);
        self
    }

    // === Backend-managed MSAA ===

    /// Render this pass at `samples`× MSAA into a **pooled
    /// intermediate**, keeping the pass's target as the single-sample
    /// resolve destination.
    ///
    /// The builder takes over the whole MSAA lifecycle retained-mode
    /// consumers used to hand-manage: it redirects the pass to an
    /// n-sample intermediate matching the target's `(width, height,
    /// format)`, pooled device-side and keyed by the target's handle —
    /// created on first use, reused by every later `.msaa(n)` pass over
    /// the same target. Load/clear ops apply to the **intermediate**:
    /// `.clear(color)` wipes it; the default (no clear, or an explicit
    /// [`load()`](Self::load)) preserves the samples the previous pass
    /// stored. Each pass ends with a plain `Store` — samples survive
    /// into the next pass — until one ends with
    /// [`msaa_resolve()`](Self::msaa_resolve), which resolves the
    /// intermediate into the target:
    ///
    /// ```ignore
    /// let target = gpu.render_target(w, h, Format::RGBA8)?; // 1x, sampleable
    /// gpu.render(&target)?
    ///     .msaa(4)
    ///     .clear(Color::BLACK)                 // clears the intermediate
    ///     .pipeline(&p4x)/* …draws… */
    ///     .pulse()?;                           // samples STORED, no resolve
    /// gpu.render(&target)?
    ///     .msaa(4)                             // SAME pooled intermediate
    ///     .load()                              // samples preserved
    ///     /* …draws… */
    ///     .msaa_resolve()                      // resolve → target at pass end
    ///     .pulse()?;
    /// // `target` now holds the resolved image and can be sampled.
    /// ```
    ///
    /// Rules, enforced at `pulse()` (builder methods defer their
    /// errors there):
    /// - `samples` must be a power of two in `2..=32`, and the bound
    ///   pipelines must be built `with_sample_count(samples)` — the
    ///   intermediate carries the count, so the standard pass-shape
    ///   validation catches a mismatch.
    /// - the pass's target must be single-sample (it is the resolve
    ///   destination, not the MSAA surface) — `InvalidParam` otherwise.
    /// - `.msaa(n)` and explicit [`color_targets()`](Self::color_targets)
    ///   conflict: the builder owns the pass's color attachment.
    /// - changing `n` between passes over the same target evicts and
    ///   recreates the pooled intermediate (don't do it while a pass
    ///   using the old one is in flight). See `quanta-core`'s
    ///   `msaa_pool` docs for the pool's full lifetime story — in
    ///   short: intermediates live until the device drops; dropping
    ///   the target does not evict its entry.
    /// - backends whose render path cannot subpass-resolve (WebGPU
    ///   today) fail the pass with `NotSupported`.
    ///
    /// The manual path (`msaa_target()` + explicit `ColorTarget` +
    /// `resolve_texture`) remains for callers that want to own the
    /// intermediate.
    #[cfg(feature = "std")]
    pub fn msaa(mut self, samples: u32) -> Self {
        if self.msaa.samples.is_some() {
            self.msaa.defer(QuantaError::invalid_param(
                "msaa() called twice on one pass — a pass has exactly one sample count",
            ));
            return self;
        }
        if samples < 2 || !samples.is_power_of_two() || samples > 32 {
            self.msaa.defer(QuantaError::invalid_param(alloc::format!(
                "msaa({samples}): sample count must be a power of two in 2..=32 — omit \
                 .msaa() entirely for a single-sample pass"
            )));
            return self;
        }
        if self.msaa.target_samples > 1 {
            self.msaa.defer(QuantaError::invalid_param(alloc::format!(
                "msaa({samples}) on a target that is itself multisampled ({} samples): \
                 .msaa() manages the MSAA intermediate — pass the single-sample resolve \
                 destination (a render_target) instead",
                self.msaa.target_samples
            )));
            return self;
        }
        self.msaa.samples = Some(samples);
        self
    }

    /// End this [`msaa`](Self::msaa) pass with a subpass resolve of the
    /// pooled intermediate into the pass's target.
    ///
    /// Without it the pass ends with a plain `Store` and the samples
    /// stay in the intermediate for the next `.msaa(n)` pass to
    /// [`load()`](Self::load). Calling it on a pass without `.msaa(n)`
    /// fails `pulse()` with `InvalidParam`.
    #[cfg(feature = "std")]
    pub fn msaa_resolve(mut self) -> Self {
        self.msaa.resolve = true;
        self
    }

    /// Explicitly mark this [`msaa`](Self::msaa) pass as **loading**
    /// the pooled intermediate — preserving the samples the previous
    /// pass stored.
    ///
    /// This is already the default for an `.msaa(n)` pass that records
    /// no `.clear()`; the method exists to document the intent in the
    /// chain. Combining it with `.clear()` — or calling it on a pass
    /// without `.msaa(n)` — fails `pulse()` with `InvalidParam` (for a
    /// single-sample pass, preserve contents with
    /// `ColorTarget::with_load_op(LoadOp::Load)` via
    /// [`color_targets()`](Self::color_targets)).
    #[cfg(feature = "std")]
    pub fn load(mut self) -> Self {
        self.msaa.explicit_load = true;
        self
    }

    // === Render state ===

    /// Clear the color attachment.
    pub fn clear(mut self, color: Color) -> Self {
        self.pass.clear(color);
        // Mirror the color for the builder-managed MSAA path: on an
        // `.msaa(n)` pass the clear applies to the pooled intermediate,
        // whose ColorTarget is assembled at pulse() from this state.
        #[cfg(feature = "std")]
        {
            self.msaa.clear_color = Some(color);
        }
        self
    }

    /// Clear the depth attachment.
    pub fn clear_depth(mut self, depth: f32) -> Self {
        self.pass.clear_depth(depth);
        self
    }

    /// Clear the stencil attachment.
    pub fn clear_stencil(mut self, value: u32) -> Self {
        self.pass.clear_stencil(value);
        self
    }

    /// Set the stencil reference value for comparison.
    pub fn stencil_ref(mut self, value: u32) -> Self {
        self.pass.set_stencil_ref(value);
        self
    }

    /// Set scissor rectangle (pixel coordinates).
    pub fn scissor(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.pass.set_scissor(x, y, width, height);
        self
    }

    /// Set viewport (normalized device coordinates mapping).
    pub fn viewport(mut self, x: f32, y: f32, width: f32, height: f32) -> Self {
        self.pass.set_viewport(x, y, width, height);
        self
    }

    /// Set viewport with depth range.
    pub fn viewport_depth(
        mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Self {
        self.pass
            .set_viewport_depth(x, y, width, height, min_depth, max_depth);
        self
    }

    // === Debug ===

    /// Push a debug label for this section of the render pass.
    pub fn debug_push(mut self, label: &str) -> Self {
        self.pass.debug_push(label);
        self
    }

    /// Pop a debug label.
    pub fn debug_pop(mut self) -> Self {
        self.pass.debug_pop();
        self
    }

    // === Occlusion Queries ===

    /// Begin an occlusion query at the given index.
    pub fn begin_occlusion_query(mut self, query: &OcclusionQuery, index: u32) -> Self {
        self.pass.begin_occlusion_query(query, index);
        self
    }

    /// End an occlusion query at the given index.
    pub fn end_occlusion_query(mut self, query: &OcclusionQuery, index: u32) -> Self {
        self.pass.end_occlusion_query(query, index);
        self
    }

    // === Variable-Rate Shading ===

    /// Set a uniform shading rate for subsequent draw calls.
    pub fn shading_rate(mut self, rate: ShadingRate) -> Self {
        self.pass.set_shading_rate(rate);
        self
    }

    /// Set a per-pixel shading rate from a shading rate image.
    pub fn shading_rate_image(mut self, texture: &Texture) -> Self {
        self.pass.set_shading_rate_image(texture);
        self
    }

    // === Multiple Render Targets ===

    /// Set the color attachment targets for this render pass.
    ///
    /// Manual control over attachments and their load/store ops (MRT,
    /// explicit `StoreOp::resolve`). Mutually exclusive with
    /// [`msaa()`](Self::msaa), which owns the pass's color attachment.
    pub fn color_targets(mut self, targets: Vec<ColorTarget>) -> Self {
        self.pass.set_color_targets(targets);
        #[cfg(feature = "std")]
        {
            self.msaa.user_color_targets = true;
        }
        self
    }

    /// Set the depth/stencil attachment target for this render pass.
    pub fn depth_target(mut self, target: DepthTarget) -> Self {
        self.pass.set_depth_target(target);
        self
    }

    // === Terminal ===

    /// Submit the render pass for execution.
    ///
    /// Consumes the builder and returns a `Pulse` that signals when the
    /// GPU finishes rendering. On an [`msaa`](Self::msaa) pass this
    /// first assembles the pooled intermediate into the pass's color
    /// attachment (and surfaces any deferred builder-time validation
    /// error).
    #[cfg_attr(not(feature = "std"), allow(unused_mut))]
    pub fn pulse(mut self) -> Result<Pulse, QuantaError> {
        #[cfg(feature = "std")]
        self.assemble_msaa()?;
        self.device.render_end(self.pass)
    }

    /// The `pulse()`-time half of the builder-managed MSAA path:
    /// surface deferred chain errors, enforce the cross-method rules,
    /// and — when `.msaa(n)` was requested — point the pass at the
    /// pooled intermediate with the load/store ops the chain implies.
    #[cfg(feature = "std")]
    fn assemble_msaa(&mut self) -> Result<(), QuantaError> {
        let st = &mut self.msaa;
        if let Some(err) = st.deferred.take() {
            return Err(err);
        }
        let Some(samples) = st.samples else {
            // Not an MSAA pass — the msaa-only markers must not dangle.
            if st.resolve {
                return Err(QuantaError::invalid_param(
                    "msaa_resolve() without msaa(n): only a builder-managed MSAA pass can \
                     end in a subpass resolve — call .msaa(n) on this pass, or use the \
                     manual msaa_target()/resolve_texture() path",
                ));
            }
            if st.explicit_load {
                return Err(QuantaError::invalid_param(
                    "load() without msaa(n): load() preserves the MSAA intermediate — for \
                     a single-sample pass use ColorTarget::with_load_op(LoadOp::Load) via \
                     .color_targets()",
                ));
            }
            return Ok(());
        };
        if st.user_color_targets {
            return Err(QuantaError::invalid_param(
                "msaa(n) and color_targets() on the same pass: .msaa() owns the pass's \
                 color attachment (the pooled intermediate) — for manual control use \
                 color_targets() with msaa_target() and StoreOp::resolve",
            ));
        }
        if st.explicit_load && st.clear_color.is_some() {
            return Err(QuantaError::invalid_param(
                "both load() and clear() on an msaa pass: clear() wipes the MSAA \
                 intermediate, load() preserves the samples the previous pass stored — \
                 pick one",
            ));
        }
        // No clear ⇒ LOAD the intermediate (samples preserved across
        // passes; a fresh intermediate's first Load hits the tracked-
        // layout virgin guard and downgrades to don't-care — contents
        // undefined, so clear on the first pass).
        let load_op = match st.clear_color {
            Some(color) => LoadOp::Clear(color),
            None => LoadOp::Load,
        };
        let store_op = if st.resolve {
            st.resolve_store
        } else {
            StoreOp::Store
        };
        let intermediate = st
            .pool
            .intermediate_color_target(
                &self.device,
                st.target_handle,
                st.target_width,
                st.target_height,
                st.target_format,
                samples,
            )?
            .with_load_op(load_op)
            .with_store_op(store_op);
        self.pass.set_color_targets(alloc::vec![intermediate]);
        Ok(())
    }
}

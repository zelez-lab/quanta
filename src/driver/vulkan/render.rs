//! Render pipeline and render pass operations for Vulkan.

// Render-only submodules — gated with the `render` feature (step 085).
#[cfg(feature = "render")]
mod pipeline;
#[cfg(feature = "render")]
mod render_pass;
// `queries` also holds the SHARED timestamp-query impls used by compute,
// so it stays compiled; its render-only items are gated inside.
mod queries;

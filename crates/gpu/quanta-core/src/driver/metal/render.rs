//! Render pipeline and pass execution for Metal.

mod pipeline;
mod queries;
mod render_pass;

/// Metal has ONE buffer-argument index space per stage, shared by
/// vertex-attribute buffers and user buffers (`.field`/`.uniform`/
/// `.value`). Geometry buffers are remapped to a high base so user
/// slots 0-15 never collide with vertex layouts (the wgpu approach).
/// Metal allows buffer indices 0-30; layouts occupy 16..16+N.
pub(crate) const VERTEX_ATTRIBUTE_BUFFER_BASE: u64 = 16;

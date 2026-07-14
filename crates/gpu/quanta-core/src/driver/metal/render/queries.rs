//! Query and timestamp operations for Metal.
//!
//! Occlusion query begin/end are currently inline in `render_pass.rs`
//! as match arms of `RenderOp::BeginOcclusionQuery` / `EndOcclusionQuery`.
//! Standalone timestamp and pipeline-statistics queries will live here
//! when implemented.

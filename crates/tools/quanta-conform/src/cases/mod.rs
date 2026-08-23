//! The conformance corpus. Every case draws with the public API only,
//! offscreen, RGBA8, row 0 = top (the orientation contract), and
//! returns the readback as a [`Frame`].
//!
//! Template notes for new cases:
//! * One module per family; the family's fragments live in it, and the
//!   geometry / vertex stages / pass plumbing several families share
//!   live in [`common`].
//! * 64×64 target unless the case is about size.
//! * Choose [`Tolerance::EXACT`] when no primitive edge crosses the
//!   frame interior (fullscreen quads, clears); [`Tolerance::EDGED`]
//!   when one does. A case whose own resolve or filtering math differs
//!   legitimately between backends names its own terms instead — see
//!   `msaa4_resolve`.
//! * Shaders are `#[quanta::vertex]` / `#[quanta::fragment]` items at
//!   module scope. A fragment reading the shared [`common::QuadVary`]
//!   interface has to import its `__quanta_varyings_QuadVary`
//!   trampoline; the struct itself is erased with the shader fn.
//! * Deterministic: no time, no randomness, no dependence on what ran
//!   before. Every case builds its own target.
//! * A new case needs a golden — re-bless (`just conform-bless`) and
//!   commit the `.frame` alongside it.

use crate::{Case, Tolerance};

mod bands;
mod blend;
mod clear;
mod clip;
mod common;
mod depth;
mod indexed;
mod instanced;
mod msaa;
mod quad;
mod texture;
mod triangle;

/// The corpus, in matrix row order.
pub fn all() -> Vec<Case> {
    vec![
        Case {
            name: "clear_only",
            tolerance: Tolerance::EXACT,
            run: clear::run,
        },
        Case {
            name: "uv_gradient_quad",
            tolerance: Tolerance::EXACT,
            run: quad::run_uv_gradient,
        },
        Case {
            name: "triangle_flat",
            tolerance: Tolerance::EDGED,
            run: triangle::run,
        },
        Case {
            name: "orientation_bands",
            tolerance: Tolerance::EXACT,
            run: bands::run_orientation,
        },
        Case {
            name: "bands_nested_if",
            tolerance: Tolerance::EXACT,
            run: bands::run_nested_if,
        },
        Case {
            name: "indexed_quad",
            tolerance: Tolerance::EXACT,
            run: indexed::run,
        },
        Case {
            name: "instanced_triangles",
            tolerance: Tolerance::EDGED,
            run: instanced::run,
        },
        Case {
            name: "depth_two_triangles",
            tolerance: Tolerance::EDGED,
            run: depth::run,
        },
        Case {
            name: "alpha_blend_quad",
            tolerance: Tolerance::EXACT,
            run: blend::run,
        },
        Case {
            name: "viewport_scissor",
            tolerance: Tolerance::EXACT,
            run: clip::run,
        },
        Case {
            name: "texture_sample_quad",
            tolerance: Tolerance::EXACT,
            run: texture::run,
        },
        Case {
            name: "msaa4_resolve",
            tolerance: msaa::RESOLVE,
            run: msaa::run,
        },
    ]
}

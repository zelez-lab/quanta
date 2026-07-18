//! Render-face proc macros for Quanta shaders.
//!
//! The graphics-stage attribute macros (`vertex`, `fragment`, the
//! tessellation / mesh / ray-tracing stages) and the `Vertex` / `Varyings`
//! derives. Shared shader-compile plumbing (parameter parsing,
//! compiler-binary invocation, MSL/WGSL emission) lives in
//! `quanta-dsl-core`; this crate never depends on the compute stack (see
//! the render-purity note on its `Cargo.toml` deps).
//!
//! # The vertex↔fragment interface
//!
//! Varyings are declared ONCE, in a `#[derive(Varyings)]` struct — the
//! shared explicit interface between the two stages (the WGSL/HLSL model):
//!
//! ```ignore
//! #[derive(quanta::Varyings)]
//! struct Surface {
//!     #[position] clip: Vec4, // gl_Position — the vertex writes it
//!     uv: Vec2,               // Location 0 (field-declaration order)
//!     kind: u32,              // Location 1, flat-interpolated
//! }
//!
//! #[quanta::vertex]
//! fn vs(pos: Vec3, in_uv: Vec2) -> Surface {
//!     Surface { clip: Vec4::new(pos.x, pos.y, 0.0, 1.0), uv: in_uv, kind: 0u32 }
//! }
//!
//! #[quanta::fragment]
//! fn fs(s: Surface) -> Vec4 {
//!     sample(0, s.uv)
//! }
//! ```
//!
//! Because a proc macro sees only its own item, the interface crosses from
//! the struct to the shader macros through the derive-generated trampoline
//! (`__quanta_varyings_<Name>!`): `#[quanta::vertex]` / `#[quanta::fragment]`
//! expand to a trampoline invocation, which forwards the field metadata plus
//! the function to the hidden second-stage macros ([`__vertex_varyings`] /
//! [`__fragment_varyings`]) that run the real shader compile. Declare the
//! struct before the shaders (or import its `__quanta_varyings_<Name>`
//! re-export alongside it from another module).

extern crate proc_macro;

mod crate_path;
mod shader_macro;
mod varyings_derive;
mod varyings_macro;
mod vertex_derive;

use proc_macro::TokenStream;
use syn::{ItemFn, parse_macro_input};

/// Mark a function as a vertex shader.
///
/// Compiles the function at build time and embeds the backend binaries as a
/// `ShaderBinary` static. Value parameters are vertex attributes (pure
/// inputs — nothing is auto-forwarded); reference parameters (`&T`) become
/// uniform buffer bindings, `&[T]` storage-buffer slices.
///
/// Two return forms:
/// - `-> Vec4` — a position-only vertex: the body's tail expression is the
///   clip-space position, and the shader has NO varyings.
/// - `-> MyVaryings` (a `#[derive(Varyings)]` struct) — the shared-struct
///   interface: the body ends in the struct literal, whose `#[position]`
///   field becomes gl_Position and whose other fields are the varyings the
///   paired fragment reads.
///
/// ```ignore
/// #[quanta::vertex]
/// fn transform(pos: Vec3, in_uv: Vec2, mvp: &Mat4) -> Surface {
///     Surface {
///         clip: mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0),
///         uv: in_uv,
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn vertex(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr.clone());
    shader_macro::expand_vertex(func, attr.into(), &cp)
}

/// Mark a function as a fragment shader.
///
/// Compiles the function at build time and embeds the backend binaries as a
/// `ShaderBinary` static. Stage inputs come from the paired vertex's
/// `#[derive(Varyings)]` struct, taken as a single param and read by field
/// (`s.uv`); reading the `#[position]` field yields the interpolated window
/// position. Reference parameters become uniform / texture / slice bindings.
/// A fragment with no varyings simply omits the struct param.
///
/// ```ignore
/// #[quanta::fragment]
/// fn shade(s: Surface, atlas: &Texture2D) -> Vec4 {
///     sample(atlas, s.uv)
/// }
/// ```
#[proc_macro_attribute]
pub fn fragment(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr.clone());
    shader_macro::expand_fragment(func, attr.into(), &cp)
}

/// Hidden second stage of `#[quanta::vertex]` for the shared-struct varying
/// model — invoked through a `#[derive(Varyings)]` struct's trampoline,
/// never written by hand.
#[doc(hidden)]
#[proc_macro]
pub fn __vertex_varyings(input: TokenStream) -> TokenStream {
    varyings_macro::expand_vertex_varyings(input)
}

/// Hidden second stage of `#[quanta::fragment]` for the shared-struct
/// varying model — invoked through a `#[derive(Varyings)]` struct's
/// trampoline, never written by hand.
#[doc(hidden)]
#[proc_macro]
pub fn __fragment_varyings(input: TokenStream) -> TokenStream {
    varyings_macro::expand_fragment_varyings(input)
}

// === Tessellation shader macros (M4.1) ===

/// Mark a function as a tessellation control (hull) shader.
///
/// The function defines per-control-point logic and sets tessellation factors.
/// Source is captured at build time for MSL/WGSL emission.
///
/// ```ignore
/// #[quanta::tess_control]
/// fn hull(patch_id: u32) -> TessFactors {
///     TessFactors { edge: [4.0; 4], inside: [4.0; 2] }
/// }
/// ```
#[proc_macro_attribute]
pub fn tess_control(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr);
    shader_macro::expand_tess_control(func, &cp)
}

/// Mark a function as a tessellation evaluation (domain) shader.
///
/// Runs once per generated vertex after tessellation. Reads patch data
/// and barycentric coordinates to compute final vertex positions.
///
/// ```ignore
/// #[quanta::tess_eval]
/// fn domain(uv: Vec2, patch: &[Vec3; 4]) -> Vec4 {
///     // Bilinear interpolation of control points
///     let p = mix(mix(patch[0], patch[1], uv.x), mix(patch[2], patch[3], uv.x), uv.y);
///     Vec4::new(p.x, p.y, p.z, 1.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn tess_eval(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr);
    shader_macro::expand_tess_eval(func, &cp)
}

// === Mesh shader macros (M4.2) ===

/// Mark a function as a task (amplification) shader.
///
/// The task shader performs coarse-grained culling and determines how many
/// mesh shader threadgroups to launch. Optional — mesh shaders can be
/// dispatched directly without a task shader.
///
/// ```ignore
/// #[quanta::task]
/// fn cull(group_id: u32, instances: &[BoundingBox]) {
///     if is_visible(instances[group_id]) {
///         emit_mesh_threadgroups(1);
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn task(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr);
    shader_macro::expand_task(func, &cp)
}

/// Mark a function as a mesh shader.
///
/// Replaces the vertex + input assembly stages. The mesh shader generates
/// vertices and primitives directly, enabling GPU-driven geometry processing.
///
/// ```ignore
/// #[quanta::mesh]
/// fn generate(group_id: u32) {
///     // Emit vertices and triangle indices directly
///     set_vertex(0, Vec4::new(-1.0, -1.0, 0.0, 1.0));
///     set_vertex(1, Vec4::new( 1.0, -1.0, 0.0, 1.0));
///     set_vertex(2, Vec4::new( 0.0,  1.0, 0.0, 1.0));
///     set_primitive(0, [0, 1, 2]);
/// }
/// ```
#[proc_macro_attribute]
pub fn mesh(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr);
    shader_macro::expand_mesh(func, &cp)
}

// === Ray tracing shader macros (M4.3) ===

/// Mark a function as a ray generation shader.
///
/// The entry point for ray tracing — launched once per pixel (or per ray).
/// Uses `trace_ray()` to fire rays into the acceleration structure.
///
/// ```ignore
/// #[quanta::ray_gen]
/// fn camera_rays(pixel: UVec2, scene: &AccelerationStructure) {
///     let ray = compute_ray(pixel);
///     trace_ray(scene, ray, 0, 1000.0);
/// }
/// ```
#[proc_macro_attribute]
pub fn ray_gen(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr);
    shader_macro::expand_ray_gen(func, &cp)
}

/// Mark a function as a closest-hit shader.
///
/// Invoked when a ray intersects the nearest surface. Computes the shading
/// result (color, material response) and may fire secondary rays (reflections).
///
/// ```ignore
/// #[quanta::closest_hit]
/// fn shade(hit: HitInfo, ray: Ray) -> Vec4 {
///     let normal = hit.normal;
///     let color = sample_texture(hit.uv);
///     color * dot(normal, light_dir).max(0.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn closest_hit(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr);
    shader_macro::expand_closest_hit(func, &cp)
}

/// Mark a function as a miss shader.
///
/// Invoked when a ray does not intersect any geometry. Typically returns
/// a sky/environment color.
///
/// ```ignore
/// #[quanta::miss]
/// fn sky(ray: Ray) -> Vec4 {
///     let t = 0.5 * (ray.direction.y + 1.0);
///     Vec4::lerp(Vec4::splat(1.0), Vec4::new(0.5, 0.7, 1.0, 1.0), t)
/// }
/// ```
#[proc_macro_attribute]
pub fn miss(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let cp = crate_path::from_attr_args(attr);
    shader_macro::expand_miss(func, &cp)
}

// === Derive macros ===

/// Derive vertex layout metadata from a struct's fields.
///
/// The struct must have `#[repr(C)]` and use only GPU-compatible field types:
/// - Scalars: `f32`, `u32`, `i32`
/// - Vectors: `[f32; 2]`, `[f32; 3]`, `[f32; 4]`, and similar for `u32`/`i32`
///
/// ```ignore
/// #[repr(C)]
/// #[derive(Copy, Clone, quanta::Vertex)]
/// struct MyVertex {
///     pos: [f32; 3],   // location 0, Float3, offset 0
///     color: [f32; 4], // location 1, Float4, offset 12
/// }
///
/// // Generated:
/// // MyVertex::ATTRIBUTES — const array of VertexAttribute
/// // MyVertex::vertex_layout() -> VertexLayout
/// ```
/// The container attribute `#[quanta(crate = <path>)]` overrides the
/// crate root the generated `VertexAttribute` / `VertexLayout` paths
/// are written against. Default `::quanta`.
#[proc_macro_derive(Vertex, attributes(quanta))]
pub fn derive_vertex(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemStruct);
    let cp = crate_path::crate_from_container_attrs(&input.attrs);
    match vertex_derive::expand_vertex_derive(&input, &cp) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Derive the shared vertex↔fragment varying interface from a struct.
///
/// Exactly one field must carry the `#[position]` marker and be a `Vec4` —
/// it becomes gl_Position / `[[position]]` (and, read from a fragment, the
/// interpolated window position). Every other field is a varying, assigned
/// Location 0, 1, … in FIELD-DECLARATION order; `u32` fields are
/// flat-interpolated on both interface ends. Supported field types: `f32`,
/// `u32`, `Vec2`, `Vec3`, `Vec4`.
///
/// ```ignore
/// #[derive(quanta::Varyings)]
/// struct Surface {
///     #[position] clip: Vec4, // gl_Position
///     uv: Vec2,               // Location 0
///     kind: u32,              // Location 1, flat
/// }
/// ```
///
/// Generated: `Surface::POSITION_FIELD` / `Surface::VARYING_FIELDS`
/// introspection consts, plus the hidden `__quanta_varyings_Surface!`
/// trampoline the shader macros expand through (declare the struct before
/// the shaders that use it, or import the trampoline alongside it).
#[proc_macro_derive(Varyings, attributes(position))]
pub fn derive_varyings(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemStruct);
    match varyings_derive::expand_varyings_derive(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

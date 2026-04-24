//! Proc macros for Quanta GPU kernels.

extern crate proc_macro;

#[allow(dead_code)]
mod compiler;
mod device_macro;
mod gpu_type;
mod kernel_macro;
mod parse;
mod shader_macro;
mod validate;

use proc_macro::TokenStream;
use syn::{ItemFn, parse_macro_input};

/// Mark a function as a GPU kernel.
///
/// ```ignore
/// #[quanta::kernel]                                      // default: O3, workgroup [64,1,1]
/// #[quanta::kernel(opt = "O2")]                          // explicit O2
/// #[quanta::kernel(opt = "O0")]                          // no optimization (debug)
/// #[quanta::kernel(workgroup = [256])]                   // [256, 1, 1]
/// #[quanta::kernel(workgroup = [16, 16])]                // [16, 16, 1]
/// #[quanta::kernel(workgroup = [16, 16, 1])]             // explicit 3D
/// #[quanta::kernel(workgroup = [256], opt = "O2")]       // both
/// #[quanta::kernel(subgroup = 32)]                       // require subgroup size 32
/// #[quanta::kernel(jit)]                                 // JIT: compile at runtime
/// ```
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    kernel_macro::expand_kernel(attr, func)
}

/// Mark a function as a GPU device function (callable from kernels).
///
/// ```ignore
/// #[quanta::device]
/// fn activate(x: f32, threshold: f32) -> f32 {
///     if x > threshold { x } else { x * 0.99 }
/// }
/// ```
///
/// Device functions are inlined into kernels by LLVM.
/// They cannot be launched from CPU — only called from `#[quanta::kernel]` functions.
///
/// The proc macro captures the function source and emits a hidden constant
/// `__QUANTA_DEVICE_{NAME_UPPERCASE}` containing the source text. Kernel
/// compilation picks this up so MSL/WGSL emitters can prepend it as a regular
/// helper function.
#[proc_macro_attribute]
pub fn device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    device_macro::expand_device(func)
}

/// Mark a variable as shared (workgroup-local) memory inside a kernel.
///
/// ```ignore
/// #[quanta::kernel]
/// fn reduce(data: &[f32], result: &mut [f32]) {
///     #[quanta::shared] let local: [f32; 256];
///     let lid = local_id();
///     local[lid] = data[quark_id()];
///     barrier();
/// }
/// ```
///
/// When used inside a `#[quanta::kernel]` body, the kernel parser handles
/// this attribute directly — it emits `SharedDecl`, `SharedLoad`, and
/// `SharedStore` ops in the IR.
///
/// The proc macro itself is a no-op pass-through; the real work happens
/// in the kernel parser which inspects `let` attributes.
#[proc_macro_attribute]
pub fn shared(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// Mark a function as a vertex shader.
///
/// Compiles the function to MSL and WGSL at build time and embeds both as a
/// `ShaderBinary` static. Value parameters become vertex attributes;
/// reference parameters (`&T`) become uniform buffer bindings.
///
/// ```ignore
/// #[quanta::vertex]
/// fn transform(
///     pos: Vec3,
///     normal: Vec3,
///     mvp: &Mat4,
/// ) -> Vec4 {
///     mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn vertex(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_vertex(func)
}

/// Mark a function as a fragment shader.
///
/// Compiles the function to GPU binary at build time and embeds it as a
/// `ShaderBinary` static. Value parameters become fragment stage inputs
/// (interpolated varyings); reference parameters become uniform/texture bindings.
///
/// ```ignore
/// #[quanta::fragment]
/// fn shade(
///     uv: Vec2,
///     color: Vec4,
/// ) -> Vec4 {
///     color * Vec4::new(uv.x, uv.y, 0.0, 1.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn fragment(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_fragment(func)
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
pub fn tess_control(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_tess_control(func)
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
pub fn tess_eval(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_tess_eval(func)
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
pub fn task(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_task(func)
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
pub fn mesh(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_mesh(func)
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
pub fn ray_gen(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_ray_gen(func)
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
pub fn closest_hit(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_closest_hit(func)
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
pub fn miss(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    shader_macro::expand_miss(func)
}

/// Mark a struct as GPU-compatible.
///
/// Generates `#[repr(C)]`, `#[derive(Copy, Clone)]`, `GpuType` impl,
/// field metadata (`GPU_SIZE`, `GPU_FIELDS`), and MSL/WGSL struct declarations.
///
/// ```ignore
/// #[quanta::gpu_type]
/// struct Particle {
///     pos: [f32; 3],
///     vel: [f32; 3],
///     mass: f32,
/// }
/// ```
///
/// Generates:
/// - `Particle::GPU_SIZE` — byte size of the struct
/// - `Particle::GPU_FIELDS` — `&[(&str, &str, usize)]` of (name, type, byte_offset)
/// - `impl GpuType for Particle`
/// - `__QUANTA_GPU_TYPE_PARTICLE` — MSL struct declaration string
/// - `__QUANTA_GPU_TYPE_PARTICLE_WGSL` — WGSL struct declaration string
#[proc_macro_attribute]
pub fn gpu_type(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemStruct);
    match gpu_type::expand_gpu_type(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

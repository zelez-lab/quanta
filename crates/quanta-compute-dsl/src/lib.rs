//! Compute-face proc macros for Quanta GPU kernels.
//!
//! The `#[quanta::kernel]` / `#[quanta::device]` / `#[quanta::shared]`
//! attribute macros, `import_devices!`, `#[quanta::gpu_type]`, and the
//! `Fields` / `Uniforms` derives, plus the rustc → wasm32 → KernelOps
//! lowering pipeline behind them. Compiler-binary invocation and the
//! shared emitters live in `quanta-dsl-core`.

extern crate proc_macro;

mod auto_dispatch;
mod compile_via_wasm;
mod device_macro;
mod fields_derive;
mod gpu_type;
mod kernel_macro;
mod kernel_signature;
mod kernel_type_inference;
mod parse;
mod uniforms_derive;
mod validate;
mod wasm_compile;

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
///
/// Unknown attribute names produce a compile error pointing at
/// the correct form. The recognised names are: `opt`,
/// `workgroup`, `subgroup`, `jit`.
///
/// # Kernel-body gotchas
///
/// The macro lowers the kernel body to wasm32, runs the
/// WASM-route translator, and emits per-backend shader source.
/// rustc + LLVM optimise the wasm output aggressively; one
/// pattern is worth flagging:
///
/// - **`bool == bool` can constant-fold to `true`.** When the
///   body unrolls a loop containing `let cond = a_bool ==
///   b_bool; if cond { ... } else { ... }`, LLVM has been seen
///   to fold the equality into a tautology and discard the
///   then-branch. Encode boolean equality as `u32` 0/1
///   comparison when the result feeds a branch:
///   `let a = if pred_a { 1u32 } else { 0u32 }; ... if a == b
///   { ... }`. The bug surfaced in quanta-prims's bitonic sort;
///   the u32 encoding is the defensive pattern there.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    kernel_macro::expand_kernel(attr, func)
}

/// Internal attribute used by `#[quanta::kernel]` to complete its
/// expansion after the qualified-call body rewriter and the sibling
/// `_src!()` invocations have run. NOT a public API — `quanta::kernel`
/// emits this on its rewritten output.
#[proc_macro_attribute]
#[doc(hidden)]
pub fn __kernel_inner(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    kernel_macro::expand_kernel_core(attr, func)
}

/// Mark a function as a GPU device function — callable from
/// `#[quanta::kernel]` bodies. The function is also emitted unchanged
/// for plain CPU use (host-side reference, tests, doctests).
///
/// ```ignore
/// #[quanta::device]
/// fn splitmix32(mut x: u32) -> u32 {
///     x = x.wrapping_add(0x9E3779B9);
///     x = (x ^ (x >> 16)).wrapping_mul(0x85EBCA6B);
///     x = (x ^ (x >> 13)).wrapping_mul(0xC2B2AE35);
///     x ^ (x >> 16)
/// }
///
/// #[quanta::kernel]
/// fn fill(d: &MyData) {
///     let id = quark_id();
///     d.out[id as usize] = splitmix32(d.seed ^ id);
/// }
/// ```
///
/// The function's source is captured at attribute-expansion time
/// and later spliced into the temporary wasm-shell crate that
/// `#[quanta::kernel]` hands to rustc, so `call $name` resolves at
/// compile time. At -O3 LLVM typically inlines device functions
/// into the caller before the WASM lowerer sees them.
///
/// Ordering: define device functions *before* the kernels that call
/// them in source order. Device functions may transitively call
/// other device functions; the kernel macro discovers them
/// recursively.
#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    device_macro::expand_device(attr, func)
}

/// Mark a variable as shared (workgroup-local) memory inside a kernel.
///
/// ```ignore
/// #[quanta::kernel]
/// fn reduce(data: &[f32], result: &mut [f32]) {
///     #[quanta::shared] let local: [f32; 256];
///     let lid = proton_id();
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

// === Derive macros ===

/// Derive GPU uniform buffer metadata from a struct's fields.
///
/// Generates `GpuType` impl, byte-level field metadata (`GPU_SIZE`, `GPU_FIELDS`),
/// and MSL/WGSL struct declarations. The struct must have `#[repr(C)]`.
///
/// ```ignore
/// #[repr(C)]
/// #[derive(Copy, Clone, quanta::Uniforms)]
/// struct Camera {
///     view: [f32; 16],     // mat4x4
///     proj: [f32; 16],     // mat4x4
///     eye_pos: [f32; 3],   // vec3
///     fov: f32,
/// }
///
/// // Generated:
/// // Camera::GPU_SIZE, Camera::GPU_FIELDS
/// // impl GpuType for Camera
/// // __QUANTA_UNIFORMS_CAMERA (MSL)
/// // __QUANTA_UNIFORMS_CAMERA_WGSL (WGSL)
/// ```
#[proc_macro_derive(Uniforms)]
pub fn derive_uniforms(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemStruct);
    match uniforms_derive::expand_uniforms_derive(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Derive GPU dispatch field metadata from a struct's fields.
///
/// Classifies each field as either a GPU storage buffer (`Vec<T>`) or a
/// push constant (scalar). Generates metadata used by `#[quanta::kernel]`
/// to auto-generate upload/bind/dispatch/readback code.
///
/// ```ignore
/// #[derive(quanta::Fields)]
/// struct Particles {
///     pos: Vec<f32>,   // GPU field (storage buffer), slot 0
///     vel: Vec<f32>,   // GPU field (storage buffer), slot 1
///     count: u32,      // Push constant, slot 0
///     dt: f32,         // Push constant, slot 1
/// }
///
/// // Generated:
/// // Particles::FIELD_COUNT = 2
/// // Particles::PUSH_CONSTANT_COUNT = 2
/// // Particles::field_names() -> &["pos", "vel"]
/// // Particles::field_types() -> &["f32", "f32"]
/// // Particles::push_constant_names() -> &["count", "dt"]
/// // Particles::push_constant_types() -> &["u32", "f32"]
/// ```
#[proc_macro_derive(Fields)]
pub fn derive_fields(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemStruct);
    match fields_derive::expand_fields_derive(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Import one or more `#[quanta::device]` functions from another
/// crate, splicing their source into the current crate's macro
/// process so a `#[quanta::kernel]` body can call them by bare name.
///
/// ```ignore
/// quanta::import_devices!(
///     quanta_rand::philox4x32_10_first_u32_kernel,
///     quanta_rand::threefry4x32_20_first_u32_kernel,
/// );
///
/// #[quanta::kernel]
/// fn my_kernel(d: &MyData) {
///     let r = philox4x32_10_first_u32_kernel(/* … */);
/// }
/// ```
///
/// Each path is rewritten to append `_src` to its final segment and
/// emitted as `<path>_src!();`. The `_src!` macro is auto-generated
/// by `#[quanta::device]` on the library side — see that attribute's
/// documentation for the mechanism.
#[proc_macro]
pub fn import_devices(input: TokenStream) -> TokenStream {
    use proc_macro2::TokenStream as TokenStream2;
    use quote::quote;
    use syn::{
        Path, Token,
        parse::{Parse, ParseStream},
        punctuated::Punctuated,
    };

    struct ImportList(Punctuated<Path, Token![,]>);

    impl Parse for ImportList {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(ImportList(Punctuated::parse_terminated(input)?))
        }
    }

    let paths = parse_macro_input!(input as ImportList);

    let calls: Vec<TokenStream2> = paths
        .0
        .into_iter()
        .map(|mut path| {
            // Append `_src` to the last segment's ident.
            let last_idx = path.segments.len() - 1;
            let last = &mut path.segments[last_idx];
            let new_name = format!("{}_src", last.ident);
            last.ident = syn::Ident::new(&new_name, last.ident.span());
            quote! { #path!(); }
        })
        .collect();

    let expanded = quote! {
        #(#calls)*
    };
    expanded.into()
}

//! Compute-face proc macros for Quanta GPU kernels.
//!
//! The `#[quanta::kernel]` / `#[quanta::device]` / `#[quanta::shared]`
//! attribute macros, `import_devices!`, `#[quanta::gpu_type]`, and the
//! `Fields` / `Uniforms` derives, plus the rustc → wasm32 → KernelOps
//! lowering pipeline behind them. Compiler-binary invocation and the
//! shared emitters live in `quanta-dsl-core`.

#![deny(missing_docs)]

extern crate proc_macro;

mod auto_dispatch;
mod compile_via_wasm;
mod crate_path;
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

/// Compile a Rust function into a GPU compute kernel.
///
/// The body is compiled to wasm32 by rustc, translated to Quanta IR,
/// and emitted as per-backend artifacts at build time. The attribute
/// replaces the function with a *wave constructor* —
/// `fn name(gpu: &Gpu) -> Result<Wave, QuantaError>` — that hands back
/// a `Wave` ready to bind and dispatch.
///
/// ```ignore
/// #[quanta::kernel]                                 // default: O3, workgroup [64, 1, 1]
/// #[quanta::kernel(opt = "O2")]                     // explicit optimization level
/// #[quanta::kernel(opt = "O0")]                     // no optimization (debug)
/// #[quanta::kernel(workgroup = [256])]              // [256, 1, 1]
/// #[quanta::kernel(workgroup = [16, 16])]           // [16, 16, 1]
/// #[quanta::kernel(workgroup = [16, 16, 1])]        // explicit 3D
/// #[quanta::kernel(workgroup = [256], opt = "O2")]  // both
/// #[quanta::kernel(jit)]                            // compile on the device, not at build time
/// #[quanta::kernel(crate = quanta_core)]            // name runtime types through quanta-core
/// fn saxpy(x: &[f32], y: &mut [f32], a: f32) {
///     let i = quark_id() as usize;
///     y[i] = a * x[i] + y[i];
/// }
/// ```
///
/// # Arguments
///
/// - `opt = "O0" | "O1" | "O2" | "O3"` — the optimization level the
///   wasm build runs at. Default `"O3"`.
/// - `workgroup = [x]` / `[x, y]` / `[x, y, z]` — the workgroup shape
///   the wave is created with; omitted trailing dimensions are 1.
///   Default `[64, 1, 1]`. Size any `#[quanta::shared]` array against
///   this.
/// - `jit` — a bare flag. Embed the serialized IR instead of compiled
///   artifacts and compile it on the device when the wave is built.
///   Trades first-use latency for build time.
/// - `crate = <path>` — the crate root the generated code names `Gpu`,
///   `Wave`, `Pulse`, `KernelBinary`, and `QuantaError` through.
///   Default `::quanta`; a crate that depends on the split crates
///   rather than the facade passes `crate = quanta_core`.
///
/// Any other name is a compile error naming the recognised set.
/// `subgroup = N` is recognised only to refuse it: no backend can
/// honor a required subgroup width today, so accepting it would be a
/// silent no-op — read the device's real width with `subgroup_size()`
/// inside the body instead.
///
/// # Parameters
///
/// Each parameter takes one binding slot, numbered by declaration
/// order: `&[T]` is a read-only buffer, `&mut [T]` a read-write
/// buffer, a bare scalar a push constant, and `&Sampled2D<f32>` /
/// `&Texture2D<T>` / `&mut Texture2D<T>` the texture forms. A kernel
/// must return `()`, may not be `async` or `unsafe`, and may take
/// const generic parameters but no type or lifetime parameters.
///
/// A lone `&MyStruct` parameter whose type derives [`Fields`] is the
/// *struct-ref* form. The macro scans the body for `p.field[i]`
/// (buffer) and `p.field` (push constant) accesses and emits, beside
/// the wave constructor — renamed `name_wave` — a one-call
/// `fn name(gpu: &Gpu, data: &mut MyStruct, quarks: u32) ->
/// Result<Pulse, QuantaError>` that allocates, uploads, binds,
/// dispatches, waits, and reads the written fields back into `data`.
///
/// The kernel's own `///` lines carry onto whichever items the macro
/// emits in its place, so a crate under `#![deny(missing_docs)]` sees
/// the author's documentation rather than a generated item.
///
/// The parameter, texture-format, and in-kernel intrinsic tables live
/// in `docs/reference/macros.md`; which Rust forms are guaranteed to
/// lower is `docs/reference/kernel-lowering.md`.
///
/// # Kernel-body gotchas
///
/// rustc and LLVM optimize the wasm output aggressively; one pattern
/// is worth flagging:
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
/// `#[quanta::kernel]` bodies. The function is also emitted unchanged,
/// so plain CPU code (a host-side reference, a test, a doctest) can
/// still call it.
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
/// The source text is captured as the attribute expands and spliced
/// into the temporary wasm-shell crate `#[quanta::kernel]` hands to
/// rustc, so the call resolves during the kernel build. At `-O3` LLVM
/// typically inlines device functions into the caller before the
/// lowerer ever sees the call.
///
/// Ordering: define device functions *before* the kernels that call
/// them, in source order — the capture happens attribute by attribute,
/// top to bottom, and a kernel that expands first sees nothing. Device
/// functions may call other device functions; the kernel macro follows
/// the chain recursively.
///
/// # Calling one from another crate
///
/// The attribute also emits `macro_rules! <name>_src` at the defining
/// crate's root, which replays the definition into a downstream crate's
/// macro process. Reach it either by calling the function through its
/// full path in a kernel body (`#[quanta::kernel]` rewrites the
/// qualified call and emits the `_src!()` invocation for you) or by
/// listing it in [`import_devices!`].
///
/// # Arguments
///
/// - `crate = <path>` — the crate root the emitted `_src!` macro names
///   the host-side intrinsic stubs and its own re-registration through.
///   Default `::quanta`; pass the same value as the kernels that call
///   this function.
///
/// `register_only` is an internal marker the generated `_src!` macro
/// passes back to this attribute; never write it by hand.
#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    device_macro::expand_device(attr, func)
}

/// Declare workgroup-local (shared) memory inside a kernel body.
///
/// ```ignore
/// #[quanta::kernel(workgroup = [256])]
/// fn reduce(data: &[f32], out: &mut [f32]) {
///     #[quanta::shared] let scratch: [f32; 256];
///
///     let lane = proton_id();
///     scratch[lane] = data[quark_id() as usize];
///     barrier();
///     // … tree-reduce through `scratch`, then one lane writes `out` …
/// }
/// ```
///
/// The declaration is a `let` with a fixed-size array type and no
/// initializer. One allocation is shared by every quark in the
/// workgroup, so size it against the kernel's `workgroup` attribute.
/// Indexing it — `scratch[i]`, `scratch[i] = v` — lowers to the
/// `SharedDecl` / `SharedLoad` / `SharedStore` IR ops directly; no
/// intrinsic call is involved, and a `barrier()` is what makes another
/// lane's store visible.
///
/// The proc macro itself is a pass-through. `#[quanta::kernel]` reads
/// the attribute off the `let` statement as it walks the body, so the
/// attribute means nothing anywhere else.
#[proc_macro_attribute]
pub fn shared(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// Mark a struct as GPU-compatible — the element type a
/// `gpu.field::<T>(n)` storage buffer can hold.
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
/// The struct is re-emitted with `#[repr(C)]` and `#[derive(Copy,
/// Clone)]` — each added only when absent — so the Rust layout matches
/// the shader-side one, and gains:
///
/// - `Particle::GPU_SIZE` — the struct's byte size
/// - `Particle::GPU_FIELDS` — one `(name, type, byte offset)` per
///   field, offsets computed under `repr(C)` rules
/// - `impl GpuType for Particle` — what `gpu.field::<Particle>(n)` needs
/// - `__QUANTA_GPU_TYPE_PARTICLE` and `__QUANTA_GPU_TYPE_PARTICLE_WGSL`
///   — the MSL and WGSL declarations of the same struct, for shader
///   sources that name the type
///
/// Only named-field structs are accepted. Each field must be a scalar,
/// a fixed-size array of one (array lengths must be literals — no const
/// generics), or another GPU struct; there is no heap type on the GPU
/// side. A nested-struct field reaches the MSL and WGSL declarations by
/// name, but counts as zero bytes in the offset arithmetic, so every
/// `GPU_FIELDS` offset after one is wrong — keep the struct flat, or
/// take offsets from `core::mem::offset_of!` instead.
///
/// # Arguments
///
/// - `crate = <path>` — the crate root the generated `GpuType` impl is
///   written against. Default `::quanta`.
///
/// For a uniform-buffer struct prefer [`Uniforms`], whose declarations
/// are shaped for uniform binding; this attribute is the storage-buffer
/// element form.
#[proc_macro_attribute]
pub fn gpu_type(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemStruct);
    let cp = crate_path::from_attr_args(attr);
    match gpu_type::expand_gpu_type(&input, &cp) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// === Derive macros ===

/// Derive uniform-buffer metadata for a struct — layout, shader
/// declarations, and the `GpuType` impl that lets it back a uniform
/// binding.
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
/// ```
///
/// Generates:
///
/// - `Camera::GPU_SIZE` — the struct's byte size
/// - `Camera::GPU_FIELDS` — one `(name, type, byte offset)` per field
/// - `impl GpuType for Camera`
/// - `__QUANTA_UNIFORMS_CAMERA` and `__QUANTA_UNIFORMS_CAMERA_WGSL` —
///   the MSL and WGSL declarations of the same struct
///
/// Offsets are computed under C layout rules, and the derive does not
/// verify that the struct carries `#[repr(C)]` — add it yourself, or
/// the metadata will not describe the bytes actually uploaded.
///
/// The container attribute `#[quanta(crate = <path>)]` overrides the
/// crate root the generated `GpuType` impl is written against (serde's
/// `#[serde(crate = "...")]` pattern). Default is `::quanta`; companion
/// crates that host kernels without depending on the facade pass
/// `#[quanta(crate = quanta_core)]`.
#[proc_macro_derive(Uniforms, attributes(quanta))]
pub fn derive_uniforms(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemStruct);
    match uniforms_derive::expand_uniforms_derive(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Derive dispatch metadata for a kernel data struct — which fields are
/// GPU storage buffers and which are push constants.
///
/// A `Vec<T>` field is a storage buffer; any scalar field is a push
/// constant. This is what makes a struct usable as the single
/// `&MyStruct` parameter of a [`kernel`], where the macro reads the
/// classification to generate the upload / bind / dispatch / readback
/// wrapper.
///
/// ```ignore
/// #[derive(quanta::Fields)]
/// struct Particles {
///     pos: Vec<f32>,   // storage buffer
///     vel: Vec<f32>,   // storage buffer
///     count: u32,      // push constant
///     dt: f32,         // push constant
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
///
/// Each list keeps declaration order, and the two are numbered
/// independently — the binding slots a kernel ends up using are
/// assigned by [`kernel`] from the fields the body actually touches,
/// not by position in the struct.
// `attributes(quanta)` so a struct that derives both `Fields` and
// `Uniforms` may carry one `#[quanta(crate = ...)]` container attribute
// without `Fields` rejecting it as unknown. `Fields` emits no
// `::quanta::` path itself (only plain metadata consts), so it ignores
// the attribute's value.
#[proc_macro_derive(Fields, attributes(quanta))]
pub fn derive_fields(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemStruct);
    match fields_derive::expand_fields_derive(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Import one or more [`device`] functions from another crate, so a
/// [`kernel`] body in this one can call them by bare name.
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
/// Invoke it at file scope, above the kernels that need the functions —
/// the import registers each definition in this crate's macro process,
/// and a kernel that expands first sees nothing. Each path is rewritten
/// by appending `_src` to its final segment and invoked as
/// `<path>_src!()`; those macros are the ones [`device`] exports on the
/// library side.
///
/// Calling a device function through its full path inside the kernel
/// body has the same effect without the list — reach for this macro
/// when you want to write the bare name.
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
